"""afd adapter for the split MegaMoE M2N mega kernels.

Selected with ``MINISGL_AFD_MOE_BACKEND=megamoe_m2n``.  Replaces the
DeepEP union adapter's per-layer dispatch/expert/combine sequence:

- AG ranks: :meth:`ag_moe` is ONE fused call (and one kernel) that quantizes
  the hidden states to FP8 (per-32 packed UE8M0), pushes route metadata to the
  EG owner ranks, waits for the expert write-back, and reduces top-k into a
  bf16 output.
- EG ranks: :meth:`eg_moe` is ONE persistent kernel that pulls FP8 tokens from
  the AG ranks, runs the fused FP8xFP8 L1/L2 grouped GEMM pipeline (SwiGLU and
  top-k weights fused in the L1 epilogue), and pushes BF16 expert outputs into
  the AG combine buffers.  Qwen FP8 checkpoint weights are requantized once at
  init to the per-32 UE8M0 mega format (:meth:`register_eg_layer_weights`).

Topology contract matches ``DeepEPM2NAdapter``: one union world, AG ranks
first; EG rank ``g`` owns experts ``[g*E_local, (g+1)*E_local)``; ``topk_ids``
are REAL model expert ids with ``-1`` for dummy routes.

Constraints (from the mega kernel): ``hidden % 512 == 0`` and
``moe_intermediate_size % 512 == 0`` (e.g. Qwen3-235B 4096/1536 OK,
Qwen3-30B's 768 intermediate is NOT supported yet).
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist

from minisgl.kernel import megamoe_m2n_mega as _mm
from minisgl.layers.moe.moe_runner.deepgemm_grouped import (
    _requant_channel_weight_ue8m0_inplace,
)


_EG_PRECOMPILE_ENV = "MINISGL_MEGAMOE_EG_PRECOMPILE"


def eg_precompile_enabled() -> bool:
    """Whether an EG worker builds every EG kernel instance before its first step.

    On by default: a first-use JIT compile inside a step holds the EG host for
    seconds while every AG rank has already launched its half of that step's
    M2N collective. ``MINISGL_MEGAMOE_EG_PRECOMPILE=0`` restores
    compile-on-first-use.
    """
    value = os.environ.get(_EG_PRECOMPILE_ENV, "1")
    if value not in ("0", "1"):
        raise RuntimeError(
            f"{_EG_PRECOMPILE_ENV} must be 0 (compile EG kernels on first use) "
            f"or 1 (build them all at startup, the default), got {value!r}"
        )
    return value == "1"


# EG early start off: one pass over every source
EG_EARLY_OFF = MappingProxyType(dict(early_max_late=0, early_wait_cycles=0, early_policy=0,
                                     early_pass_cycles=0, early_expert_cycles=0, early_margin_pct=0))


def parse_eg_early(spec: str) -> dict[str, int]:
    """The EG early-start launch arguments for a MINISGL_MEGAMOE_EG_EARLY value.

    Early start serves a round whose last <= K source ranks are late in two passes,
    the arrived sources first, so their FFN work hides behind the late ones'
    attention (P2P only; "" keeps one pass):
      "K:US"   split once the late ones are still missing US microseconds later;
      "K:auto[:MARGIN_PCT[:PASS_US[:EXPERT_US]]]"  split right away when the late
               ones are expected to come no sooner than the first pass ends
               (+MARGIN_PCT, default 15). The kernel prices a pass at PASS_US
               (default 10) plus a per-expert time it learns per lane (starting at
               EXPERT_US, default 1.2), and learns how late the late sources come;
               otherwise one pass.
    """
    out = dict(EG_EARLY_OFF)
    if not spec:
        return out
    # clock64() counts SM cycles; ~1.9 GHz on SM100 under load
    cycles = lambda us: int(float(us) * 1900)  # noqa: E731
    k, sep, rest = spec.partition(":")
    if not sep or int(k) <= 0:
        raise ValueError(f"MINISGL_MEGAMOE_EG_EARLY={spec!r}: expected K:US or K:auto[...] with K >= 1")
    out["early_max_late"] = int(k)
    fields = rest.split(":")
    if fields[0] == "auto":
        margin, pass_us, expert_us = (fields[1:] + [""] * 3)[:3]
        out.update(
            early_policy=1,
            early_margin_pct=int(margin) if margin else 15,
            early_pass_cycles=cycles(pass_us or 10),
            early_expert_cycles=cycles(expert_us or 1.2),
        )
    else:
        out["early_wait_cycles"] = cycles(rest)
    return out


def check_eg_layer_block(global_to_compact: dict[int, int], layer_base: int, first_compact: int = 0) -> None:
    """P2P's layer-numbering guard for one EG launch.

    The EG launches number layers by registration (compact ids), the AG ranks by
    model layer, and a launch's layer i is model layer ``layer_base + i`` (the
    combine counters are indexed by its parity). So the registered layers must be
    one contiguous block of model layers -- every one an MoE layer -- and the
    launch's first layer, compact ``first_compact``, must be model layer
    ``layer_base``: global == compact - first_compact + layer_base for all.
    """
    offset = int(layer_base) - int(first_compact)
    bad = sorted(g for g, c in global_to_compact.items() if g != c + offset)
    if bad:
        raise RuntimeError(
            "megamoe_m2n P2P needs every model layer of the EG block to be an MoE layer, "
            f"registered as one contiguous block from model layer {offset}: registered layers "
            f"{sorted(global_to_compact)}, launch layer_base={int(layer_base)} at compact "
            f"layer {int(first_compact)}, off the block: {bad}"
        )


class MegaMoEM2NAfdAdapter:
    backend = "megamoe_m2n"

    # Protocol switches, all off: __init__ sets them from the environment
    # (MINISGL_MEGAMOE_P2P, _AG_SPLIT, _EG_EARLY, _PUSH_DISPATCH, ...), and an adapter
    # built without it keeps the barrier protocol and the fused AG kernel.
    p2p = False
    ag_split = "fused"
    push_dispatch = False
    ag_waited_combine = False
    ag_wait_mono = False
    eg_early = EG_EARLY_OFF

    def __init__(
        self,
        *,
        group: Any,
        ag_size: int,
        eg_size: int,
        real_num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        num_max_dispatch_tokens_per_rank: int,
        num_lanes: int = 1,
        gate_renormalize: bool = True,
        shared_experts_per_rank: int = 0,
        routed_scaling_factor: float = 1.0,
        p2p: bool = False,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("MegaMoEM2NAfdAdapter requires torch.distributed")
        # Point-to-point protocol instead of the union-wide barriers (every rank of
        # the union must agree; MINISGL_MEGAMOE_P2P=1 turns it on): AG ranks wait
        # only for the EG ranks, never for each other. Same rows as the barriers.
        self.p2p = bool(p2p) or os.environ.get("MINISGL_MEGAMOE_P2P") == "1"
        if hidden_size % 512 != 0 or intermediate_size % 512 != 0:
            raise RuntimeError(
                "megamoe_m2n requires hidden and moe intermediate sizes divisible "
                f"by 512, got hidden={hidden_size} intermediate={intermediate_size}"
            )
        self.group = group
        self.ag_size = int(ag_size)
        self.eg_size = int(eg_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.routed_top_k = int(top_k)
        self.shared_experts_per_rank = int(shared_experts_per_rank)
        if self.shared_experts_per_rank not in (0, 1):
            raise RuntimeError(
                "megamoe_m2n currently supports at most one replicated shared "
                f"expert per EG rank, got {self.shared_experts_per_rank}"
            )
        self.top_k = self.routed_top_k + self.shared_experts_per_rank
        self.real_num_experts = int(real_num_experts)
        self.num_experts = self.real_num_experts + self.eg_size * self.shared_experts_per_rank
        if self.num_experts % self.eg_size != 0:
            raise RuntimeError(
                f"protocol expert count must divide eg_size: experts={self.num_experts} "
                f"eg_size={self.eg_size}"
            )
        self.bucket = int(num_max_dispatch_tokens_per_rank)
        self.num_lanes = max(1, int(num_lanes))
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.buffers = [
            _mm.MegaMoEM2NSymmBuffer(
                group,
                ag_size=self.ag_size,
                eg_size=self.eg_size,
                num_experts=self.num_experts,
                num_max_tokens_per_rank=self.bucket,
                num_topk=self.top_k,
                hidden=self.hidden_size,
                intermediate_hidden=self.intermediate_size,
            )
            for _ in range(self.num_lanes)
        ]
        # This rank's union position, the `rank` of every launch: AG positions come
        # first, EG position ag_size + g owns the g-th expert slice.
        self.rank = int(self.buffers[0].position)
        assert all(int(buf.position) == self.rank for buf in self.buffers), "lanes disagree on the union position"
        self.is_ag = self.rank < self.ag_size
        self.is_eg = not self.is_ag
        self._y_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._eager_buffers: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}
        self._eg_weights: dict[int, tuple] = {}
        self._eg_desc_blobs: dict[int, torch.Tensor] = {}
        self._eg_l1_ptrs: dict[int, int] = {}
        self._eg_global_to_compact: dict[int, int] = {}
        self._eg_l1_nbytes = 0
        # Per-layer [E_local, L1 + L2 blocks] block scale residuals (exact block scales)
        self._eg_block_scale_residuals: dict[int, torch.Tensor] = {}
        self._block_scale_residual_ptrs_cache: torch.Tensor | None = None
        self._desc_blob_all = None
        self._l1_ptrs_all = None
        self._destroyed = False

        # Grid sizing. AG leaves SMs free for attention-side compute. EG runs on
        # dedicated expert ranks, so it defaults to full-device kernels.
        self.ag_num_sms = int(os.environ.get("MINISGL_MEGAMOE_AG_SMS", "24"))
        # How the AG side waits for the EG round trip (MINISGL_MEGAMOE_AG_SPLIT):
        #   fused     one kernel dispatches, spins on its SMs, and reduces;
        #   spin      dispatch kernel, then the combine kernel right behind it on
        #             the lane stream, which spins on ITS SMs (the split by itself);
        #   waitvalue dispatch kernel, the lane stream waits on the combine counter
        #             (cuStreamWaitValue32: the GPU front end, no SM), combine kernel;
        #   waitcta   dispatch kernel, a one-CTA kernel spins on the counter (one
        #             SM), combine kernel.
        # Split launches size their grids separately: neither holds SMs through
        # the round trip, so both can take more than the fused kernel's budget.
        self.ag_split = os.environ.get("MINISGL_MEGAMOE_AG_SPLIT", "fused")
        if self.ag_split not in ("fused", "spin", "waitvalue", "waitcta"):
            raise ValueError(f"unknown MINISGL_MEGAMOE_AG_SPLIT {self.ag_split!r}")
        if self.ag_split in ("waitvalue", "waitcta") and not self.p2p:
            raise ValueError("MINISGL_MEGAMOE_AG_SPLIT waits on the P2P combine counter; enable P2P")
        # EG early start (P2P), MINISGL_MEGAMOE_EG_EARLY: see parse_eg_early
        early = os.environ.get("MINISGL_MEGAMOE_EG_EARLY", "")
        if early and not self.p2p:
            raise ValueError("MINISGL_MEGAMOE_EG_EARLY needs the P2P protocol")
        self.eg_early = parse_eg_early(early)
        # MINISGL_MEGAMOE_PUSH_DISPATCH=1: the AG quant also writes each token's rows
        # into every EG rank's staging rows, and the EG reads them locally instead
        # of pulling them from AG ranks whose HBM is busy with attention
        self.push_dispatch = os.environ.get("MINISGL_MEGAMOE_PUSH_DISPATCH") == "1"
        # MINISGL_MEGAMOE_EXACT_FP8_SCALES=1: FP8 experts keep their 128x128 block
        # scales (the EG applies each K block's scale residual) instead of being
        # requantized to UE8M0 scales, which rounds every weight a second time
        self.exact_block_scales = os.environ.get("MINISGL_MEGAMOE_EXACT_FP8_SCALES") == "1"
        # MINISGL_MEGAMOE_AG_WAITED_COMBINE=1 (waitcta): the wait kernel also clears
        # the combine counter, and the combine launch skips its own counter wait
        # and grid sync
        self.ag_waited_combine = (os.environ.get("MINISGL_MEGAMOE_AG_WAITED_COMBINE") == "1"
                                  and self.ag_split == "waitcta")
        # MINISGL_MEGAMOE_AG_WAIT_MONO=1 (waitcta): the one-CTA wait treats the combine
        # counter as monotonic (a consumed word of this rank instead of a system-scope
        # subtraction), and the combine launch skips its own counter wait and grid sync
        self.ag_wait_mono = (os.environ.get("MINISGL_MEGAMOE_AG_WAIT_MONO") == "1"
                             and self.ag_split == "waitcta")
        self.ag_dispatch_sms = int(os.environ.get("MINISGL_MEGAMOE_AG_SMS_DISPATCH", self.ag_num_sms))
        self.ag_combine_sms = int(os.environ.get("MINISGL_MEGAMOE_AG_SMS_COMBINE", self.ag_num_sms))
        self.eg_num_sms = 0
        is_minimax_m25_shape = (
            self.hidden_size == 3072
            and self.intermediate_size == 1536
            and self.real_num_experts == 256
            and self.routed_top_k == 8
            and self.shared_experts_per_rank == 0
        )
        # Optional override for the GEMM block/wave heuristic input.  GLM's
        # remote shared expert injects a replicated 9th route: routed experts
        # are sparse, but the shared expert sees one route per token from a
        # subset of AG ranks.  A slightly larger decode-scale heuristic keeps
        # the mixed routed/shared persistent kernel on the faster block_m=32
        # tiling.  Set MINISGL_MEGAMOE_EG_EXPECTED_TOKENS=0 to force the raw
        # per-call bucket-derived heuristic.  MiniMax M2.5 decode has far more
        # routed experts than Qwen; a smaller expected-token hint keeps the
        # persistent EG kernel on the fast decode tiling.
        eg_expected_env = os.environ.get("MINISGL_MEGAMOE_EG_EXPECTED_TOKENS")
        glm_remote_shared = bool(self.shared_experts_per_rank)
        self.eg_expected_tokens_override = (
            int(eg_expected_env)
            if eg_expected_env is not None
            else (
                24
                if is_minimax_m25_shape
                else (16 if glm_remote_shared else 0)
            )
        )
        self.eg_expected_floor = os.environ.get("MINISGL_MEGAMOE_EG_EXPECTED_FLOOR") == "1"
        self.eg_expected_tokens_source = (
            "env" if eg_expected_env is not None
            else (
                "minimax_m25_default"
                if is_minimax_m25_shape
                else (
                    "remote_shared_default"
                    if glm_remote_shared
                    else "bucket"
                )
            )
        )
        # L2 prefetch of the head of each layer's L1 weights, evict-last while lane 0
        # waits for counts (a launch argument). Off by default: the FP8 EG GEMM already
        # streams the weights near the HBM peak and the prefetch only competes.
        # MINISGL_MEGAMOE_EG_PREFETCH_MB sets it.
        self.eg_prefetch_bytes = int(os.environ.get("MINISGL_MEGAMOE_EG_PREFETCH_MB") or 0) << 20
        self.gate_renormalize = bool(gate_renormalize)

    @property
    def role(self) -> str:
        return "ag" if self.is_ag else "eg"

    def barrier_head(self, lane: int = 0) -> tuple[int, int, int]:
        """`(nvlink counter, phase-0 signal, phase-1 signal)` for one lane's buffer.

        The M2N kernels chain ACROSS launches: each layer's tag-3 cleanup barrier
        is split, `arrive` signalling at the end of one launch and the matching
        `wait` being consumed by the next (`comm/barrier.cuh:83-90`,
        `sm100_mega_moe_m2n.cuh:357-368`). The union's only agreement protocol is
        `counter & 3` (`comm/barrier.cuh:53-55`), and the words are seeded exactly
        once at construction (`megamoe_m2n_mega.py:85-91`) and never reset. So a
        slip of a multiple of 4 leaves `status` identical on every rank and the
        barrier passes SILENTLY on a stale round instead of hitting the 300 s
        timeout. Reading the counter is the only way to see that.

        Debug use only. This reads device memory the collective writes, so the
        caller must be at a step boundary and must synchronize first.
        """
        head = self.buffers[lane % self.num_lanes].buffer[16:28].view(torch.int32)
        counter, signal0, signal1 = (int(x) for x in head.tolist())
        return counter, signal0, signal1

    # -- AG side -----------------------------------------------------------------
    def ag_moe(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        lane: int = 0,
        external_quant: bool = False,
        layer: int | None = None,
    ):
        """Fused MoE layer on an AG rank: quantize + dispatch + combine.

        Under P2P ``layer`` (the model layer) is required: its parity indexes the
        combine counter, as the EG launches' ``layer_base`` does.
        """
        if not self.is_ag:
            raise RuntimeError("ag_moe called on an EG rank")
        if self.p2p and layer is None:
            raise RuntimeError("megamoe_m2n P2P needs the layer index (counter parity)")
        buf = self.buffers[lane % self.num_lanes]
        n = int(hidden_states.shape[0])
        if n > self.bucket:
            raise RuntimeError(f"tokens={n} exceeds dispatch bucket={self.bucket}")

        # Per-32 FP8 quant, symm-buffer staging, dispatch, wait, and combine
        # reduce are fused into one AG kernel; the inputs are read in place.
        x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
        if ids.dtype not in (torch.int32, torch.int64):
            ids = ids.long()
        # Diagnostic: the AG megakernel indexes per-expert buffers by these ids and
        # traps on a bad one, and the trap carries no message. Off unless asked for;
        # it syncs, so it must never be on in a measured run.
        if (os.environ.get("MINISGL_MEGAMOE_CHECK_INPUTS") == "1"
                and not torch.cuda.is_current_stream_capturing()):
            x_finite = bool(torch.isfinite(x).all())
            w_finite = bool(torch.isfinite(topk_weights).all())
            valid = ids[ids >= 0]
            lo = int(valid.min()) if valid.numel() else -1
            hi = int(valid.max()) if valid.numel() else -1
            ids_ok = valid.numel() == 0 or (lo >= 0 and hi < self.num_experts)
            # The kernel reads exactly top_k ids and weights per token; a shape the
            # caller got wrong is read out of bounds rather than rejected.
            shape_ok = (
                tuple(ids.shape) == (n, self.top_k)
                and tuple(topk_weights.shape) == (n, self.top_k)
            )
            if not (x_finite and w_finite and ids_ok and shape_ok):
                # Warmup feeds uninitialized dummy tensors, so NaN here is
                # EXPECTED then; report instead of killing the run and let the
                # per-launch output grading below tell the real story.
                print(
                    "[megamoe-check] AG inputs degenerate: "
                    f"x_finite={x_finite} w_finite={w_finite} "
                    f"ids_ok={ids_ok} shape_ok={shape_ok} n={n}",
                    flush=True)
        w = topk_weights
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()
        input_topk = int(ids.shape[1])
        if int(w.shape[1]) != input_topk:
            raise RuntimeError(
                f"topk id/weight shape mismatch: ids={tuple(ids.shape)} weights={tuple(w.shape)}"
            )
        if input_topk not in (self.routed_top_k, self.top_k):
            raise RuntimeError(
                f"invalid topk width for megamoe_m2n: got {input_topk}, "
                f"expected routed={self.routed_top_k} or protocol={self.top_k}"
            )
        shared_weight = 0.0
        if self.shared_experts_per_rank:
            if input_topk == self.top_k:
                shared_weight = 0.0
            else:
                shared_weight = 1.0 / self.routed_scaling_factor

        y = self._lane_buffer(
            "y", self._y_cache, (lane % self.num_lanes, n), n, torch.bfloat16, hidden_states.device
        )
        if os.environ.get("MINISGL_MEGAMOE_LOG_LAUNCH") == "1":
            # Both halves of an M2N pair must be in flight together; AG spins on a
            # 300 s cross-rank barrier for EG. Printing both launch times is the
            # only way to see whether they actually pair up.
            import time as _t
            print(f"[m2n-launch] role=ag rank={self.rank} t={_t.time():.3f} tokens={n}", flush=True)
        timings = self._ag_debug_timings()
        # With CHECK_INPUTS on, also grade the OUTPUT per layer: where the
        # all-zero (or non-finite) stream first appears localizes the fault to
        # attention (bad x in), this MoE (bad y out of good x), or sampling.
        _check_out = (os.environ.get("MINISGL_MEGAMOE_CHECK_INPUTS") == "1"
                      and not torch.cuda.is_current_stream_capturing())
        parity = (int(layer) & 1) if layer is not None else 0

        def launch(phase: int, num_sms: int, waited: bool = False) -> None:
            _mm.mega_moe_m2n_ag(y, x, ids, w, buf, rank=self.rank,
                                num_sms=num_sms,
                                renormalize=self.gate_renormalize,
                                phase=phase,
                                external_quant=bool(external_quant),
                                direct_combine=False,
                                input_num_topk=input_topk,
                                num_routed_experts=self.real_num_experts,
                                shared_topk_weight=shared_weight,
                                debug_timings=timings,
                                p2p=self.p2p,
                                layer_parity=parity,
                                push_dispatch=self.push_dispatch,
                                waited_combine=waited)

        if self.ag_split == "fused":
            launch(0, self.ag_num_sms)
        else:
            # Dispatch, then (in this stream's order) the combine: phase 2 must
            # follow its phase 1 before the next phase 1 on this lane's buffer.
            launch(1, self.ag_dispatch_sms)
            if self.ag_split == "waitvalue":
                _mm.stream_wait_ge(buf.combine_count_view(parity), self.eg_size + 1)
            elif self.ag_split == "waitcta":
                tag = (self.rank << 24) | (int(layer or 0) << 8) | (int(lane) << 1) | parity
                if self.ag_wait_mono:
                    _mm.wait_ge_cta_mono(buf.combine_count_view(parity), buf.combine_consumed_view(parity),
                                         self.eg_size + 1, tag=tag)
                else:
                    _mm.wait_ge_cta(buf.combine_count_view(parity), self.eg_size + 1,
                                    reset=self.ag_waited_combine, tag=tag)
            launch(2, self.ag_combine_sms, waited=self.ag_waited_combine or self.ag_wait_mono)
        if _check_out:
            torch.cuda.synchronize()
            zf = float((y == 0).float().mean())
            print(f"[megamoe-check] ag_moe out: finite={bool(torch.isfinite(y).all())} "
                  f"absmax={float(y.float().abs().max()):.3e} zero_frac={zf:.3f} "
                  f"x_absmax={float(x.float().abs().max()):.3e}", flush=True)
        return y

    def _lane_buffer(
        self,
        tag: str,
        cache: dict,
        key: tuple,
        n: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """A [n, hidden] buffer the AG kernel uses in place, private to one lane.

        A buffer used while capturing a CUDA graph must keep its address for every
        replay, so it is cached per key. Eager calls (prefill, warmup) see arbitrary
        token counts, and caching one buffer per count grew without bound: on
        Qwen3.5-397B the AG gained ~16 GiB over a long prefill phase and ran out of
        memory. They instead share one bucket-sized buffer per lane and get a
        prefix slice of it.
        """
        buf = cache.get(key)
        if buf is not None:
            return buf
        if (
            torch.device(device).type == "cuda" and torch.cuda.is_current_stream_capturing()
        ) or n > self.bucket:
            buf = torch.empty((n, self.hidden_size), dtype=dtype, device=device)
            cache[key] = buf
            return buf
        eager_key = (tag, int(key[0]), dtype)
        buf = self._eager_buffers.get(eager_key)
        if buf is None:
            buf = torch.empty((self.bucket, self.hidden_size), dtype=dtype, device=device)
            self._eager_buffers[eager_key] = buf
        return buf[:n]

    # Slot names from AGTimingSlot in sm100_mega_moe_m2n.cuh. Only the ones that
    # say how far the kernel got are listed; a written CombineBarrierStartTime
    # with an unwritten CombineBarrierEndTime means it entered the cross-rank
    # barrier and never came out.
    _AG_TIMING_SLOTS = {
        0: "KernelStart",
        7: "DispatchGridSync",
        8: "DispatchCountPush",
        9: "CombineBarrierWait",
        10: "CombineReduce",
        12: "KernelEnd",
        24: "CombineBarrierStartTime",
        25: "CombineBarrierEndTime",
        27: "CombineReduceStartTime",
        28: "CombineReduceEndTime",
    }

    def _ag_debug_timings(self) -> torch.Tensor | None:
        """A pinned progress buffer for the AG kernel, when asked for.

        Pinned rather than device memory on purpose: this exists to explain a
        kernel that TRAPS, and a trap makes the context sticky so a device
        buffer can never be copied back afterwards. Pinned host memory is
        device-writable under unified addressing and outlives the context.
        It costs a slow uncached store per timing point, so it is diagnostic
        only and must never be enabled in a measured run.
        """
        if os.environ.get("MINISGL_MEGAMOE_AG_TIMING") != "1":
            return None
        if getattr(self, "_ag_timings", None) is None:
            self._ag_timings = torch.zeros(29, dtype=torch.int64).pin_memory()
        self._ag_timings.zero_()
        return self._ag_timings

    def report_ag_timings(self) -> str:
        """Human-readable progress record, valid even after the kernel trapped."""
        t = getattr(self, "_ag_timings", None)
        if t is None:
            return "AG timings not enabled"
        return " ".join(
            f"{name}={'--' if int(t[slot]) == 0 else int(t[slot])}"
            for slot, name in sorted(self._AG_TIMING_SLOTS.items())
        )

    def _eg_debug_timings(self, num_layers: int, lanes: int) -> torch.Tensor | None:
        """Per-(layer, lane, pass) EG timing slots, when MINISGL_MEGAMOE_EG_TIMING=1.

        Diagnostic only (selects the kernel instance with kEnableDebugTiming; SM 0's
        leader stores clock64 stamps). The kernel indexes (layer * 2 * lanes +
        2 * lane + pass) * 31 + slot, so the buffer holds two passes per lane.
        """
        if os.environ.get("MINISGL_MEGAMOE_EG_TIMING") != "1":
            return None
        shape = (int(num_layers), 2 * max(int(lanes), 1), 31)
        if getattr(self, "_eg_timings", None) is None or tuple(self._eg_timings.shape) != shape:
            self._eg_timings = torch.zeros(shape, dtype=torch.int64, device="cuda")
            self._eg_timing_stream = torch.cuda.Stream()
        return self._eg_timings

    def dump_eg_timings(self, path: str) -> bool:
        """Save the raw [layer, 2*lane+pass, slot] EG timing buffer (copied on a side stream)."""
        t = getattr(self, "_eg_timings", None)
        if t is None:
            return False
        with torch.cuda.stream(self._eg_timing_stream):
            host = t.to("cpu", non_blocking=True)
        self._eg_timing_stream.synchronize()
        torch.save(host, path)
        return True

    # -- EG side -----------------------------------------------------------------
    def register_eg_layer_weights(
        self,
        layer_id: int,
        experts: Any,
        *,
        shared_experts: Any | None = None,
    ) -> None:
        """Requantize one layer's FP8 expert weights to the mega format.

        ``experts`` holds ``gate_up_proj [E_local, 2I, H]`` (fp8) +
        ``gate_up_proj_scale`` and ``down_proj``/``down_proj_scale``.  Qwen and
        MiniMax use 128x128 block scales; GLM compressed-tensors checkpoints use
        per-output-channel scales and are first repacked to the same block-scale
        FP8 representation.  One-time per layer; the originals can be freed
        afterwards by the caller.
        """
        if not self.is_eg:
            raise RuntimeError("register_eg_layer_weights called on an AG rank")
        quant_method = getattr(getattr(experts, "quant", None), "method", None)
        scale_format = getattr(experts, "_fp8_scale_format", None)
        block_shape = tuple(getattr(getattr(experts, "quant", None), "weight_block_size", (128, 128)))
        if quant_method == "fp8_channel" or scale_format == "channel":
            experts.gate_up_proj_scale = _requant_channel_weight_ue8m0_inplace(
                experts.gate_up_proj, experts.gate_up_proj_scale)
            experts.down_proj_scale = _requant_channel_weight_ue8m0_inplace(
                experts.down_proj, experts.down_proj_scale)
            experts._fp8_scale_format = "block"
        elif quant_method != "fp8" or block_shape != (128, 128):
            raise RuntimeError(
                "megamoe_m2n expert weights must be FP8 with 128x128 block scales "
                f"or GLM-style channel scales, got method={quant_method!r} "
                f"block_shape={block_shape}")
        l1_weight = experts.gate_up_proj
        l1_scale = experts.gate_up_proj_scale
        l2_weight = experts.down_proj
        l2_scale = experts.down_proj_scale
        if self.shared_experts_per_rank:
            if shared_experts is None:
                raise RuntimeError("remote shared expert is enabled but no shared weights were provided")
            shared_l1_weight = shared_experts.gate_up_proj.weight.unsqueeze(0).contiguous()
            shared_l1_scale = shared_experts.gate_up_proj.weight_scale.unsqueeze(0).contiguous()
            shared_l2_weight = shared_experts.down_proj.weight.unsqueeze(0).contiguous()
            shared_l2_scale = shared_experts.down_proj.weight_scale.unsqueeze(0).contiguous()
            shared_scale_format = getattr(shared_experts.gate_up_proj, "_fp8_scale_format", None)
            if shared_scale_format == "channel":
                shared_l1_scale = _requant_channel_weight_ue8m0_inplace(
                    shared_l1_weight, shared_l1_scale)
                shared_l2_scale = _requant_channel_weight_ue8m0_inplace(
                    shared_l2_weight, shared_l2_scale)
            elif tuple(getattr(shared_experts.gate_up_proj, "_fp8_block_size", (128, 128))) != (128, 128):
                raise RuntimeError(
                    "remote shared expert requires 128x128 block scales or "
                    "GLM-style channel scales"
                )
            l1_weight = torch.cat([l1_weight, shared_l1_weight], dim=0).contiguous()
            l1_scale = torch.cat([l1_scale, shared_l1_scale], dim=0).contiguous()
            l2_weight = torch.cat([l2_weight, shared_l2_weight], dim=0).contiguous()
            l2_scale = torch.cat([l2_scale, shared_l2_scale], dim=0).contiguous()
        if self.exact_block_scales:
            l1_weight, l1_sf, l1_residuals = _mm.fp8_weights_with_block_scale_residuals(
                l1_weight, l1_scale)
            l2_weight, l2_sf, l2_residuals = _mm.fp8_weights_with_block_scale_residuals(
                l2_weight, l2_scale)
            self._eg_block_scale_residuals[int(layer_id)] = torch.cat(
                [l1_residuals.flatten(1), l2_residuals.flatten(1)], dim=1).contiguous()
            self._block_scale_residual_ptrs_cache = None
            self._store_eg_layer_weights(layer_id, (l1_weight, l1_sf), (l2_weight, l2_sf))
        else:
            l1 = _mm.requant_qwen_fp8_weights_per32(l1_weight, l1_scale, inplace=True)
            l2 = _mm.requant_qwen_fp8_weights_per32(l2_weight, l2_scale, inplace=True)
            self._store_eg_layer_weights(layer_id, l1, l2)

    def _store_eg_layer_weights(
        self,
        layer_id: int,
        l1: tuple[torch.Tensor, torch.Tensor],
        l2: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Interleave, descriptor-build and register one layer's EG weights."""
        l1_t, l2_t = _mm.transform_weights_for_mega_moe(l1, l2)
        global_layer_id = int(layer_id)
        compact_layer_id = len(self._eg_global_to_compact)
        self._eg_global_to_compact[global_layer_id] = compact_layer_id
        self._eg_weights[compact_layer_id] = (l1_t, l2_t)
        # Per-layer weight tensormap blob for the (persistent) EG kernel
        blob = _mm.make_m2n_weight_descs(
            l1_t, l2_t,
            num_experts_per_rank=self.num_experts // self.eg_size,
            hidden=self.hidden_size,
            intermediate_hidden=self.intermediate_size,
        ).cuda()
        self._eg_desc_blobs[compact_layer_id] = blob
        self._eg_l1_ptrs[compact_layer_id] = int(l1_t[0].data_ptr())
        self._eg_l1_nbytes = int(l1_t[0].numel())
        self._desc_blob_all = None  # invalidate the stacked cache

    def _block_scale_residual_ptrs(self) -> torch.Tensor | None:
        """Device int64 [num_layers] of block scale residual pointers (compact
        layer order), or None without exact block scales. A null pointer compiles
        the kernel's exact path out, and the tensors stay held by
        `_eg_block_scale_residuals`."""
        if not self.exact_block_scales:
            return None
        if self._block_scale_residual_ptrs_cache is None:
            layers = sorted(self._eg_global_to_compact, key=self._eg_global_to_compact.get)
            missing = [lid for lid in layers if lid not in self._eg_block_scale_residuals]
            if missing:
                raise RuntimeError(
                    "MINISGL_MEGAMOE_EXACT_FP8_SCALES needs FP8 block-scale experts in every "
                    f"layer; layers {missing} registered none")
            self._block_scale_residual_ptrs_cache = torch.tensor(
                [int(self._eg_block_scale_residuals[lid].data_ptr()) for lid in layers],
                dtype=torch.int64, device="cuda")
        return self._block_scale_residual_ptrs_cache

    def _block_scale_residual_ptrs_for_layer(self, compact_layer_id: int) -> torch.Tensor | None:
        """The one-element view a single-layer EG launch takes: its weight descriptors
        are sliced to ``[lid:lid+1]`` with ``num_layers=1``, so the kernel only reads
        index 0 (the persistent launch passes the whole array)."""
        pointers = self._block_scale_residual_ptrs()
        return None if pointers is None else pointers[compact_layer_id : compact_layer_id + 1]

    def _compact_layer_id(self, layer_id: int) -> int:
        try:
            return self._eg_global_to_compact[int(layer_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"Layer {int(layer_id)} has no registered MegaMoE EG weights"
            ) from exc

    def _desc_tensors(self):
        if self._desc_blob_all is None:
            layers = sorted(self._eg_desc_blobs)
            assert layers == list(range(len(layers))), layers
            self._desc_blob_all = torch.stack(
                [self._eg_desc_blobs[i] for i in layers]).contiguous()
            self._l1_ptrs_all = torch.tensor(
                [self._eg_l1_ptrs[i] for i in layers],
                dtype=torch.int64, device="cuda")
        return self._desc_blob_all, self._l1_ptrs_all

    def _expected_tokens_for_eg(
        self,
        expected_tokens_per_rank: int | None,
        *,
        dual_lane_decode: bool,
    ) -> int:
        expected = int(expected_tokens_per_rank or self.bucket)
        decode_limit = self.bucket * (self.num_lanes if dual_lane_decode else 1)
        if 0 < self.eg_expected_tokens_override and expected <= decode_limit:
            # MINISGL_MEGAMOE_EG_EXPECTED_FLOOR=1: the hint is a floor, not a replacement. The MiniMax M2.5 hint (24) lifts the
            # few tokens a long-context decode step has onto the fast tiling, but it also pins a short-context step (161
            # tokens per rank at 1K) to that small tiling: 1K 1:1 b323 24.89 ms with the hint, 22.55 ms on the bucket's own.
            expected = (max(expected, self.eg_expected_tokens_override) if getattr(self, "eg_expected_floor", False)
                        else self.eg_expected_tokens_override)
        return min(expected, self.bucket)

    @property
    def _eg_layer_base(self) -> int:
        """The model layer of the first registered EG layer (0: the EG rank holds
        the whole model)."""
        return min(self._eg_global_to_compact, default=0)

    def _eg_protocol_kwargs(self, layer_base: int, first_compact: int = 0) -> dict:
        # The EG launches number layers among the MoE layers, the AG ranks by model
        # layer; P2P's combine counters are indexed by the model layer's parity, so
        # the launch's first layer (compact `first_compact`) must be model layer
        # `layer_base` and the registered layers one contiguous block (also on the
        # startup precompile, so a mismatch fails there).
        if self.p2p:
            check_eg_layer_block(self._eg_global_to_compact, layer_base, first_compact)
        return dict(
            p2p=self.p2p,
            layer_base=int(layer_base),
            push_dispatch=self.push_dispatch,
            **self.eg_early,
        )

    def eg_moe(
        self,
        layer_id: int,
        *,
        lane: int = 0,
        expected_tokens_per_rank: int | None = None,
        compile_only: bool = False,
    ) -> str | None:
        """Fused MoE layer on an EG rank: recv + FP8 experts + send.

        ``expected_tokens_per_rank`` drives the GEMM block/wave heuristics and
        must be the CURRENT step's per-AG-rank token count (e.g. the decode
        graph bucket), NOT the max dispatch bucket — feeding the 8k prefill
        bucket here selects a block_m=192 prefill tiling with one expert per
        wave, several times slower for ~46-token decode steps.

        With ``compile_only`` nothing is launched: the kernel instance this call
        would use is built and loaded and its generated source returned (see
        :meth:`precompile_eg_kernels`).
        """
        if not self.is_eg:
            raise RuntimeError("eg_moe called on an AG rank")
        expected = self._expected_tokens_for_eg(
            expected_tokens_per_rank,
            dual_lane_decode=False,
        )
        blob, ptrs = self._desc_tensors()
        lid = self._compact_layer_id(layer_id)
        if os.environ.get("MINISGL_MEGAMOE_LOG_LAUNCH") == "1" and not compile_only:
            import time as _t
            print(f"[m2n-launch] role=eg rank={self.rank} t={_t.time():.3f} layer={layer_id}", flush=True)
        source = _mm.mega_moe_m2n_eg(
            blob[lid:lid + 1], ptrs[lid:lid + 1],
            self.buffers[lane % self.num_lanes],
            rank=self.rank,
            expected_num_tokens_per_rank=expected,
            num_experts_per_rank=self.num_experts // self.eg_size,
            l1_weights_nbytes=self._eg_l1_nbytes,
            block_scale_residual_ptrs=self._block_scale_residual_ptrs_for_layer(lid),
            fast_math=True,
            num_sms=self.eg_num_sms,
            num_prefetch_bytes=self.eg_prefetch_bytes,
            compile_only=compile_only,
            # the launch's one layer is model layer `layer_id` (compact `lid`)
            **self._eg_protocol_kwargs(int(layer_id), lid),
        )
        if compile_only:
            return source
        # Diagnostic: a non-finite expert output would reach the AG combine as
        # garbage. Off unless asked for; it synchronizes, so it must never be on in
        # a measured run.
        if os.environ.get("MINISGL_MEGAMOE_CHECK_EG_OUTPUT") == "1":
            torch.cuda.synchronize()
            out = self.buffers[lane % self.num_lanes].l2_acts
            if not bool(torch.isfinite(out.view(torch.float8_e4m3fn).float()).all()):
                raise RuntimeError(
                    f"megamoe_m2n EG produced non-finite expert output at layer "
                    f"{layer_id} (expected_tokens={expected})."
                )
        return None

    def eg_moe_dual(
        self,
        layer_id: int,
        *,
        expected_tokens_per_rank: int | None = None,
    ) -> None:
        """Dual-lane fused MoE layer on an EG rank: ONE full-width kernel
        serves both microbatch lanes back-to-back (the EG rank is dedicated,
        so the 76-SM lane-co-residency constraint disappears; the GEMMs run
        at full device width and lane 1 reuses lane 0's L2-resident
        weights)."""
        if not self.is_eg:
            raise RuntimeError("eg_moe_dual called on an AG rank")
        if self.num_lanes < 2:
            raise RuntimeError("eg_moe_dual requires two adapter lanes")
        expected = self._expected_tokens_for_eg(
            expected_tokens_per_rank,
            dual_lane_decode=True,
        )
        blob, ptrs = self._desc_tensors()
        lid = self._compact_layer_id(layer_id)
        if os.environ.get("MINISGL_MEGAMOE_LOG_LAUNCH") == "1":
            import time as _t
            print(f"[m2n-launch] role=eg rank={self.rank} t={_t.time():.3f} layer={layer_id}", flush=True)
        _mm.mega_moe_m2n_eg(
            blob[lid:lid + 1], ptrs[lid:lid + 1],
            self.buffers[0],
            rank=self.rank,
            expected_num_tokens_per_rank=expected,
            num_experts_per_rank=self.num_experts // self.eg_size,
            l1_weights_nbytes=self._eg_l1_nbytes,
            block_scale_residual_ptrs=self._block_scale_residual_ptrs_for_layer(lid),
            fast_math=True,
            num_sms=0,  # full device: nothing co-resides on a dedicated EG rank
            num_prefetch_bytes=self.eg_prefetch_bytes,
            sym_buffer1=self.buffers[1],
            # the launch's one layer is model layer `layer_id` (compact `lid`)
            **self._eg_protocol_kwargs(int(layer_id), lid),
        )

    def eg_moe_persistent(
        self,
        num_layers: int,
        *,
        num_mb: int = 1,
        expected_tokens_per_rank: int | None = None,
        compile_only: bool = False,
        layer_base: int | None = None,
    ) -> str | None:
        """ONE persistent kernel serves ALL layers and active microbatch lanes.

        The tuned mb=2 path still passes exactly lane0/lane1. For mb>2 the
        same persistent kernel receives lane2/lane3 buffers instead of graphing
        many per-layer EG launches.

        ``layer_base`` is the model layer of the first registered layer (0 when
        this EG rank holds the whole model). None takes the lowest
        registered layer, as the startup precompile does, so the two launch the
        same layer numbering; under P2P a block that is not contiguous from it
        still fails (``check_eg_layer_block``).

        With ``compile_only`` nothing is launched: the kernel instance this call
        would use is built and loaded and its generated source returned (see
        :meth:`precompile_eg_kernels`).
        """
        if not self.is_eg:
            raise RuntimeError("eg_moe_persistent called on an AG rank")
        lanes = max(1, min(int(num_mb), self.num_lanes))
        if lanes > 4:
            raise RuntimeError(f"megamoe_m2n persistent supports <=4 lanes, got {lanes}")
        expected = self._expected_tokens_for_eg(
            expected_tokens_per_rank,
            dual_lane_decode=lanes >= 2,
        )
        blob, ptrs = self._desc_tensors()
        assert int(num_layers) == int(blob.shape[0]), (num_layers, blob.shape)
        source = _mm.mega_moe_m2n_eg(
            blob, ptrs,
            self.buffers[0],
            rank=self.rank,
            expected_num_tokens_per_rank=expected,
            num_experts_per_rank=self.num_experts // self.eg_size,
            l1_weights_nbytes=self._eg_l1_nbytes,
            block_scale_residual_ptrs=self._block_scale_residual_ptrs(),
            num_layers=int(num_layers),
            fast_math=True,
            num_sms=0,
            num_prefetch_bytes=self.eg_prefetch_bytes,
            sym_buffer1=self.buffers[1] if lanes >= 2 else None,
            sym_buffer2=self.buffers[2] if lanes >= 3 else None,
            sym_buffer3=self.buffers[3] if lanes >= 4 else None,
            debug_timings=self._eg_debug_timings(num_layers, lanes),
            compile_only=compile_only,
            **self._eg_protocol_kwargs(self._eg_layer_base if layer_base is None else layer_base),
        )
        return source if compile_only else None

    def eg_precompile_buckets(self, num_mb: int) -> list[int]:
        """One dispatch bucket per EG expected-token value a ``num_mb`` step can select.

        ``_expected_tokens_for_eg`` is the only step-dependent input to the EG
        kernel's instance: ``get_mega_moe_config`` derives the block_m tier,
        experts per wave and pipeline stages from it, and every other template
        argument is fixed once the weights are registered. So instead of copying
        that table, apply the selection itself to every dispatch bucket up to one
        past its last threshold (the ``bucket * num_lanes`` decode limit); beyond
        that it returns ``self.bucket``, which the last value already produces.
        """
        dual_lane_decode = self.use_persistent(num_mb)
        first_bucket_per_value: dict[int, int] = {}
        for bucket in range(1, self.bucket * self.num_lanes + 2):
            value = self._expected_tokens_for_eg(bucket, dual_lane_decode=dual_lane_decode)
            first_bucket_per_value.setdefault(value, bucket)
        return sorted(first_bucket_per_value.values())

    def precompile_eg_kernels(self, num_mb: int) -> tuple[int, int]:
        """Build and load every EG kernel instance a ``num_mb`` step can launch.

        DeepGEMM compiles an EG instance the first time a step selects it, on the
        EG host, for about 3 s. By then every AG rank has launched its half of
        that step's M2N collective. Calling this after weight registration moves
        every compile before the first collective step.

        Each bucket goes through the same launcher the step path uses
        (persistent for ``use_persistent``, per-layer otherwise; the dual-lane
        graph capture selects the same instance as persistent), with
        ``compile_only`` so nothing is launched. Returns
        ``(distinct instances, buckets tried)``.
        """
        if not self.is_eg:
            raise RuntimeError("precompile_eg_kernels called on an AG rank")
        if not self._eg_global_to_compact:
            raise RuntimeError(
                "precompile_eg_kernels needs every EG layer registered first: the "
                "instance depends on the registered weights"
            )
        buckets = self.eg_precompile_buckets(num_mb)
        persistent = self.use_persistent(num_mb)
        # The per-layer instance does not depend on which layer launches it.
        any_layer = next(iter(self._eg_global_to_compact))
        sources = set()
        for bucket in buckets:
            if persistent:
                source = self.eg_moe_persistent(
                    len(self._eg_global_to_compact),
                    num_mb=num_mb,
                    expected_tokens_per_rank=bucket,
                    compile_only=True,
                    layer_base=self._eg_layer_base,
                )
            else:
                source = self.eg_moe(
                    any_layer,
                    expected_tokens_per_rank=bucket,
                    compile_only=True,
                )
            sources.add(source)
        return len(sources), len(buckets)

    def use_persistent(self, num_mb: int) -> bool:
        mb = int(num_mb)
        return self.use_dual_lane(mb) or (2 < mb <= 4 and self.num_lanes >= mb)

    def use_dual_lane(self, num_mb: int) -> bool:
        return (
            int(num_mb) == 2
            and self.num_lanes >= 2
        )

    def destroy(self) -> None:
        if self._destroyed:
            return
        for buf in self.buffers:
            buf.destroy()
        self._eg_weights.clear()
        self._eg_global_to_compact.clear()
        self._y_cache.clear()
        self._eager_buffers.clear()
        self._destroyed = True

    close = destroy


__all__ = ["MegaMoEM2NAfdAdapter"]
