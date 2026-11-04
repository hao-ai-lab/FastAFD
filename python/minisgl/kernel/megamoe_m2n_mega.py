"""Split MegaMoE M2N: the DeepGEMM mega kernel split into AG/EG role kernels.

This wraps the vendored DeepGEMM extension's split mega-MoE entry points:

- ``mega_moe_m2n_ag``: AG/attention half — one kernel that only dispatch-sends
  (token-topk index + count push; EG ranks pull the FP8 payload) and
  combine-receives (waits for EG write-back, reduces top-k, writes ``y``).
- ``mega_moe_m2n_eg``: EG/MLP half — one persistent kernel that pulls FP8
  tokens from AG ranks over NVLink, runs the fused FP8xFP4 L1/L2 grouped GEMM
  pipeline (SwiGLU + top-k weights fused in the L1 epilogue), and pushes BF16
  expert outputs into the source AG ranks' combine buffers.

Topology: one union process group of ``ag_size + eg_size`` ranks, AG first.
EG rank ``g`` owns experts ``[g * E_local, (g + 1) * E_local)``.  The
symmetric buffer is allocated with ``torch.distributed._symmetric_memory``
(NVLink/MNNVL fabric), identical layout on every rank.
"""

from __future__ import annotations

import functools
import os
import types
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from . import deepgemm as _dg


@functools.lru_cache(maxsize=None)
def _cu_stream_wait_value32():
    import ctypes

    fn = ctypes.CDLL("libcuda.so.1").cuStreamWaitValue32_v2
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    return fn


def stream_wait_ge(flag: torch.Tensor, value: int) -> None:
    """Make the current stream wait until the int32 at ``flag`` is >= ``value``.

    cuStreamWaitValue32(GEQ): the GPU front end polls the word, no SM is held, and
    later work on the stream starts once it holds. The comparison is cyclic, which
    the small P2P counters never reach. Captured into a CUDA graph as a memory-op
    node. Ordering of the data the counter guards is left to the consumer kernel,
    whose own acquire load of the counter is then satisfied at once.
    """
    import ctypes

    stream = torch.cuda.current_stream()
    rc = _cu_stream_wait_value32()(ctypes.c_void_p(stream.cuda_stream), ctypes.c_uint64(flag.data_ptr()),
                                   ctypes.c_uint32(int(value)), 0)  # CU_STREAM_WAIT_VALUE_GEQ
    if rc != 0:
        raise RuntimeError(f"cuStreamWaitValue32 failed with CUresult {rc}")


@functools.lru_cache(maxsize=None)
def _wait_ge_kernel():
    import triton
    import triton.language as tl

    from triton.language.extra.cuda import globaltimer

    # the function name is the kernel's name in profiles
    @triton.jit
    def wait_ge_cta_kernel(flag_ptr, target, tag, RESET: tl.constexpr, GUARD: tl.constexpr):
        start = globaltimer()
        v = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
        while v < target:
            v = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
            # hang guard (MINISGL_MEGAMOE_HANG_GUARD=1), like the M2N kernels': report and trap
            if GUARD and globaltimer() - start > 300_000_000_000:
                tl.device_print("DG hang guard: wait_ge_cta tag value target", tag, v, target)
                tl.inline_asm_elementwise("trap;\n mov.u32 $0, 0;", "=r", [], dtype=tl.int32,
                                          is_pure=False, pack=1)
        if RESET:
            # take this round's target off rather than zeroing: an early increment
            # of the next round on this counter survives
            tl.atomic_add(flag_ptr, -target, sem="relaxed", scope="sys")

    return wait_ge_cta_kernel


# MINISGL_PDL=1: programmatic dependent launch (see minisgl/utils.cuh)
_PDL = os.environ.get("MINISGL_PDL", "0") == "1"


@functools.lru_cache(maxsize=None)
def _wait_ge_mono_kernel():
    import triton
    import triton.language as tl

    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait, globaltimer

    # the function name is the kernel's name in profiles
    @triton.jit
    def wait_ge_cta_kernel(flag_ptr, consumed_ptr, value, tag, GUARD: tl.constexpr, PDL: tl.constexpr = False):
        if PDL:
            gdc_wait()
        # a counter that only grows: this round is complete once it passes what
        # earlier rounds consumed by `value` (wrap-safe signed difference); then
        # it is consumed with a plain store of this rank's own word, no atomic
        target = tl.load(consumed_ptr) + value
        start = globaltimer()
        v = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
        while v - target < 0:
            v = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
            if GUARD and globaltimer() - start > 300_000_000_000:
                tl.device_print("DG hang guard: wait_ge_cta tag value target", tag, v, target)
                tl.inline_asm_elementwise("trap;\n mov.u32 $0, 0;", "=r", [], dtype=tl.int32,
                                          is_pure=False, pack=1)
        tl.store(consumed_ptr, target)
        if PDL:
            gdc_launch_dependents()

    return wait_ge_cta_kernel


def wait_ge_cta_mono(flag: torch.Tensor, consumed: torch.Tensor, value: int, tag: int = 0) -> None:
    """One-CTA wait on a monotonic counter: spins until the int32 at ``flag`` has
    grown by ``value`` past the int32 at ``consumed`` (this rank's own word, which
    only this wait reads and writes), then advances ``consumed`` by ``value``. The
    same rounds as ``wait_ge_cta(..., reset=True)`` -- an early increment of the
    next round still counts toward it -- without a system-scope atomic; the combine
    launch behind it then skips its own wait and grid sync (``waited_combine``)."""
    _wait_ge_mono_kernel()[(1,)](flag, consumed, int(value), int(tag),
                                 GUARD=os.environ.get("MINISGL_MEGAMOE_HANG_GUARD", "0") not in ("", "0"),
                                 PDL=_PDL, num_warps=1, launch_pdl=_PDL)


def wait_ge_cta(flag: torch.Tensor, value: int, reset: bool = False, tag: int = 0) -> None:
    """One-CTA kernel on the current stream that spins until the int32 at
    ``flag`` is >= ``value`` (an SM while it waits, instead of a whole grid);
    with ``reset`` it then takes ``value`` off the counter. ``tag`` only names the
    wait in the hang guard's report (MINISGL_MEGAMOE_HANG_GUARD=1: after 300 s it
    prints and traps)."""
    _wait_ge_kernel()[(1,)](flag, int(value), int(tag), RESET=bool(reset),
                            GUARD=os.environ.get("MINISGL_MEGAMOE_HANG_GUARD", "0") not in ("", "0"),
                            num_warps=1)


def _ext():
    return _dg._load_extension()


def get_token_alignment() -> int:
    return int(_ext().get_token_alignment_for_mega_moe())


def align(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y) * int(y)


# Control words of the P2P protocol at the end of the symmetric buffer
# (layout::P2PRegionsM2N::kNumCtrlBytes): [0] credit, [1 + parity] combine count,
# [8 + parity] how far an AG rank's monotonic combine wait consumed it.
P2P_CTRL_BYTES = 64


class MegaMoEM2NSymmBuffer:
    """Union-symmetric buffer for the split M2N mega kernels.

    AG ranks use the ``x``/``x_sf``/``topk_idx``/``topk_weights`` input views
    (copy inputs in before every AG launch) and receive combine output via the
    kernel's ``y``.  EG ranks use the L1/L2 pool views implicitly.

    The kernels address the union's ranks by their group rank (AG ranks first),
    through ``buffer_ptrs`` and the ``rank`` of each launch (``position``).
    """

    def __init__(self, group: dist.ProcessGroup, *,
                 ag_size: int, eg_size: int,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.ag_size = int(ag_size)
        self.eg_size = int(eg_size)
        if group.size() != self.ag_size + self.eg_size:
            raise RuntimeError(
                f"group size {group.size()} != ag_size + eg_size = {self.ag_size + self.eg_size}"
            )
        self.position = int(dist.get_rank(group))
        self.num_experts = int(num_experts)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.num_max_tokens_per_rank = align(num_max_tokens_per_rank, get_token_alignment())

        num_bytes, num_max_pool_tokens, slice_input_buffers = (
            _ext().get_symm_buffer_size_for_mega_moe_m2n(
                self.ag_size, self.eg_size, self.num_experts,
                self.num_max_tokens_per_rank, self.num_topk,
                self.hidden, self.intermediate_hidden,
            )
        )
        self.num_max_pool_tokens = int(num_max_pool_tokens)
        self.num_bytes = int(num_bytes)
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(int(num_bytes), dtype=torch.int8, device="cuda")
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        # Seed the phase-0 NVLink barrier signal with a completed +1 round:
        # the kernels defer the cleanup-barrier WAIT into the next launch, so
        # the very first wait must observe a finished phase.  Workspace head:
        # [0:16) grid-sync counters, [16:20) barrier counter, [20:24) signal
        # phase 0, [24:28) signal phase 1.
        self.buffer[20:24].view(torch.int32).fill_(group.size())
        # P2P protocol: the credit word (first of the 64 control bytes that end
        # the buffer, layout::P2PRegionsM2N) starts as one full round of EG
        # credits, so an AG rank's first push does not wait.
        ctrl = int(num_bytes) - P2P_CTRL_BYTES
        self.buffer[ctrl:ctrl + 4].view(torch.int32).fill_(self.eg_size)
        dist.barrier(group=group)
        torch.cuda.synchronize()

        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf,
         self.combine_rows) = slice_input_buffers(self.buffer)

    @property
    def buffer_ptrs(self) -> list[int]:
        """The peer table the kernels index by union position."""
        return [int(p) for p in self.handle.buffer_ptrs]

    def combine_count_view(self, parity: int) -> torch.Tensor:
        """P2P: int32 [1] view of this rank's combine counter of layer parity
        `parity` (layout::P2PRegionsM2N::get_combine_count_ptr). Every EG rank and
        the dispatching AG rank add 1 to it; it is complete at eg_size + 1."""
        off = self.num_bytes - P2P_CTRL_BYTES + 4 * (1 + int(parity))
        return self.buffer[off:off + 4].view(torch.int32)

    def combine_consumed_view(self, parity: int) -> torch.Tensor:
        """P2P, AG ranks: int32 [1] view of how far this rank consumed its combine
        counter of parity `parity` (control word 8 + parity; nobody else writes it),
        for the monotonic counter wait (wait_ge_cta_mono)."""
        off = self.num_bytes - P2P_CTRL_BYTES + 4 * (8 + int(parity))
        return self.buffer[off:off + 4].view(torch.int32)

    def destroy(self) -> None:
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l1_acts_sf = None
        self.l2_acts = None
        self.l2_acts_sf = None


def mega_moe_m2n_ag(y: torch.Tensor,
                    hidden_states: torch.Tensor,
                    topk_ids: Optional[torch.Tensor],
                    topk_weights: Optional[torch.Tensor],
                    sym_buffer: MegaMoEM2NSymmBuffer,
                    *, rank: int,
                    num_sms: int = 0,
                    gate_weight: Optional[torch.Tensor] = None,
                    renormalize: bool = True,
                    phase: int = 0,
                    external_quant: bool = False,
                    direct_combine: bool = False,
                    input_num_topk: Optional[int] = None,
                    num_routed_experts: Optional[int] = None,
                    shared_topk_weight: float = 0.0,
                    debug_timings: Optional[torch.Tensor] = None,
                    p2p: bool = False,
                    layer_parity: int = 0,
                    push_dispatch: bool = False,
                    waited_combine: bool = False) -> None:
    """AG half: fused per-32 FP8 quant + dispatch-send + combine-recv.

    ``phase`` splits the kernel: 0 = fused (default), 1 = dispatch-only
    (quant + pushes, exits without waiting so its SMs go back to attention
    during the EG round trip), 2 = combine-only (tag-2 wait + top-k reduce +
    cleanup).  A phase-1 launch MUST be followed by exactly one phase-2
    launch before the next phase-1 on the same buffer.

    One kernel does everything: quantizes ``hidden_states`` (bf16
    [num_tokens, hidden]) into the symmetric buffer (e4m3 + ceil-UE8M0 per-32
    scales), stages ``topk_weights`` for the EG pull, pushes route metadata,
    waits for the expert write-back and reduces top-k into ``y`` (bf16).
    ``topk_ids`` may be int32 or int64 (read in place, not copied).

    If ``gate_weight`` (bf16 [num_experts, hidden]) is given, the ROUTER runs
    inside the kernel too (GEMV + softmax + top-k + optional renormalize) and
    ``topk_ids``/``topk_weights`` must be None.  Intended for decode-sized
    batches (a warp-per-token GEMV; use the host gate for prefill).

    ``num_sms`` limits the persistent grid (0 = all SMs).  The kernel spends
    most of its time spin-waiting for the EG combine push, so a small grid
    keeps the GPU free for concurrent compute (e.g. the other microbatch's
    attention).

    ``p2p`` selects the point-to-point protocol (every rank of the union must
    agree; ``layer_parity`` is the model layer's parity, which indexes the
    combine counter).

    ``push_dispatch`` (in-kernel quant with host top-k only): the quant also writes
    each token's rows into every EG rank's staging rows of this rank, and the EG
    launches (``push_dispatch`` too) read them locally instead of pulling them.

    ``waited_combine`` (P2P, phase 2): this combine launch follows
    ``wait_ge_cta(..., reset=True)`` on the same counter, so it neither waits for
    nor clears the counter, and needs no grid sync.
    """
    _ext().mega_moe_m2n_ag(
        y,
        hidden_states, topk_ids, topk_weights,
        gate_weight, bool(renormalize),
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs, int(rank),
        sym_buffer.ag_size, sym_buffer.eg_size,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        sym_buffer.intermediate_hidden,
        int(num_sms),
        int(phase),
        bool(external_quant),
        bool(direct_combine),
        int(input_num_topk if input_num_topk is not None else sym_buffer.num_topk),
        int(num_routed_experts if num_routed_experts is not None else sym_buffer.num_experts),
        float(shared_topk_weight),
        debug_timings,
        bool(p2p),
        int(layer_parity),
        bool(push_dispatch),
        bool(waited_combine),
    )


def make_m2n_weight_descs(l1_weights: Tuple[torch.Tensor, torch.Tensor],
                          l2_weights: Tuple[torch.Tensor, torch.Tensor],
                          *, num_experts_per_rank: int,
                          hidden: int, intermediate_hidden: int) -> torch.Tensor:
    """Build the per-layer weight tensormap blob (uint8 [4, 128], CPU) for the
    persistent EG kernel.  Config-independent, so it is built once at weight
    registration."""
    return _ext().make_m2n_weight_descs(
        l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1],
        int(num_experts_per_rank), int(hidden), int(intermediate_hidden),
    )


def mega_moe_m2n_eg(weight_descs: torch.Tensor,
                    l1_weight_ptrs: torch.Tensor,
                    sym_buffer: MegaMoEM2NSymmBuffer,
                    *, rank: int,
                    expected_num_tokens_per_rank: int,
                    num_experts_per_rank: int,
                    l1_weights_nbytes: int,
                    num_layers: int = 1,
                    use_fp8_weights: bool = True,
                    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                    activation_clamp: Optional[float] = None,
                    block_scale_residual_ptrs: Optional[torch.Tensor] = None,
                    fast_math: bool = True,
                    num_sms: int = 0,
                    num_prefetch_bytes: int = 0,
                    sym_buffer1: Optional[MegaMoEM2NSymmBuffer] = None,
                    sym_buffer2: Optional[MegaMoEM2NSymmBuffer] = None,
                    sym_buffer3: Optional[MegaMoEM2NSymmBuffer] = None,
                    debug_timings: Optional[torch.Tensor] = None,
                    compile_only: bool = False,
                    p2p: bool = False,
                    layer_base: int = 0,
                    early_max_late: int = 0,
                    early_wait_cycles: int = 0,
                    early_policy: int = 0,
                    early_pass_cycles: int = 0,
                    early_expert_cycles: int = 0,
                    early_margin_pct: int = 0,
                    push_dispatch: bool = False) -> str:
    """EG half: dispatch-recv + FP8 expert MLP + combine-send.

    ``weight_descs`` is the device uint8 [num_layers, 4, 128] tensormap blob
    (built via :func:`make_m2n_weight_descs`); ``l1_weight_ptrs`` the device
    int64 [num_layers] L1-weight base pointers (for the L2 prefetch).  With
    ``num_layers > 1`` ONE persistent kernel serves the whole step (per-launch
    smem/TMEM init amortized); with ``sym_buffer1`` it also serves both
    microbatch lanes per layer at full device width.

    With ``compile_only`` the kernel instance these arguments select is JIT
    built and loaded but NOT launched, and its generated source is returned
    (equal strings are the same instance); otherwise it launches and returns
    ``""``. The EG kernel is one half of a cross-rank protocol and cannot run
    alone, so this is how startup takes the compile off the step path.

    ``p2p`` must match the AG launches; ``layer_base`` is the model layer of the
    launch's first layer (the combine counters are indexed by the model layer's
    parity). ``push_dispatch`` (with the AG side's): read the tokens from this
    rank's staging rows instead of pulling them.

    ``block_scale_residual_ptrs`` (device int64 [num_layers], FP8 weights only):
    per layer, a float [E_local, L1 blocks + L2 blocks] of the residuals
    :func:`fp8_weights_with_block_scale_residuals` returns (L1's then L2's, each
    flattened row-block major); the weights then keep their own block scales.

    Early start (P2P only): with at most ``early_max_late`` source ranks still
    missing, the round may be served in two passes, the arrived sources first;
    ``early_max_late`` 0 keeps one pass. When, is ``early_policy``: 0 splits once
    the late ones are still missing ``early_wait_cycles`` SM cycles later; 1
    (adaptive) splits right away if the late ones are expected to come no sooner
    than the first pass ends, with ``early_margin_pct`` to spare. The kernel
    prices the first pass at ``early_pass_cycles`` plus a per-expert time (the
    experts the arrived sources route to) that it learns per lane, starting
    from ``early_expert_cycles``, and learns per lane how late the late sources
    come. The rows are the same either way.
    """
    return _ext().mega_moe_m2n_eg(
        weight_descs, l1_weight_ptrs,
        int(num_layers),
        int(num_experts_per_rank),
        sym_buffer.hidden, sym_buffer.intermediate_hidden,
        int(l1_weights_nbytes),
        bool(use_fp8_weights),
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer1.buffer if sym_buffer1 is not None else None,
        sym_buffer2.buffer if sym_buffer2 is not None else None,
        sym_buffer3.buffer if sym_buffer3 is not None else None,
        sym_buffer.buffer_ptrs,
        sym_buffer1.buffer_ptrs if sym_buffer1 is not None else sym_buffer.buffer_ptrs,
        sym_buffer2.buffer_ptrs if sym_buffer2 is not None else sym_buffer.buffer_ptrs,
        sym_buffer3.buffer_ptrs if sym_buffer3 is not None else sym_buffer.buffer_ptrs,
        int(rank),
        sym_buffer.ag_size, sym_buffer.eg_size,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        int(expected_num_tokens_per_rank),
        activation_clamp,
        block_scale_residual_ptrs,
        bool(fast_math),
        int(num_sms),
        int(num_prefetch_bytes),
        debug_timings,
        bool(compile_only),
        bool(p2p),
        int(layer_base),
        int(early_max_late),
        int(early_wait_cycles),
        int(early_policy),
        int(early_pass_cycles),
        int(early_expert_cycles),
        int(early_margin_pct),
        bool(push_dispatch),
    )


# ---------------------------------------------------------------------------
# FP4 weight utilities (ported from upstream DeepGEMM `deep_gemm.utils.math`
# and `deep_gemm.mega`)
# ---------------------------------------------------------------------------
def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                              device=x.device, dtype=torch.float)
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def per_token_cast_to_fp4(x: torch.Tensor, *, use_ue8m0: bool = True, gran_k: int = 32,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    assert n % 2 == 0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = _dg.ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, :n // 2].contiguous(), \
        (_dg.pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf)


def dequant_fp4(packed: torch.Tensor, sf: torch.Tensor, *, gran_k: int = 32) -> torch.Tensor:
    """Dequantize packed FP4 e2m1 with float UE8M0 scales back to fp32."""
    m, n2 = packed.shape
    n = n2 * 2
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x = _dequantize_from_fp4_e2m1(unpacked)
    scale = torch.repeat_interleave(sf.float(), gran_k, dim=1)[:, :n]
    return x * scale


def requant_qwen_fp8_weights_per32(
    w_fp8: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    block: int = 128,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen FP8 weights -> mega-kernel FP8 weights with per-32 UE8M0 scales.

    Input: ``w_fp8`` [E, N, K] e4m3 with float 128x128 ``block_scales``
    [E, ceil(N/128), ceil(K/128)].  The block scales are rounded UP to UE8M0
    and the weights are rescaled by ``s / s_ue8m0`` (one fp8 round trip), so
    ``w' * sf == w * s`` up to e4m3 rounding.  Returns the requantized weights
    and the DeepGEMM-layout packed UE8M0 SF for the (1, 32) recipe.
    """
    assert w_fp8.dtype == torch.float8_e4m3fn and w_fp8.dim() == 3
    e, n, k = w_fp8.shape
    assert k % 32 == 0
    s = block_scales.float()
    s_u = _dg.ceil_to_ue8m0(s)
    ratio = torch.where(s_u > 0, s / s_u, torch.zeros_like(s))
    if inplace and not w_fp8.is_contiguous():
        raise RuntimeError("in-place MegaMoE FP8 weight requant requires contiguous weights")
    w_new = w_fp8 if inplace else torch.empty_like(w_fp8, memory_format=torch.contiguous_format)
    for expert in range(e):
        ratio_e = torch.repeat_interleave(ratio[expert], block, dim=0)[:n]
        ratio_e = torch.repeat_interleave(ratio_e, block, dim=1)[:, :k]
        w_new[expert].copy_(
            (w_fp8[expert].float() * ratio_e).to(torch.float8_e4m3fn)
        )
        del ratio_e
    return w_new, _packed_per32_sf(s_u, n, k, block)


def fp8_weights_with_block_scale_residuals(
    w_fp8: torch.Tensor,
    block_scales: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen FP8 weights -> mega-kernel operands that keep the weights' values.

    Input as :func:`requant_qwen_fp8_weights_per32`. Returns the weights as they
    are, the packed per-32 UE8M0 SF of each block scale rounded UP, and the float
    residuals ``s / s_ue8m0`` in (0.5, 1], [E, ceil(N/128), ceil(K/128)]. The EG
    kernel multiplies each 128-wide K block's partial sum by its block's residual
    (``block_scale_residual_ptrs``), so it computes ``w * s`` exactly, where the
    requantized weights carry a second e4m3 rounding (2.7% relative L2 on
    MiniMax-M2.5's experts).
    """
    assert w_fp8.dtype == torch.float8_e4m3fn and w_fp8.dim() == 3
    _, n, k = w_fp8.shape
    assert k % 32 == 0
    s = block_scales.float()
    s_u = _dg.ceil_to_ue8m0(s)
    residuals = torch.where(s_u > 0, s / s_u, torch.zeros_like(s)).contiguous()
    return w_fp8, _packed_per32_sf(s_u, n, k, 128), residuals


def _packed_per32_sf(s_u: torch.Tensor, n: int, k: int, block: int) -> torch.Tensor:
    """UE8M0 block scales [E, ceil(N/block), ceil(K/block)] -> the DeepGEMM-layout
    packed SF of the (1, 32) recipe."""
    sf32 = torch.repeat_interleave(s_u, block, dim=1)[:, :n]
    sf32 = torch.repeat_interleave(sf32, block // 32, dim=2)[:, :, : k // 32].contiguous()
    return _dg.transform_sf_into_required_layout(sf32, n, k, (1, 32), s_u.shape[0])


def dequant_fp8_per32(w_fp8: torch.Tensor, sf32: torch.Tensor) -> torch.Tensor:
    """Dequantize [N, K] fp8 with per-32 float scales [N, K/32] (reference)."""
    return w_fp8.float() * torch.repeat_interleave(sf32.float(), 32, dim=1)


def cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """[E, N, K] bf16 -> packed FP4 weights + DeepGEMM-layout UE8M0 SF."""
    num_groups, n, k = bf16_weights.shape
    w = torch.empty((num_groups, n, k // 2), device="cuda", dtype=torch.int8)
    w_sf = torch.empty((num_groups, n, k // 32), device="cuda", dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
    w_sf_t = _dg.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
    return w, w_sf_t


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    out = torch.empty_like(t)
    out_view = out.reshape(g, half // gran, 2, gran, *rest)
    out_view[:, :, 0].copy_(gate)
    out_view[:, :, 1].copy_(up)
    return out


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return (l1_w, l1_sf), l2_transformed


__all__ = [
    "MegaMoEM2NSymmBuffer",
    "cast_weights_to_fp4",
    "dequant_fp4",
    "dequant_fp8_per32",
    "get_token_alignment",
    "mega_moe_m2n_ag",
    "mega_moe_m2n_eg",
    "per_token_cast_to_fp4",
    "requant_qwen_fp8_weights_per32",
    "transform_weights_for_mega_moe",
]
