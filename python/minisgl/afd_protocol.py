from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Literal
import warnings

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine.sample import BatchSamplingArgs, make_device_tensor


_TENSOR_CODEC_TAG = "__minisgl_afd_tensor__"
_TOKEN_BLOCK_CODEC_TAG = "__minisgl_afd_token_block__"
_AG_STEP_CMD_CODEC_TAG = "__minisgl_afd_ag_step_cmd__"


@dataclass(frozen=True)
class AfdTopology:
    """Static AG/EG worker layout for AFD.

    AFD uses one torch distributed "union" world that contains all attention
    workers first, followed by all expert workers:

        [attn dp0/tp0..N, attn dp1/tp0..N, ...,
         mlp  dp0/tp0..N, mlp  dp1/tp0..N, ...]

    This value object centralizes the integer mapping rules for that layout so
    coordinator, launcher, and worker code do not each open-code slightly
    different rank/fan-in formulas.

    Supported MoE EP modes:
      ep_size == 1:
        no EP.
      ep_size == mlp_tp_size:
        per-MLP-DP EP; experts are replicated across MLP DP replicas.
      ep_size == mlp_dp_size * mlp_tp_size:
        full-world EP across all MLP workers.
    """

    attn_dp_size: int
    mlp_dp_size: int
    attn_tp_size: int
    mlp_tp_size: int
    ep_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "attn_dp_size",
            "mlp_dp_size",
            "attn_tp_size",
            "mlp_tp_size",
            "ep_size",
        ):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"afd topology {name} must be >= 1, got {value}")
            object.__setattr__(self, name, value)
        self._validate_mutually_divisible(
            "attn_dp_size",
            self.attn_dp_size,
            "mlp_dp_size",
            self.mlp_dp_size,
        )
        self._validate_mutually_divisible(
            "attn_tp_size",
            self.attn_tp_size,
            "mlp_tp_size",
            self.mlp_tp_size,
        )
        mlp_world = self.mlp_world_size
        if self.ep_size > mlp_world or mlp_world % self.ep_size != 0:
            raise ValueError(
                "afd topology requires ep_size to be a divisor of the MLP world: "
                f"ep_size={self.ep_size} mlp_world_size={mlp_world} "
                f"(mlp_dp_size={self.mlp_dp_size} mlp_tp_size={self.mlp_tp_size})"
            )
        if self.ep_size not in (1, self.mlp_tp_size, mlp_world):
            raise ValueError(
                "afd topology supports ep_size in "
                "{1, mlp_tp_size, mlp_dp_size*mlp_tp_size}: "
                f"ep_size={self.ep_size} mlp_tp_size={self.mlp_tp_size} "
                f"mlp_world_size={mlp_world}"
            )

    @staticmethod
    def _validate_mutually_divisible(
        lhs_name: str,
        lhs: int,
        rhs_name: str,
        rhs: int,
    ) -> None:
        small, large = sorted((int(lhs), int(rhs)))
        if large % small != 0:
            raise ValueError(
                f"afd topology requires {lhs_name} and {rhs_name} to be "
                "mutually divisible, "
                f"got {lhs_name}={lhs} {rhs_name}={rhs}"
            )

    @property
    def is_cross_mlp_ep(self) -> bool:
        return self.ep_size > self.mlp_tp_size

    @property
    def mlp_world_size(self) -> int:
        return self.mlp_dp_size * self.mlp_tp_size

    @property
    def total_attn_workers(self) -> int:
        return self.attn_dp_size * self.attn_tp_size

    @property
    def total_mlp_workers(self) -> int:
        return self.mlp_dp_size * self.mlp_tp_size

    @property
    def total_workers(self) -> int:
        return self.total_attn_workers + self.total_mlp_workers

    @property
    def attn_fanin_per_mlp_dp(self) -> int:
        """How many attention DP lanes can feed one MLP DP lane."""
        return max(1, self.attn_dp_size // self.mlp_dp_size)

    @property
    def mlp_fanout_per_attn_dp(self) -> int:
        """How many MLP DP lanes one attention DP lane may drive."""
        return max(1, self.mlp_dp_size // self.attn_dp_size)

    def mlp_dp_for_attn_dp(self, attn_dp_rank: int) -> int:
        """Primary MLP DP lane assigned to an attention DP lane."""
        return int(attn_dp_rank) * self.mlp_dp_size // self.attn_dp_size

    def attn_dp_for_mlp_dp(self, mlp_dp_rank: int) -> int:
        """Primary attention DP lane paired with an MLP DP lane."""
        return int(mlp_dp_rank) * self.attn_dp_size // self.mlp_dp_size

    def attn_worker_rank(self, attn_dp_rank: int, tp_rank: int) -> int:
        return int(attn_dp_rank) * self.attn_tp_size + int(tp_rank)

    def mlp_worker_rank(self, mlp_dp_rank: int, tp_rank: int) -> int:
        return (
            self.total_attn_workers
            + int(mlp_dp_rank) * self.mlp_tp_size
            + int(tp_rank)
        )

    def attn_tp_group(self, attn_dp_rank: int) -> tuple[int, ...]:
        return tuple(
            self.attn_worker_rank(attn_dp_rank, tp_rank)
            for tp_rank in range(self.attn_tp_size)
        )

    def mlp_tp_group(self, mlp_dp_rank: int) -> tuple[int, ...]:
        return tuple(
            self.mlp_worker_rank(mlp_dp_rank, tp_rank)
            for tp_rank in range(self.mlp_tp_size)
        )

    def union_tp_groups(self) -> list[tuple[int, ...]]:
        """TP subgroups inside the AG+EG union distributed world."""
        return [
            *(self.attn_tp_group(dp_rank) for dp_rank in range(self.attn_dp_size)),
            *(self.mlp_tp_group(dp_rank) for dp_rank in range(self.mlp_dp_size)),
        ]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported afd protocol tensor dtype: {name!r}")
    return dtype


def _encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.device.type != "cpu":
        raise ValueError(f"afd protocol only serializes CPU tensors, got {tensor.device}")
    contiguous = tensor.detach().contiguous()
    return {
        _TENSOR_CODEC_TAG: True,
        "dtype": _dtype_name(contiguous.dtype),
        "shape": tuple(int(dim) for dim in contiguous.shape),
        "data": contiguous.numpy().tobytes(order="C"),
    }


def _decode_tensor(payload: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(dim) for dim in payload["shape"])
    dtype = _dtype_from_name(str(payload["dtype"]))
    if not payload["data"]:
        return torch.empty(shape, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor = torch.frombuffer(
            payload["data"],
            dtype=dtype,
        )
    return tensor.reshape(shape)


@dataclass
class AfdSamplingPlan:
    temperatures: list[float]
    top_ks: list[int]
    top_ps: list[float]

    @classmethod
    def from_sampling_params(cls, params: Iterable[SamplingParams]) -> AfdSamplingPlan:
        temperatures: list[float] = []
        top_ks: list[int] = []
        top_ps: list[float] = []
        all_greedy = True
        for p in params:
            temperature = float(p.temperature)
            top_k = int(p.top_k)
            top_p = float(p.top_p)
            temperatures.append(temperature)
            top_ks.append(top_k)
            top_ps.append(top_p)
            if not ((temperature <= 0.0 or top_k == 1) and top_p == 1.0):
                all_greedy = False
        # Greedy is the common AFD benchmark path. An empty plan is equivalent
        # to BatchSamplingArgs(None) and avoids shipping three per-request lists.
        if all_greedy:
            return cls([], [], [])
        return cls(temperatures=temperatures, top_ks=top_ks, top_ps=top_ps)

    def to_sampling_args(self, device: torch.device, vocab_size: int) -> BatchSamplingArgs:
        is_greedy = [
            (temperature <= 0.0 or top_k == 1) and top_p == 1.0
            for temperature, top_k, top_p in zip(self.temperatures, self.top_ks, self.top_ps)
        ]
        if all(is_greedy):
            return BatchSamplingArgs(temperatures=None)

        min_p = min_t = 1e-6
        temperatures = [
            max(0.0 if greedy else temperature, min_t)
            for temperature, greedy in zip(self.temperatures, is_greedy)
        ]
        top_ks = [top_k if top_k >= 1 else vocab_size for top_k in self.top_ks]
        top_ps = [min(max(top_p, min_p), 1.0) for top_p in self.top_ps]

        temp_tensor = make_device_tensor(temperatures, torch.float32, device)
        top_k_tensor = None
        top_p_tensor = None
        if any(top_k != vocab_size for top_k in top_ks):
            top_k_tensor = make_device_tensor(top_ks, torch.int32, device)
        if any(top_p < 1.0 for top_p in top_ps):
            top_p_tensor = make_device_tensor(top_ps, torch.float32, device)
        return BatchSamplingArgs(temp_tensor, top_k=top_k_tensor, top_p=top_p_tensor)


@dataclass
class AfdCommand:
    """Base class for coordinator→worker commands."""


@dataclass
class AfdStopCmd(AfdCommand):
    pass


@dataclass
class AfdFlushStepCmd(AfdCommand):
    """Ask workers to finalize their one-step-ahead pending step."""


@dataclass
class AfdProfilerCmd(AfdCommand):
    action: Literal["start", "stop"]
    sync: bool = False


@dataclass
class AfdReply:
    """Base class for worker→coordinator replies."""
    worker_rank: int
    step_id: int | None
    # Step replies copy the coordinator-side timestamp from the matching
    # AfdRun*StepCmd.  The coordinator can then measure command→reply latency
    # with its own clock; worker clocks on other nodes are not comparable.
    sent_ns: int = 0


@dataclass
class AfdWorkerReadyReply(AfdReply):
    pass


@dataclass
class AfdTokenBlock:
    table_idx: int
    start_pos: int
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        self.table_idx = int(self.table_idx)
        self.start_pos = int(self.start_pos)
        if self.tokens.dtype != torch.int32:
            self.tokens = self.tokens.to(dtype=torch.int32)
        self.tokens = self.tokens.contiguous()


# ============================================================================
# AG/EG (afd near-complete form) plans, commands, replies.
#
# Built on the afd scheduling output to stay consistent:
#   - AfdAGStepPlan carries AG-side KV scheduling fields plus token-pool /
#     sampling fields because AG owns embed + token_pool + sampling. The QKV/O
#     comm metadata is dropped:
#     attention is AG-local, so the only comm is the MoE dispatch(AG->EG) /
#     combine(EG->AG).
#   - AfdEGStepPlan is nearly empty: EG is a stateless expert executor; the
#     dispatch/combine routing is data-driven (topk ids), so it only needs the
#     step id, phase, num_mb and the shared dispatch_bucket.
#
# One command per (step, microbatch); the worker runs the WHOLE layer loop on
# GPU and the per-layer DeepEP collectives keep AG/EG in lockstep (no per-layer
# coordinator round-trip).
# ============================================================================


@dataclass
class AfdAGStepPlan:
    step_id: int
    phase: Literal["prefill", "decode"]
    real_size: int
    # KV / req scheduling (AG owns the KV cache)
    table_indices: tuple[int, ...] = ()
    extend_lens: tuple[int, ...] = ()
    free_table_indices: tuple[int, ...] = ()
    microbatch_offsets: tuple[int, ...] = (0, 0)
    microbatch_token_offsets: tuple[int, ...] = (0, 0)
    microbatch_real_token_counts: tuple[int, ...] = (0,)
    # token_pool + sampling metadata in flattened execution-span order. Each
    # span corresponds to a contiguous token range in one table row. Keeping
    # these as primitive tuples avoids sending one Python object per span.
    exec_table_indices: tuple[int, ...] = ()
    exec_start_positions: tuple[int, ...] = ()
    exec_extend_lens: tuple[int, ...] = ()
    exec_sample_indices: tuple[int, ...] = ()
    exec_writebacks: tuple[bool, ...] = ()
    token_blocks: tuple[AfdTokenBlock, ...] = ()
    # sampling (AG samples now; built from the scheduled model plan)
    sampling: AfdSamplingPlan = field(
        default_factory=lambda: AfdSamplingPlan([], [], [])
    )
    # MoE directional dispatch bucket (num_max_dispatch_tokens_per_rank)
    dispatch_bucket: int = 0
    use_decode_graph: bool = False
    # Explicit paging (the coordinator owns page ids): per table_indices entry, the
    # request's cached length and how many pages it is granted this step; the granted
    # pages (token slot of each page's first token) follow table by table. A worker
    # writes them right after the pages it already holds for that table. Empty = the
    # worker allocates from its own mirror of the coordinator's free list.
    cached_lens: tuple[int, ...] = ()
    grant_counts: tuple[int, ...] = ()
    grant_pages: tuple[int, ...] = ()
    # Speculative verify (spec.config): an extend of the spec block per request whose
    # input ids are the AG token pool's bonus + draft tokens, so no token blocks.
    spec_verify: bool = False

    def __post_init__(self) -> None:
        self.step_id = int(self.step_id)
        self.real_size = int(self.real_size)
        self.spec_verify = bool(self.spec_verify)
        self.table_indices = tuple(int(x) for x in self.table_indices)
        self.extend_lens = tuple(int(x) for x in self.extend_lens)
        self.cached_lens = tuple(int(x) for x in self.cached_lens)
        self.grant_counts = tuple(int(x) for x in self.grant_counts)
        self.grant_pages = tuple(int(x) for x in self.grant_pages)
        if self.grant_counts and (
            len(self.grant_counts) != len(self.table_indices)
            or len(self.cached_lens) != len(self.table_indices)
            or sum(self.grant_counts) != len(self.grant_pages)
        ):
            raise ValueError(
                "AfdAGStepPlan page grants do not match its tables: "
                f"tables={len(self.table_indices)} cached_lens={len(self.cached_lens)} "
                f"counts={len(self.grant_counts)} pages={len(self.grant_pages)}"
            )
        self.free_table_indices = tuple(int(x) for x in self.free_table_indices)
        self.microbatch_offsets = tuple(int(x) for x in self.microbatch_offsets)
        self.microbatch_token_offsets = tuple(
            int(x) for x in self.microbatch_token_offsets
        )
        self.microbatch_real_token_counts = tuple(
            int(x) for x in self.microbatch_real_token_counts
        )
        self.dispatch_bucket = int(self.dispatch_bucket)
        self.use_decode_graph = bool(self.use_decode_graph)
        self.exec_table_indices = tuple(int(x) for x in self.exec_table_indices)
        self.exec_start_positions = tuple(int(x) for x in self.exec_start_positions)
        self.exec_extend_lens = tuple(int(x) for x in self.exec_extend_lens)
        self.exec_sample_indices = tuple(int(x) for x in self.exec_sample_indices)
        self.exec_writebacks = tuple(bool(x) for x in self.exec_writebacks)
        self.token_blocks = tuple(
            b if isinstance(b, AfdTokenBlock) else AfdTokenBlock(*b)
            for b in self.token_blocks
        )
        exec_count = len(self.exec_table_indices)
        if any(
            len(values) != exec_count
            for values in (
                self.exec_start_positions,
                self.exec_extend_lens,
                self.exec_sample_indices,
                self.exec_writebacks,
            )
        ):
            raise ValueError(
                "AfdAGStepPlan exec tuple lengths must match: "
                f"tables={len(self.exec_table_indices)} "
                f"starts={len(self.exec_start_positions)} "
                f"lens={len(self.exec_extend_lens)} "
                f"samples={len(self.exec_sample_indices)} "
                f"writebacks={len(self.exec_writebacks)}"
            )
        if any(length < 0 for length in self.exec_extend_lens):
            raise ValueError(
                "AfdAGStepPlan exec_extend_lens must be non-negative: "
                f"{self.exec_extend_lens}"
            )
        if len(self.microbatch_offsets) < 2:
            raise ValueError("AfdAGStepPlan requires at least one microbatch")
        if len(self.microbatch_offsets) != len(self.microbatch_token_offsets):
            raise ValueError(
                "AfdAGStepPlan req/token offset length mismatch: "
                f"req={self.microbatch_offsets} token={self.microbatch_token_offsets}"
            )
        if len(self.microbatch_real_token_counts) != self.num_mb:
            raise ValueError(
                "AfdAGStepPlan requires one real-token count per microbatch: "
                f"expected={self.num_mb} got={len(self.microbatch_real_token_counts)}"
            )
        expected_tokens = (
            int(self.microbatch_token_offsets[-1])
            if self.microbatch_token_offsets
            else 0
        )
        exec_tokens = sum(self.exec_extend_lens)
        if exec_tokens != expected_tokens:
            raise ValueError(
                "AfdAGStepPlan exec token count must match graph token count: "
                f"exec={exec_tokens} graph={expected_tokens}"
            )
        seen_sample_indices = [False] * self.real_size
        for sample_index in self.exec_sample_indices:
            if sample_index < 0:
                continue
            if sample_index >= self.real_size or seen_sample_indices[sample_index]:
                raise ValueError(
                    "AfdAGStepPlan sampled req indices must be dense [0, real_size): "
                    f"real_size={self.real_size} duplicate_or_oob={sample_index}"
            )
            seen_sample_indices[sample_index] = True
        if not all(seen_sample_indices):
            missing = [i for i, seen in enumerate(seen_sample_indices) if not seen][:8]
            raise ValueError(
                "AfdAGStepPlan sampled req indices must be dense [0, real_size): "
                f"real_size={self.real_size} missing={missing}"
            )
        if any(
            writeback and sample_index < 0
            for writeback, sample_index in zip(
                self.exec_writebacks,
                self.exec_sample_indices,
                strict=True,
            )
        ):
            raise ValueError("AfdAGStepPlan writeback reqs must have sample_index >= 0")

    @property
    def num_mb(self) -> int:
        return max(1, len(self.microbatch_offsets) - 1)


@dataclass
class AfdEGStepPlan:
    step_id: int
    phase: Literal["prefill", "decode"]
    dispatch_bucket: int
    num_mb: int = 1
    use_decode_graph: bool = False

    def __post_init__(self) -> None:
        self.step_id = int(self.step_id)
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"Unsupported AFD EG phase: {self.phase!r}")
        self.dispatch_bucket = int(self.dispatch_bucket)
        self.num_mb = max(1, int(self.num_mb))
        self.use_decode_graph = bool(self.use_decode_graph)


# ============================================================================
# Coordinator-local model plan builders.
#
# AfdModelPlan is not sent over the worker protocol. It is a coordinator-side
# intermediate built from a scheduled local attention-DP Batch, then compacted
# into AfdAGStepPlan or into the tiny global AfdEGStepPlan. Keep this block
# adjacent to the wire dataclasses so the exact translation boundary is visible.
# ============================================================================


def _require_microbatch_offsets(
    batch: Batch,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    req_offsets = batch.afd_microbatch_offsets
    token_offsets = batch.afd_microbatch_token_offsets
    if req_offsets is None or token_offsets is None:
        raise ValueError(
            "AfdModelPlan.from_batch requires batch.afd_microbatch_offsets "
            "and batch.afd_microbatch_token_offsets to be attached by the scheduler"
        )
    req_offsets = tuple(int(x) for x in req_offsets)
    token_offsets = tuple(int(x) for x in token_offsets)
    if req_offsets[-1] != int(batch.padded_size) or len(token_offsets) != len(req_offsets):
        raise ValueError(
            "inconsistent AFD microbatch offsets: "
            f"req={req_offsets} token={token_offsets} padded_size={int(batch.padded_size)}"
        )
    return req_offsets, token_offsets


def _count_batch_tokens(batch: Batch) -> int:
    return sum(int(req.extend_len) for req in batch.padded_reqs)


def _microbatch_real_token_counts(
    batch: Batch,
    req_offsets: tuple[int, ...],
) -> tuple[int, ...]:
    real_reqs = set(batch.reqs)
    return tuple(
        sum(
            int(req.extend_len)
            for req in batch.padded_reqs[int(req_offsets[mb]) : int(req_offsets[mb + 1])]
            if req in real_reqs
        )
        for mb in range(len(req_offsets) - 1)
    )


def _build_model_plan_exec(
    batch: Batch,
    token_count: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[AfdTokenBlock, ...],
]:
    real_sample_by_req = {req: idx for idx, req in enumerate(batch.reqs)}
    writeback_samples = {
        idx for idx, req in enumerate(batch.reqs) if req.can_decode
    }
    exec_table_indices: list[int] = []
    exec_start_positions: list[int] = []
    exec_extend_lens: list[int] = []
    exec_sample_indices: list[int] = []
    exec_writebacks: list[bool] = []
    token_blocks: list[AfdTokenBlock] = []
    cursor = 0
    is_decode = batch.phase == "decode"
    spec_verify = bool(getattr(batch, "afd_spec_verify", False))

    for req in batch.padded_reqs:
        length = int(req.extend_len)
        if length <= 0:
            continue
        if is_decode and length != 1:
            raise ValueError(
                "AFD decode model plan expects one token per padded request: "
                f"table_idx={int(req.table_idx)} extend_len={length}"
            )
        table_idx = int(req.table_idx)
        start_pos = int(req.cached_len)
        sample_index = int(real_sample_by_req.get(req, -1))
        exec_table_indices.append(table_idx)
        exec_start_positions.append(start_pos)
        exec_extend_lens.append(length)
        exec_sample_indices.append(sample_index)
        exec_writebacks.append(
            sample_index >= 0 and sample_index in writeback_samples
        )
        if not is_decode and not spec_verify:
            token_blocks.append(
                AfdTokenBlock(
                    table_idx=table_idx,
                    start_pos=start_pos,
                    tokens=req.input_ids[
                        int(req.cached_len) : int(req.device_len)
                    ].to(torch.int32).clone(),
                )
            )
        cursor += length

    if cursor != int(token_count):
        raise ValueError(
            "AFD model plan exec token count mismatch: "
            f"exec={cursor} token_count={int(token_count)}"
        )
    return (
        tuple(exec_table_indices),
        tuple(exec_start_positions),
        tuple(exec_extend_lens),
        tuple(exec_sample_indices),
        tuple(exec_writebacks),
        tuple(token_blocks),
    )


@dataclass
class AfdModelPlan:
    step_id: int
    phase: Literal["prefill", "decode"]
    size: int
    sampling: AfdSamplingPlan
    microbatch_offsets: tuple[int, ...]
    microbatch_token_offsets: tuple[int, ...]
    microbatch_real_token_counts: tuple[int, ...]
    exec_table_indices: tuple[int, ...]
    exec_start_positions: tuple[int, ...]
    exec_extend_lens: tuple[int, ...]
    exec_sample_indices: tuple[int, ...]
    exec_writebacks: tuple[bool, ...]
    token_blocks: tuple[AfdTokenBlock, ...] = ()
    spec_verify: bool = False

    def __post_init__(self) -> None:
        self.step_id = int(self.step_id)
        self.size = int(self.size)
        self.spec_verify = bool(self.spec_verify)
        self.microbatch_offsets = tuple(int(x) for x in self.microbatch_offsets)
        self.microbatch_token_offsets = tuple(int(x) for x in self.microbatch_token_offsets)
        self.microbatch_real_token_counts = tuple(
            int(x) for x in self.microbatch_real_token_counts
        )
        self.exec_table_indices = tuple(int(x) for x in self.exec_table_indices)
        self.exec_start_positions = tuple(int(x) for x in self.exec_start_positions)
        self.exec_extend_lens = tuple(int(x) for x in self.exec_extend_lens)
        self.exec_sample_indices = tuple(int(x) for x in self.exec_sample_indices)
        self.exec_writebacks = tuple(bool(x) for x in self.exec_writebacks)
        self.token_blocks = tuple(
            block if isinstance(block, AfdTokenBlock) else AfdTokenBlock(*block)
            for block in self.token_blocks
        )
        if len(self.microbatch_offsets) < 2:
            raise ValueError("AfdModelPlan requires at least one microbatch")
        if len(self.microbatch_offsets) != len(self.microbatch_token_offsets):
            raise ValueError(
                "AfdModelPlan req/token offset length mismatch: "
                f"req={self.microbatch_offsets} token={self.microbatch_token_offsets}"
            )
        if len(self.microbatch_real_token_counts) != self.num_mb:
            raise ValueError(
                "AfdModelPlan requires one real-token count per microbatch: "
                f"expected={self.num_mb} got={len(self.microbatch_real_token_counts)}"
            )
        if self.microbatch_offsets[0] != 0 or self.microbatch_token_offsets[0] != 0:
            raise ValueError(
                "AfdModelPlan microbatch offsets must start at zero: "
                f"req={self.microbatch_offsets} token={self.microbatch_token_offsets}"
            )
        if any(b < a for a, b in zip(self.microbatch_offsets, self.microbatch_offsets[1:])):
            raise ValueError(f"AfdModelPlan req offsets must be monotonic: {self.microbatch_offsets}")
        if any(
            b < a
            for a, b in zip(self.microbatch_token_offsets, self.microbatch_token_offsets[1:])
        ):
            raise ValueError(
                f"AfdModelPlan token offsets must be monotonic: {self.microbatch_token_offsets}"
            )
        for mb, real_tokens in enumerate(self.microbatch_real_token_counts):
            token_span = self.microbatch_token_offsets[mb + 1] - self.microbatch_token_offsets[mb]
            if real_tokens < 0 or real_tokens > token_span:
                raise ValueError(
                    f"AfdModelPlan real token count out of range for mb={mb}: "
                    f"real={real_tokens} span={token_span}"
                )
        if self.size > self.padded_size:
            raise ValueError(f"AfdModelPlan size={self.size} exceeds padded_size={self.padded_size}")
        exec_count = len(self.exec_table_indices)
        if any(
            len(values) != exec_count
            for values in (
                self.exec_start_positions,
                self.exec_extend_lens,
                self.exec_sample_indices,
                self.exec_writebacks,
            )
        ):
            raise ValueError(
                "AfdModelPlan exec tuple lengths must match: "
                f"tables={len(self.exec_table_indices)} "
                f"starts={len(self.exec_start_positions)} "
                f"lens={len(self.exec_extend_lens)} "
                f"samples={len(self.exec_sample_indices)} "
                f"writebacks={len(self.exec_writebacks)}"
            )
        if sum(self.exec_extend_lens) != self.graph_token_count:
            raise ValueError(
                "AfdModelPlan exec token count must match graph token count: "
                f"exec={sum(self.exec_extend_lens)} graph={self.graph_token_count}"
            )

    @property
    def padded_size(self) -> int:
        return int(self.microbatch_offsets[-1])

    @property
    def graph_token_count(self) -> int:
        return int(self.microbatch_token_offsets[-1])

    @property
    def num_mb(self) -> int:
        return len(self.microbatch_offsets) - 1

    @classmethod
    def from_batch(
        cls,
        step_id: int,
        batch: Batch,
        *,
        graph_token_count: int | None = None,
    ) -> AfdModelPlan:
        if batch.phase == "decode":
            return cls.from_decode_batch_fast(
                step_id,
                batch,
                graph_token_count=graph_token_count,
            )
        token_count = (
            int(graph_token_count)
            if graph_token_count is not None
            else _count_batch_tokens(batch)
        )
        actual_token_count = _count_batch_tokens(batch)
        if token_count != actual_token_count:
            raise ValueError(
                f"graph_token_count={int(graph_token_count)} does not match "
                f"batch token count {actual_token_count}"
            )
        req_offsets, token_offsets = _require_microbatch_offsets(batch)
        if batch.phase == "decode" and token_count != len(batch.padded_reqs):
            raise ValueError(
                "AFD decode model plan expects one read position per padded request: "
                f"tokens={token_count} padded_reqs={len(batch.padded_reqs)}"
            )
        (
            exec_table_indices,
            exec_start_positions,
            exec_extend_lens,
            exec_sample_indices,
            exec_writebacks,
            token_blocks,
        ) = _build_model_plan_exec(batch, token_count)
        return cls(
            step_id=int(step_id),
            phase=batch.phase,
            size=int(batch.size),
            sampling=AfdSamplingPlan.from_sampling_params(
                [req.sampling_params for req in batch.reqs]
            ),
            microbatch_offsets=req_offsets,
            microbatch_token_offsets=token_offsets,
            microbatch_real_token_counts=_microbatch_real_token_counts(batch, req_offsets),
            exec_table_indices=exec_table_indices,
            exec_start_positions=exec_start_positions,
            exec_extend_lens=exec_extend_lens,
            exec_sample_indices=exec_sample_indices,
            exec_writebacks=exec_writebacks,
            token_blocks=token_blocks,
            spec_verify=bool(getattr(batch, "afd_spec_verify", False)),
        )

    @classmethod
    def from_decode_batch_fast(
        cls,
        step_id: int,
        batch: Batch,
        *,
        graph_token_count: int | None = None,
    ) -> AfdModelPlan:
        """Build the coordinator-local decode plan without dataclass revalidation.

        Decode is one flat token per padded request. The centralized scheduler
        has already attached balanced microbatch offsets and owns the dummy Req,
        so this hot path can avoid the generic prefill/token-block machinery.
        """
        if batch.phase != "decode":
            raise ValueError(f"decode fast path got phase={batch.phase!r}")
        req_offsets, token_offsets = _require_microbatch_offsets(batch)
        padded_reqs = batch.padded_reqs
        token_count = len(padded_reqs)
        if graph_token_count is not None and int(graph_token_count) != token_count:
            raise ValueError(
                f"graph_token_count={int(graph_token_count)} does not match "
                f"decode padded token count {token_count}"
            )
        real_index_by_id = {id(req): idx for idx, req in enumerate(batch.reqs)}
        exec_table_indices: list[int] = []
        exec_start_positions: list[int] = []
        exec_sample_indices: list[int] = []
        exec_writebacks: list[bool] = []
        for req in padded_reqs:
            if int(req.extend_len) != 1:
                raise ValueError(
                    "AFD decode model plan expects one token per padded request: "
                    f"table_idx={int(req.table_idx)} extend_len={int(req.extend_len)}"
                )
            sample_index = int(real_index_by_id.get(id(req), -1))
            exec_table_indices.append(int(req.table_idx))
            exec_start_positions.append(int(req.cached_len))
            exec_sample_indices.append(sample_index)
            exec_writebacks.append(sample_index >= 0)
        real_token_counts = tuple(
            sum(
                1
                for sample_index in exec_sample_indices[
                    int(req_offsets[mb]) : int(req_offsets[mb + 1])
                ]
                if sample_index >= 0
            )
            for mb in range(len(req_offsets) - 1)
        )
        plan = object.__new__(cls)
        plan.step_id = int(step_id)
        plan.phase = "decode"
        plan.size = int(batch.size)
        plan.sampling = AfdSamplingPlan.from_sampling_params(
            req.sampling_params for req in batch.reqs
        )
        plan.microbatch_offsets = req_offsets
        plan.microbatch_token_offsets = token_offsets
        plan.microbatch_real_token_counts = real_token_counts
        plan.exec_table_indices = tuple(exec_table_indices)
        plan.exec_start_positions = tuple(exec_start_positions)
        plan.exec_extend_lens = (1,) * token_count
        plan.exec_sample_indices = tuple(exec_sample_indices)
        plan.exec_writebacks = tuple(exec_writebacks)
        plan.token_blocks = ()
        plan.spec_verify = False
        return plan


def build_afd_ag_plan(
    step_id: int,
    model_plan: AfdModelPlan,
    batch: Batch,
    *,
    dispatch_bucket: int,
    free_table_indices: tuple[int, ...] = (),
    use_decode_graph: bool = False,
    grants: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = ((), (), ()),
) -> AfdAGStepPlan:
    """Assemble the AG step plan from a scheduled batch + its model plan.
    `grants` = (cached_lens, grant_counts, grant_pages), see AfdAGStepPlan."""
    table_indices = tuple(int(req.table_idx) for req in batch.reqs)
    extend_lens = tuple(int(req.extend_len) for req in batch.reqs)
    cached_lens, grant_counts, grant_pages = (tuple(int(x) for x in g) for g in grants)
    if model_plan.phase == "decode" and not model_plan.token_blocks:
        plan = object.__new__(AfdAGStepPlan)
        plan.step_id = int(step_id)
        plan.phase = "decode"
        plan.real_size = int(batch.size)
        plan.table_indices = table_indices
        plan.extend_lens = extend_lens
        plan.free_table_indices = tuple(int(x) for x in free_table_indices)
        plan.microbatch_offsets = model_plan.microbatch_offsets
        plan.microbatch_token_offsets = model_plan.microbatch_token_offsets
        plan.microbatch_real_token_counts = model_plan.microbatch_real_token_counts
        plan.exec_table_indices = model_plan.exec_table_indices
        plan.exec_start_positions = model_plan.exec_start_positions
        plan.exec_extend_lens = model_plan.exec_extend_lens
        plan.exec_sample_indices = model_plan.exec_sample_indices
        plan.exec_writebacks = model_plan.exec_writebacks
        plan.token_blocks = ()
        plan.sampling = model_plan.sampling
        plan.dispatch_bucket = int(dispatch_bucket)
        plan.use_decode_graph = bool(use_decode_graph)
        plan.cached_lens = cached_lens
        plan.grant_counts = grant_counts
        plan.grant_pages = grant_pages
        plan.spec_verify = False
        return plan
    return AfdAGStepPlan(
        step_id=int(step_id),
        phase=str(model_plan.phase),  # type: ignore[arg-type]
        real_size=int(batch.size),
        table_indices=table_indices,
        extend_lens=extend_lens,
        free_table_indices=tuple(int(x) for x in free_table_indices),
        microbatch_offsets=tuple(model_plan.microbatch_offsets),
        microbatch_token_offsets=tuple(model_plan.microbatch_token_offsets),
        microbatch_real_token_counts=model_plan.microbatch_real_token_counts,
        exec_table_indices=model_plan.exec_table_indices,
        exec_start_positions=model_plan.exec_start_positions,
        exec_extend_lens=model_plan.exec_extend_lens,
        exec_sample_indices=model_plan.exec_sample_indices,
        exec_writebacks=model_plan.exec_writebacks,
        token_blocks=model_plan.token_blocks,
        sampling=model_plan.sampling,
        dispatch_bucket=int(dispatch_bucket),
        use_decode_graph=bool(use_decode_graph),
        cached_lens=cached_lens,
        grant_counts=grant_counts,
        grant_pages=grant_pages,
        spec_verify=bool(model_plan.spec_verify),
    )


# ============================================================================
# End coordinator-local model plan builders.
# ============================================================================


@dataclass
class AfdRunAGStepCmd(AfdCommand):
    plan: AfdAGStepPlan
    sent_ns: int = 0


@dataclass
class AfdRunEGStepCmd(AfdCommand):
    plan: AfdEGStepPlan
    sent_ns: int = 0


@dataclass
class AfdAGStepReply(AfdReply):
    next_tokens: list[int] = field(default_factory=list)


@dataclass
class AfdEGStepReply(AfdReply):
    pass


# ============================================================================
# CAD decode (minisgl.cad, MINISGL_AFD_MODE=cad). Prefill steps stay AFD
# steps (AfdRunAGStepCmd / AfdRunEGStepCmd); decode steps go as one AfdRunCadStepCmd per GPU, no EG command.
# ============================================================================


@dataclass
class AfdCadInitCmd(AfdCommand):
    """Every attention and model rank, once, before the first step (collective in this order on every rank:
    minisgl.cad.groups.cad_groups, CadWindow.allocate over the union world, [model ranks only: the dual-role MegaMoE
    rendezvous over the model group + its EG precompile], CadNodeRuntime.warm(sorted buckets)). The window is the FIRST
    symm_mem rendezvous of the CAD init on every rank: torch keys every rendezvous by one per-process sequence number,
    so a model-only rendezvous before it would leave the ranks on different keys.
    MINISGL_AFD_CAD_INIT_TIMEOUT_S (default 900) bounds the coordinator's wait for every rank's ready reply. buckets: the per-mb decode buckets (rows of
    one attention GPU in one microbatch; the coordinator's decode_graph_bs); t_max = max(buckets); table_rows: the
    attention GPUs' request-table rows (the model GPU's last-token slots, minisgl.cad.tokens)."""

    num_mb: int
    t_max: int
    buckets: tuple[int, ...]
    table_rows: int
    use_graph: bool = True


@dataclass
class AfdCadReadyReply(AfdReply):
    step_id: int | None = None
    captured: tuple[int, ...] = ()   # the buckets this GPU captured a graph for


@dataclass
class AfdRunCadStepCmd(AfdCommand):
    """One CAD decode step for one GPU. An attention rank gets plans = [its own AfdAGStepPlan] (KV grants and the
    materialized sub-batches, as an AFD AG step, kv_only); model GPU j gets plans = the r plans of its attention GPUs in
    peer order (minisgl.cad.layout.peers_of) and model_rows = one minisgl.cad.tokens.CadRow per real request of those
    plans. bucket: the per-mb rows of every attention GPU (uniform over the step; the model GPU's mb has r x bucket)."""

    step_id: int
    plans: list[AfdAGStepPlan]
    bucket: int
    use_graph: bool
    model_rows: tuple = ()
    sent_ns: int = 0


@dataclass
class AfdCadStepReply(AfdReply):
    """Model GPU: tokens = {peer: the step's sampled tokens of that peer's requests, in model_rows order}; attention
    rank: tokens = {}. err: the window's abort word after the step (non-zero = a flag wait timed out)."""

    tokens: dict = field(default_factory=dict)
    err: int = 0


def cad_microbatch_rows(padded_reqs: list, real_reqs: list, microbatch_offsets: tuple[int, ...]) -> list[tuple[int, int]]:
    """(mb, i) of every real request (in real_reqs order) in its attention GPU's materialized sub-batches: the
    coordinator's padded layout (afd_scheduler._pad_and_split_batch: per mb a real prefix + dummy tail when balanced,
    else real requests first) is the one the attention runtime rebuilds (afd_attention_runtime._build_padded_reqs)."""
    where: dict[int, tuple[int, int]] = {}
    offs = tuple(int(x) for x in microbatch_offsets)
    for mb in range(len(offs) - 1):
        for i, req in enumerate(padded_reqs[offs[mb]:offs[mb + 1]]):
            where.setdefault(id(req), (mb, i))
    out = []
    for req in real_reqs:
        if id(req) not in where:
            raise ValueError(f"CAD: request table {getattr(req, 'table_idx', '?')} is not in the padded batch")
        out.append(where[id(req)])
    return out


# AFD ZMQ queues use pickle. Most commands/replies are faster and clearer as
# plain pickle payloads. The only tensor-bearing command is prefill's
# AfdRunAGStepCmd, whose token_blocks carry CPU input-id tensors; encode just
# those tensors as raw bytes to avoid PyTorch tensor pickle overhead.
def _encode_token_block(block: AfdTokenBlock) -> dict[str, Any]:
    return {
        _TOKEN_BLOCK_CODEC_TAG: True,
        "table_idx": int(block.table_idx),
        "start_pos": int(block.start_pos),
        "tokens": _encode_tensor(block.tokens),
    }


def _decode_token_block(payload: Any) -> AfdTokenBlock:
    if isinstance(payload, AfdTokenBlock):
        return payload
    if not isinstance(payload, dict) or payload.get(_TOKEN_BLOCK_CODEC_TAG) is not True:
        raise TypeError(f"Unsupported AFD token block payload: {type(payload)!r}")
    return AfdTokenBlock(
        table_idx=int(payload["table_idx"]),
        start_pos=int(payload["start_pos"]),
        tokens=_decode_tensor(payload["tokens"]),
    )


def _encode_ag_step_cmd(cmd: AfdRunAGStepCmd) -> dict[str, Any]:
    plan = cmd.plan
    plan_payload = {
        dataclass_field.name: getattr(plan, dataclass_field.name)
        for dataclass_field in fields(AfdAGStepPlan)
    }
    plan_payload["token_blocks"] = tuple(
        _encode_token_block(block) for block in plan.token_blocks
    )
    return {
        _AG_STEP_CMD_CODEC_TAG: True,
        "sent_ns": int(cmd.sent_ns),
        "plan": plan_payload,
    }


def _decode_ag_step_cmd(payload: dict[str, Any]) -> AfdRunAGStepCmd:
    plan_payload = dict(payload["plan"])
    plan_payload["token_blocks"] = tuple(
        _decode_token_block(block) for block in plan_payload.get("token_blocks", ())
    )
    return AfdRunAGStepCmd(
        plan=AfdAGStepPlan(**plan_payload),
        sent_ns=int(payload.get("sent_ns", 0)),
    )


def _encode_afd_payload(payload: Any) -> Any:
    if isinstance(payload, AfdRunAGStepCmd) and payload.plan.token_blocks:
        return _encode_ag_step_cmd(payload)
    return payload


def _decode_afd_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and payload.get(_AG_STEP_CMD_CODEC_TAG) is True:
        return _decode_ag_step_cmd(payload)
    return payload


def _decode_instance(payload: Any, cls: type[Any]) -> Any:
    payload = _decode_afd_payload(payload)
    if not isinstance(payload, cls):
        raise TypeError(f"Expected {cls.__name__} payload, got {type(payload)!r}")
    return payload


def _decode_afd_command(payload: Any) -> AfdCommand:
    return _decode_instance(payload, AfdCommand)


def _decode_afd_reply(payload: Any) -> AfdReply:
    return _decode_instance(payload, AfdReply)


AfdCommand.encoder = staticmethod(_encode_afd_payload)  # type: ignore[attr-defined]
AfdCommand.decoder = staticmethod(_decode_afd_command)  # type: ignore[attr-defined]
AfdReply.encoder = staticmethod(_encode_afd_payload)  # type: ignore[attr-defined]
AfdReply.decoder = staticmethod(_decode_afd_reply)  # type: ignore[attr-defined]


__all__ = [
    "AfdAGStepPlan",
    "AfdCadInitCmd",
    "AfdCadReadyReply",
    "AfdCadStepReply",
    "AfdRunCadStepCmd",
    "cad_microbatch_rows",
    "AfdEGStepPlan",
    "AfdModelPlan",
    "AfdRunAGStepCmd",
    "AfdRunEGStepCmd",
    "AfdAGStepReply",
    "AfdEGStepReply",
    "AfdCommand",
    "AfdTopology",
    "AfdTokenBlock",
    "AfdWorkerReadyReply",
    "AfdReply",
    "AfdFlushStepCmd",
    "AfdProfilerCmd",
    "AfdSamplingPlan",
    "AfdStopCmd",
    "build_afd_ag_plan",
]
