from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tvm_ffi import Module

_EMPTY_VALID_COUNT_BY_DEVICE: dict[tuple[str, int | None], torch.Tensor] = {}


def _empty_valid_count(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    tensor = _EMPTY_VALID_COUNT_BY_DEVICE.get(key)
    if tensor is None:
        tensor = torch.empty((0,), dtype=torch.int64, device=device)
        _EMPTY_VALID_COUNT_BY_DEVICE[key] = tensor
    return tensor


@functools.cache
def _jit_moe_topk_softmax_module() -> Module:
    from .utils import load_jit

    return load_jit(
        "moe_topk_softmax",
        cuda_files=["moe_topk_softmax.cu"],
        cuda_wrappers=[
            ("launch", "MoeTopkSoftmaxKernel::run"),
            ("launch_group_local", "MoeTopkSoftmaxGroupLocalKernel::run"),
            ("launch_v2", "MoeTopkGatingV2Kernel::run"),
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


@functools.cache
def _jit_gate_topk_fused_module() -> Module:
    from .utils import load_jit

    return load_jit(
        "gate_topk_fused",
        cuda_files=["gate_topk_fused.cu"],
        cuda_wrappers=[("launch", "GateTopkFusedKernel::run")],
    )


@functools.cache
def _jit_minimax_route_fused_module() -> Module:
    from .utils import load_jit

    return load_jit(
        "minimax_route_fused",
        cuda_files=["minimax_route_fused.cu"],
        cuda_wrappers=[
            ("launch_logits", "MiniMaxRouteFusedKernel::run_logits"),
            ("launch_logits4", "MiniMaxRouteFusedKernel::run_logits4"),
            ("launch_logits16", "MiniMaxRouteFusedKernel::run_logits16"),
            ("launch_logits4_quant", "MiniMaxRouteFusedKernel::run_logits4_quant"),
            ("launch_logits16_quant", "MiniMaxRouteFusedKernel::run_logits16_quant"),
            (
                "launch_logits16_quant_sigmoid",
                "MiniMaxRouteFusedKernel::run_logits16_quant_sigmoid",
            ),
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


_GATE_TOPK_WS: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
_MINIMAX_ROUTE_LOGITS_WS: dict[tuple[str, int | None], torch.Tensor] = {}


def _gate_topk_workspace(device: torch.device, num_mtiles: int):
    """Persistent (graph-stable) K-split partial buffer + self-resetting
    ticket counters for the fused gate kernel."""
    key = (device.type, device.index)
    ws = _GATE_TOPK_WS.get(key)
    if ws is None or ws[0].shape[1] < num_mtiles:
        cap = max(num_mtiles, 32)
        partials = torch.empty((16, cap, 16, 128), dtype=torch.float32, device=device)
        counters = torch.zeros((cap,), dtype=torch.int32, device=device)
        _GATE_TOPK_WS[key] = (partials, counters)
        ws = _GATE_TOPK_WS[key]
    return ws


def gate_topk(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
    quant_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused router for decode-sized batches: bf16 gate GEMM (tensor cores,
    16-way K-split) + softmax + top-k (+ optional renormalize) in ONE kernel
    — replaces the cublas gate GEMM + splitK + topk_gating launches.
    Returns (ids int32, weights fp32), both [num_tokens, top_k].

    ``quant_out`` = (x_q u8 [n, hidden], x_sf u8 [n, hidden/32],
    w_stage f32 [n, top_k]) additionally emits the per-32 FP8 quant of
    ``hidden`` (bit-identical to the MegaMoE AG kernel's phase-0) and a copy
    of the top-k weights — the comm kernel then skips its quant phase."""
    assert hidden.is_cuda and hidden.dtype == torch.bfloat16
    assert gate_weight.is_cuda and gate_weight.dtype == torch.bfloat16
    n = hidden.shape[0]
    ids = torch.empty((n, top_k), dtype=torch.int32, device=hidden.device)
    weights = torch.empty((n, top_k), dtype=torch.float32, device=hidden.device)
    if n == 0:
        return ids, weights
    partials, counters = _gate_topk_workspace(hidden.device, (n + 15) // 16)
    module = _jit_gate_topk_fused_module()
    x_q = x_sf = w_stage = None
    if quant_out is not None:
        x_q, x_sf, w_stage = quant_out
    module.launch(
        weights, ids,
        hidden.contiguous(), gate_weight.contiguous(),
        partials, counters,
        bool(renormalize),
        x_q, x_sf, w_stage,
    )
    return ids, weights


def _minimax_route_logits_workspace(
    device: torch.device,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    key = (device.type, device.index)
    ws = _MINIMAX_ROUTE_LOGITS_WS.get(key)
    if ws is None or ws.shape[0] < num_tokens or ws.shape[1] != num_experts:
        cap = max(num_tokens, 512)
        ws = torch.empty((cap, num_experts), dtype=torch.float32, device=device)
        _MINIMAX_ROUTE_LOGITS_WS[key] = ws
    return ws[:num_tokens]


def minimax_route_topk_fp32_dot_topk(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
    warps_per_cta: int = 0,
    quant_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    pre_sigmoid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MiniMax fp32 route using a custom dot kernel plus optimized topk_gating.

    This preserves the fp32 gate path while replacing:
      hidden.to(fp32) + cublasLt split-K GEMM + splitKreduce + topk_gating
    with:
      fp32 dot logits kernel + topk_gating.
    """
    assert hidden.is_cuda and hidden.dtype == torch.bfloat16
    assert gate_weight.is_cuda and gate_weight.dtype == torch.float32
    assert bias.is_cuda and bias.dtype == torch.float32
    hidden_2d = hidden.contiguous().view(-1, hidden.shape[-1])
    weight = gate_weight.contiguous()
    bias = bias.contiguous()
    n = hidden_2d.shape[0]
    ids = torch.empty((n, top_k), dtype=torch.int32, device=hidden.device)
    if quant_out is None:
        pre_sigmoid = False
        weights = torch.empty((n, top_k), dtype=torch.float32, device=hidden.device)
        x_q = x_sf = None
    else:
        x_q, x_sf, weights = quant_out
        assert weights.is_cuda and weights.dtype == torch.float32
        assert weights.is_contiguous()
        assert tuple(weights.shape) == (n, top_k)
    if n == 0:
        return ids, weights
    logits = _minimax_route_logits_workspace(hidden.device, n, weight.shape[0])
    module = _jit_minimax_route_fused_module()
    if warps_per_cta == 0:
        warps_per_cta = 16
    if quant_out is not None:
        if pre_sigmoid:
            module.launch_logits16_quant_sigmoid(logits, hidden_2d, weight, x_q, x_sf)
        elif warps_per_cta == 4:
            module.launch_logits4_quant(logits, hidden_2d, weight, x_q, x_sf)
        elif warps_per_cta == 16:
            module.launch_logits16_quant(logits, hidden_2d, weight, x_q, x_sf)
        else:
            # The production path uses 16 warps; keep non-standard settings on
            # the normal logits-only path to avoid silently skipping quant.
            module.launch_logits16_quant(logits, hidden_2d, weight, x_q, x_sf)
    elif warps_per_cta == 4:
        module.launch_logits4(logits, hidden_2d, weight)
    elif warps_per_cta == 16:
        module.launch_logits16(logits, hidden_2d, weight)
    else:
        module.launch_logits(logits, hidden_2d, weight)
    topk_gating(
        weights,
        ids,
        logits,
        scoring="identity" if pre_sigmoid else "sigmoid",
        bias=bias,
        renormalize=renormalize,
    )
    return ids, weights


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """Plain softmax + topk gating. Power-of-2 num_experts ≤ 256 routes through
    the fused v2 kernel; non-pow-of-2 / >256 falls back to softmax+topk via
    workspace path (kept for completeness, unused in production models)."""
    assert topk_weights.is_cuda and topk_ids.is_cuda and gating_output.is_cuda
    assert topk_weights.is_contiguous() and topk_ids.is_contiguous()
    assert gating_output.is_contiguous()
    assert topk_weights.dtype == torch.float32 and topk_ids.dtype == torch.int32
    assert gating_output.dim() == 2 and topk_weights.dim() == 2 and topk_ids.dim() == 2
    assert gating_output.shape[0] == topk_weights.shape[0] == topk_ids.shape[0]
    assert topk_weights.shape[1] == topk_ids.shape[1]

    if gating_output.dtype not in (torch.float32, torch.bfloat16):
        gating_output = gating_output.float()
    num_tokens, num_experts = gating_output.shape
    if num_tokens == 0:
        return

    is_pow2_small = (num_experts & (num_experts - 1)) == 0 and num_experts <= 256
    if is_pow2_small:
        topk_gating(topk_weights, topk_ids, gating_output,
                    scoring="softmax", renormalize=renormalize)
        return
    if gating_output.dtype != torch.float32:
        gating_output = gating_output.float()

    workspace = torch.empty(
        (num_tokens * num_experts,), dtype=torch.float32, device=gating_output.device
    )
    module = _jit_moe_topk_softmax_module()
    module.launch(topk_weights, topk_ids, gating_output, workspace, renormalize)


def topk_gating(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    *,
    scoring: str = "softmax",
    bias: torch.Tensor | None = None,
    renormalize: bool = False,
    weight_stage: torch.Tensor | None = None,
) -> None:
    """Fused top-K gating with selectable scoring (softmax | sigmoid) and
    optional bias correction (used during argmax only; output weights are the
    unbiased scores).

    Power-of-two num_experts ≤ 256 only (matches existing fast path).
    Reduces to topk_softmax behavior when scoring='softmax' and bias=None.
    """
    assert topk_weights.is_cuda and topk_ids.is_cuda and gating_output.is_cuda
    assert topk_weights.is_contiguous() and topk_ids.is_contiguous()
    assert gating_output.is_contiguous()
    assert topk_weights.dtype == torch.float32 and topk_ids.dtype == torch.int32
    assert gating_output.dim() == 2
    if gating_output.dtype not in (torch.float32, torch.bfloat16):
        gating_output = gating_output.float()
    num_tokens, num_experts = gating_output.shape
    if num_tokens == 0:
        return
    scoring_id = {"softmax": 0, "sigmoid": 1, "identity": 2}[scoring]
    if bias is None:
        bias_t = torch.empty((0,), dtype=torch.float32, device=gating_output.device)
    else:
        assert bias.is_cuda and bias.dtype == torch.float32 and bias.is_contiguous()
        assert bias.dim() == 1 and bias.shape[0] == num_experts
        bias_t = bias
    if weight_stage is not None:
        assert weight_stage.is_cuda and weight_stage.dtype == torch.float32
        assert weight_stage.is_contiguous()
        assert tuple(weight_stage.shape) == tuple(topk_weights.shape)
    module = _jit_moe_topk_softmax_module()
    module.launch_v2(
        topk_weights, topk_ids, gating_output, bias_t,
        int(scoring_id), bool(renormalize),
        weight_stage,
    )


def _valid_token_count_tensor(
    num_token_non_padded: int | torch.Tensor | None,
    *,
    device: torch.device,
    num_tokens: int,
) -> torch.Tensor:
    if num_token_non_padded is None:
        return _empty_valid_count(device)
    if isinstance(num_token_non_padded, int):
        if num_token_non_padded >= num_tokens:
            return _empty_valid_count(device)
        return torch.tensor([num_token_non_padded], dtype=torch.int64, device=device)
    assert num_token_non_padded.is_cuda, "num_token_non_padded must be on CUDA"
    if num_token_non_padded.numel() != 1:
        raise RuntimeError(
            "num_token_non_padded tensor must contain one element, "
            f"got shape={tuple(num_token_non_padded.shape)}"
        )
    if num_token_non_padded.dtype != torch.int64:
        num_token_non_padded = num_token_non_padded.to(torch.int64)
    return num_token_non_padded.reshape(1)


def topk_softmax_group_local(
    topk_weights: torch.Tensor,
    group_topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    expert_map: torch.Tensor,
    num_token_non_padded: int | torch.Tensor | None = None,
    renormalize: bool = False,
) -> None:
    assert topk_weights.is_cuda, "topk_weights must be on CUDA"
    assert group_topk_ids.is_cuda, "group_topk_ids must be on CUDA"
    assert gating_output.is_cuda, "gating_output must be on CUDA"
    assert expert_map.is_cuda, "expert_map must be on CUDA"
    assert topk_weights.is_contiguous(), "topk_weights must be contiguous"
    assert group_topk_ids.is_contiguous(), "group_topk_ids must be contiguous"
    assert gating_output.is_contiguous(), "gating_output must be contiguous"
    assert expert_map.is_contiguous(), "expert_map must be contiguous"
    assert topk_weights.dtype == torch.float32, "topk_weights must be float32"
    assert group_topk_ids.dtype in (
        torch.int32,
        torch.int64,
    ), "group_topk_ids must be int32 or int64"
    assert expert_map.dtype in (
        torch.int32,
        torch.int64,
    ), "expert_map must be int32 or int64"
    assert gating_output.dim() == 2, "gating_output must be 2D"
    assert topk_weights.dim() == 2, "topk_weights must be 2D"
    assert group_topk_ids.dim() == 2, "group_topk_ids must be 2D"
    assert gating_output.shape[0] == topk_weights.shape[0] == group_topk_ids.shape[0]
    assert topk_weights.shape[1] == group_topk_ids.shape[1]
    assert expert_map.shape[0] == gating_output.shape[1]

    assert gating_output.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "gating_output must be float32 or bfloat16"

    num_tokens, num_experts = gating_output.shape
    if num_tokens == 0:
        return
    if (num_experts & (num_experts - 1)) != 0 or num_experts > 512:
        raise RuntimeError(
            "topk_softmax_group_local supports only power-of-two num_experts <= 512, "
            f"got {num_experts}"
        )

    valid_count = _valid_token_count_tensor(
        num_token_non_padded,
        device=gating_output.device,
        num_tokens=int(num_tokens),
    )
    module = _jit_moe_topk_softmax_module()
    module.launch_group_local(
        topk_weights,
        group_topk_ids,
        gating_output,
        expert_map,
        valid_count,
        renormalize,
    )
