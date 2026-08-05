from __future__ import annotations

from typing import Any

import torch

from minisgl.kernel.nvfp4_moe import (
    NvFp4PreparedMoEWeights,
    prepare_nvfp4_moe_weights,
    run_nvfp4_moe,
)

from .base import MoeRunner, MoeRunnerBackend, MoeRunnerConfig


def _deepep_valid_rows(dispatch_output: Any) -> torch.Tensor | None:
    handle = getattr(dispatch_output, "handle", None)
    runtime_handle = getattr(handle, "handle", None)
    psum = getattr(runtime_handle, "psum_num_recv_tokens_per_scaleup_rank", None)
    if psum is None:
        return None
    rows = torch.arange(
        dispatch_output.hidden_states.shape[0],
        device=dispatch_output.hidden_states.device,
        dtype=psum.dtype,
    )
    return rows < psum[-1]


def normalize_deepep_nvfp4_topk(
    dispatch_output: Any,
    layer: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert DeepEP rank-local routes into the model-global NVFP4 contract.

    Non-expanded DeepEP output is graph-static: rows beyond the device-side
    receive count are allocation padding.  Mask those rows without a host
    ``item()`` synchronization, preserve ``-1`` route sentinels, and translate
    valid receiving-rank-local expert ids to the global ids expected by the
    FlashInfer EP launcher.
    """

    topk_ids = getattr(dispatch_output, "topk_ids", None)
    topk_weights = getattr(dispatch_output, "topk_weights", None)
    if topk_ids is None or topk_weights is None:
        raise RuntimeError("NVFP4 DeepEP dispatch requires explicit top-k metadata")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise RuntimeError(
            "NVFP4 DeepEP top-k shape mismatch: "
            f"ids={tuple(topk_ids.shape)} weights={tuple(topk_weights.shape)}"
        )

    ids = topk_ids.to(dtype=torch.int32)
    valid = ids >= 0
    local_num_experts = int(layer.local_num_experts)
    valid &= ids < local_num_experts
    valid_rows = _deepep_valid_rows(dispatch_output)
    if valid_rows is not None:
        valid &= valid_rows[:, None]
    global_ids = torch.where(
        valid,
        ids + int(layer.local_expert_offset),
        torch.full_like(ids, -1),
    ).contiguous()
    weights = torch.where(
        valid,
        topk_weights.to(dtype=torch.float32),
        torch.zeros((), device=topk_weights.device, dtype=torch.float32),
    ).contiguous()
    return weights, global_ids


class FlashInferNvFp4Runner(MoeRunner):
    """Blackwell native ModelOpt NVFP4 routed-expert runner."""

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.FLASHINFER_NVFP4

    @staticmethod
    def _topk(dispatch_output: Any, layer: Any) -> tuple[torch.Tensor, torch.Tensor]:
        topk_output = getattr(dispatch_output, "topk_output", None)
        if isinstance(topk_output, tuple) and len(topk_output) == 2:
            return topk_output
        topk_ids = getattr(dispatch_output, "topk_ids", None)
        topk_weights = getattr(dispatch_output, "topk_weights", None)
        if topk_ids is not None and topk_weights is not None:
            if _deepep_valid_rows(dispatch_output) is not None:
                return normalize_deepep_nvfp4_topk(dispatch_output, layer)
            return topk_weights, topk_ids
        if isinstance(topk_output, torch.Tensor):
            from minisgl.moe.fused import fused_topk

            return fused_topk(
                hidden_states=dispatch_output.hidden_states,
                gating_output=topk_output,
                topk=layer.top_k,
                renormalize=layer.renormalize,
            )
        raise RuntimeError("NVFP4 runner requires router logits or explicit top-k tensors")

    @staticmethod
    def _prepare(layer: Any) -> NvFp4PreparedMoEWeights:
        cached = getattr(layer, "_nvfp4_prepared", None)
        if cached is not None:
            return cached
        prepared = prepare_nvfp4_moe_weights(
            layer.gate_up_proj,
            layer.gate_up_proj_scale,
            layer.gate_up_proj_weight_scale_2,
            layer.gate_up_proj_input_scale,
            layer.down_proj,
            layer.down_proj_scale,
            layer.down_proj_weight_scale_2,
            layer.down_proj_input_scale,
            global_num_experts=layer.num_experts,
            local_expert_offset=layer.local_expert_offset,
        )
        # Replace serialized tensors with their native reordered equivalents,
        # then release all now-fused global-scale inputs.  The private dataclass
        # owns the remaining FP32 scale vectors used by the kernel.
        layer.gate_up_proj = prepared.gate_up_proj
        layer.gate_up_proj_scale = prepared.gate_up_proj_scale
        layer.down_proj = prepared.down_proj
        layer.down_proj_scale = prepared.down_proj_scale
        layer.gate_up_proj_weight_scale_2 = None
        layer.gate_up_proj_input_scale = None
        layer.down_proj_weight_scale_2 = None
        layer.down_proj_input_scale = None
        layer._nvfp4_prepared = prepared
        return prepared

    def prewarm(self, layer: Any, *, dtype: torch.dtype = torch.bfloat16) -> None:
        """Prepare one layer and compile/launch the shared MoE kernel before serving."""

        prepared = self._prepare(layer)
        top_k = int(self.config.top_k or layer.top_k)
        local_num_experts = int(layer.local_num_experts)
        if top_k > local_num_experts:
            raise RuntimeError(
                "NVFP4 prewarm requires at least top_k local experts: "
                f"top_k={top_k} local_num_experts={local_num_experts}"
            )
        device = prepared.gate_up_proj.device
        hidden_states = torch.zeros(
            (1, prepared.hidden_size), device=device, dtype=dtype
        )
        topk_ids = (
            torch.arange(top_k, device=device, dtype=torch.int32)
            + int(layer.local_expert_offset)
        ).unsqueeze(0)
        topk_weights = torch.full(
            (1, top_k), 1.0 / top_k, device=device, dtype=torch.float32
        )
        output = run_nvfp4_moe(
            prepared,
            hidden_states,
            topk_ids,
            topk_weights,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del output

    def apply(self, dispatch_output: Any, layer: Any) -> torch.Tensor:
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states
        valid_rows = _deepep_valid_rows(dispatch_output)
        if valid_rows is not None:
            hidden_states = torch.where(
                valid_rows[:, None],
                hidden_states,
                torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
            )
        topk_weights, topk_ids = self._topk(dispatch_output, layer)
        prepared = self._prepare(layer)
        return run_nvfp4_moe(
            prepared,
            hidden_states,
            topk_ids,
            topk_weights,
        )


__all__ = ["FlashInferNvFp4Runner", "normalize_deepep_nvfp4_topk"]
