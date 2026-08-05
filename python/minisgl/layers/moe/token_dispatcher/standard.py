from __future__ import annotations

from typing import Any, NamedTuple

import torch

from .base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from ..moe_runner.base import MoeRunnerConfig
from ..permute_registry import register_post_permute, register_pre_permute

class StandardDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    topk_output: Any
    use_expert_map_override: bool = False
    expert_map_override: torch.Tensor | None = None
    global_num_experts_override: int | None = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)


class StandardDispatcher(BaseDispatcher):
    """No-DeepEP dispatcher.

    dp_size == 1 is pass-through.  dp_size > 1 uses AgRs across the MLP-DP
    group with the same TP rank: all-gather tokens before local expert compute,
    then reduce-scatter back to each DP's token shard.
    """

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        dp_size: int = 1,
    ):
        self.moe_runner_config = moe_runner_config
        self.dp_size = int(dp_size)

    @staticmethod
    def _capturing_cuda_graph() -> bool:
        return bool(
            torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )

    @staticmethod
    def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
        if x.shape[0] == rows:
            return x.contiguous()
        out = x.new_zeros((rows, *x.shape[1:]))
        out[: x.shape[0]].copy_(x)
        return out

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: Any
    ) -> StandardDispatchOutput:
        if self.dp_size <= 1:
            return StandardDispatchOutput(hidden_states=hidden_states, topk_output=topk_output)
        raise RuntimeError(
            "StandardDispatcher no longer supports MoE DP collectives. "
            "Use --afd-moe-a2a-backend=deepep for mlp_dp_size > 1."
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        if self.dp_size <= 1:
            return combine_input.hidden_states
        raise RuntimeError(
            "StandardDispatcher no longer supports MoE DP collectives. "
            "Use --afd-moe-a2a-backend=deepep for mlp_dp_size > 1."
        )


@register_pre_permute("standard", "triton")
def _pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
) -> StandardDispatchOutput:
    return dispatch_output


@register_post_permute("triton", "standard")
def _post_permute_triton_to_standard(
    runner_output: torch.Tensor,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:
    del dispatch_output
    return StandardCombineInput(hidden_states=runner_output)


@register_pre_permute("standard", "triton_fp8")
def _pre_permute_standard_to_triton_fp8(
    dispatch_output: StandardDispatchOutput,
) -> StandardDispatchOutput:
    return dispatch_output


@register_post_permute("triton_fp8", "standard")
def _post_permute_triton_fp8_to_standard(
    runner_output: torch.Tensor,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:
    del dispatch_output
    return StandardCombineInput(hidden_states=runner_output)


@register_pre_permute("standard", "flashinfer_nvfp4")
def _pre_permute_standard_to_flashinfer_nvfp4(
    dispatch_output: StandardDispatchOutput,
) -> StandardDispatchOutput:
    return dispatch_output


@register_post_permute("flashinfer_nvfp4", "standard")
def _post_permute_flashinfer_nvfp4_to_standard(
    runner_output: torch.Tensor,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:
    del dispatch_output
    return StandardCombineInput(hidden_states=runner_output)


__all__ = ["StandardDispatchOutput", "StandardCombineInput", "StandardDispatcher"]
