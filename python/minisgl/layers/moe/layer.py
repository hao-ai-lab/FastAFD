import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info, try_get_ep_info
from minisgl.models.config import QuantConfig
from minisgl.utils import div_ceil, div_even

from ..base import BaseOP
from .moe_runner import create_moe_runner
from .moe_runner.base import MoeRunnerConfig
from .permute_registry import get_post_permute, get_pre_permute
from .token_dispatcher import DeepEPDispatcher, create_moe_dispatcher


def _build_expert_map(
    num_experts: int, ep_rank: int, ep_size: int
) -> tuple[int, torch.Tensor]:
    """Return (local_num_experts, expert_map) for full EP.

    Each rank owns a contiguous expert slice. Non-local experts map to -1 so
    the fused MoE kernel can skip those blocks.
    """

    local_num_experts = div_ceil(num_experts, ep_size)
    start = ep_rank * local_num_experts
    end = min(num_experts, start + local_num_experts)
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cpu")
    if start < end:
        expert_map[start:end] = torch.arange(end - start, dtype=torch.int32, device="cpu")
    return local_num_experts, expert_map


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        quant: QuantConfig | None = None,
        a2a_backend: str | None = None,
        runner_backend: str | None = None,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        ep_info = try_get_ep_info()
        ep_size = ep_info.size if ep_info is not None else 1
        if ep_size > 1 and ep_size % max(1, tp_info.size) != 0:
            raise RuntimeError(
                "This MoELayer implementation currently requires ep_size to be "
                "a multiple of dense tp_size when EP is enabled. AFD only wires "
                "per-MLP-DP EP and full-world EP; partial MoE TP/EP splits are "
                "not supported. Got "
                f"ep_size={ep_size}, tp_size={tp_info.size}"
            )
        self.tp_size = tp_info.size
        self.ep_size = ep_size
        self.moe_dp_size = ep_size // tp_info.size if ep_size > tp_info.size else 1
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.quant = quant

        if ep_size == 1:
            self.local_num_experts = num_experts
            self.local_expert_offset = 0
            self._expert_map_dev: torch.Tensor | None = None
            intermediate_size_per_partition = div_even(intermediate_size, tp_info.size)
        else:
            assert ep_info is not None
            self.local_num_experts, expert_map_cpu = _build_expert_map(
                num_experts, ep_info.rank, ep_size
            )
            self.local_expert_offset = int(ep_info.rank) * self.local_num_experts
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            )
            self._expert_map_dev = (
                expert_map_cpu.to(device=device) if device is not None else expert_map_cpu
            )
            intermediate_size_per_partition = intermediate_size

        self._is_fp8 = quant is not None and quant.method in ("fp8", "fp8_channel")
        self._is_nvfp4 = quant is not None and quant.method == "modelopt_nvfp4"
        self._fp8_scale_format = None
        if self._is_nvfp4:
            if hidden_size % 16 or intermediate_size_per_partition % 16:
                raise RuntimeError(
                    "ModelOpt NVFP4 requires hidden and local intermediate sizes "
                    f"divisible by 16, got {hidden_size} and "
                    f"{intermediate_size_per_partition}"
                )
            self.gate_up_proj = torch.empty(
                self.local_num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            )
            self.gate_up_proj_scale = torch.empty(
                self.local_num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 16,
                dtype=torch.float8_e4m3fn,
            )
            self.gate_up_proj_weight_scale_2 = torch.empty(
                self.local_num_experts, 2, dtype=torch.float32
            )
            self.gate_up_proj_input_scale = torch.empty(2, dtype=torch.float32)
            self.down_proj = torch.empty(
                self.local_num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            )
            self.down_proj_scale = torch.empty(
                self.local_num_experts,
                hidden_size,
                intermediate_size_per_partition // 16,
                dtype=torch.float8_e4m3fn,
            )
            self.down_proj_weight_scale_2 = torch.empty(
                self.local_num_experts, dtype=torch.float32
            )
            self.down_proj_input_scale = torch.empty((), dtype=torch.float32)
        elif self._is_fp8:
            assert quant is not None
            if quant.method == "fp8_channel":
                block_out, block_in = (1, 1)
                self._fp8_scale_format = "channel"
            else:
                block_out, block_in = quant.weight_block_size
                self._fp8_scale_format = "block"
                if hidden_size % block_in != 0 or hidden_size % block_out != 0:
                    raise RuntimeError(
                        f"FP8 block_size {(block_out, block_in)} does not tile hidden "
                        f"dimension: hidden={hidden_size}"
                    )
            self.gate_up_proj = torch.empty(
                self.local_num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            )
            if self._fp8_scale_format == "channel":
                self.gate_up_proj_scale = torch.empty(
                    self.local_num_experts,
                    2 * intermediate_size_per_partition,
                    1,
                    dtype=torch.float32,
                )
            else:
                self.gate_up_proj_scale = torch.empty(
                    self.local_num_experts,
                    2 * div_ceil(intermediate_size_per_partition, block_out),
                    hidden_size // block_in,
                    dtype=torch.float32,
                )
            self.down_proj = torch.empty(
                self.local_num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
            )
            if self._fp8_scale_format == "channel":
                self.down_proj_scale = torch.empty(
                    self.local_num_experts,
                    hidden_size,
                    1,
                    dtype=torch.float32,
                )
            else:
                self.down_proj_scale = torch.empty(
                    self.local_num_experts,
                    hidden_size // block_out,
                    div_ceil(intermediate_size_per_partition, block_in),
                    dtype=torch.float32,
                )
        else:
            self.gate_up_proj = torch.empty(
                self.local_num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
            )
            self.down_proj = torch.empty(
                self.local_num_experts,
                hidden_size,
                intermediate_size_per_partition,
            )

        ctx = get_global_ctx()
        server_args = getattr(ctx, "server_args", None)
        runner_backend = str(
            runner_backend
            if runner_backend is not None
            else getattr(server_args, "afd_moe_runner_backend", "auto")
        )
        a2a_backend = str(
            a2a_backend
            if a2a_backend is not None
            else getattr(server_args, "afd_moe_a2a_backend", "none")
        )
        if self._is_nvfp4:
            if runner_backend == "auto":
                runner_backend = "flashinfer_nvfp4"
            elif runner_backend != "flashinfer_nvfp4":
                raise RuntimeError(
                    "ModelOpt NVFP4 routed experts require "
                    "afd_moe_runner_backend=flashinfer_nvfp4"
                )
        if a2a_backend == "deepep" and runner_backend == "auto":
            runner_backend = "deep_gemm"
        runner_config = MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.local_num_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            top_k=self.top_k,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            renormalize=self.renormalize,
            ep_size=self.ep_size,
            tp_size=self.tp_size,
            tp_rank=tp_info.rank,
            dp_size=self.moe_dp_size,
            dp_rank=int(getattr(ctx, "replica_id", 0)),
            params_dtype=self.gate_up_proj.dtype,
            runner_backend=runner_backend,
        )
        self.dispatcher = create_moe_dispatcher(
            a2a_backend=a2a_backend,
            moe_runner_config=runner_config,
            dp_size=self.moe_dp_size,
        )
        if isinstance(self.dispatcher, DeepEPDispatcher) and runner_backend not in {
            "deep_gemm",
            "flashinfer_nvfp4",
        }:
            raise RuntimeError(
                "DeepEP elastic MoE requires deep_gemm or flashinfer_nvfp4"
            )
        runner_config.runner_backend = runner_backend
        self.runner = create_moe_runner(
            runner_backend=runner_backend,
            moe_runner_config=runner_config,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        empty_deepep = hidden_states.shape[0] == 0 and isinstance(
            self.dispatcher, DeepEPDispatcher
        )
        if hidden_states.shape[0] == 0 and not empty_deepep:
            return hidden_states
        if empty_deepep:
            hidden_states = hidden_states.new_zeros((1, self.hidden_size))
            router_logits = router_logits.new_zeros((1, self.num_experts))
        dispatch_output = self.dispatcher.dispatch(hidden_states, router_logits)
        combine_input = self._run_moe_core(dispatch_output)
        final_hidden_states = self.dispatcher.combine(combine_input)
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        if empty_deepep:
            return final_hidden_states[:0]
        return final_hidden_states

    def _run_moe_core(self, dispatch_output):
        src_format = dispatch_output.format.value
        runner_key = self.runner.runner_backend.value
        pre_permute = get_pre_permute(src_format, runner_key)
        runner_input = pre_permute(dispatch_output)
        runner_output = self.runner.apply(runner_input, self)
        post_permute = get_post_permute(runner_key, src_format)
        return post_permute(runner_output, dispatch_output)
