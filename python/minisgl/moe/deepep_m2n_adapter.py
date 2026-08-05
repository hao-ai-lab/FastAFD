from __future__ import annotations

from typing import Any, Callable, NamedTuple

import torch
import torch.distributed as dist

from minisgl.layers.moe.token_dispatcher.base import DispatchOutputFormat
from minisgl.kernel import DeepEPMoeElasticBuffer

from .deepep_m2n_union import (
    DeepEPUnionM2NLayout,
    build_deepep_union_m2n_layout,
    remap_real_to_deepep_experts,
)


class DeepEPM2NDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    handle: Any
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    recv_count: torch.Tensor
    layout: DeepEPUnionM2NLayout
    rank: int
    is_ag: bool
    is_eg: bool
    hidden_states_scale: torch.Tensor | None = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_ELASTIC


class DeepEPM2NAdapter:
    """Role-aware M2N adapter over a DeepEP V2 union group.

    DeepEP still sees a symmetric group.  This adapter hides that contract by
    assigning dummy expert slots to AG/source ranks and remapping real expert
    ids into EG/expert rank ranges before dispatch.
    """

    def __init__(
        self,
        *,
        group: Any,
        ag_size: int,
        eg_size: int,
        real_num_experts: int,
        hidden_size: int,
        top_k: int,
        num_max_dispatch_tokens_per_rank: int,
        use_expanded_layout: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("DeepEPM2NAdapter requires torch.distributed initialization")
        self.group = group
        self.layout = build_deepep_union_m2n_layout(
            ag_size=int(ag_size),
            eg_size=int(eg_size),
            real_num_experts=int(real_num_experts),
        )
        self.rank = int(dist.get_rank(group=group))
        self.world_size = int(dist.get_world_size(group=group))
        if self.world_size != self.layout.union_size:
            raise RuntimeError(
                "DeepEPM2NAdapter group size mismatch: "
                f"group={self.world_size} layout={self.layout.union_size}"
            )
        if self.rank < 0 or self.rank >= self.layout.union_size:
            raise RuntimeError(
                f"DeepEPM2NAdapter rank={self.rank} outside union_size={self.layout.union_size}"
            )
        self.is_ag = self.rank < self.layout.ag_size
        self.is_eg = not self.is_ag
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.use_expanded_layout = bool(use_expanded_layout)
        self.num_max_dispatch_tokens_per_rank = int(num_max_dispatch_tokens_per_rank)
        self._deepep_expert_map_by_device: dict[tuple[str, int | None, str], torch.Tensor] = {}
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.num_max_dispatch_tokens_per_rank < 0:
            raise ValueError(
                "num_max_dispatch_tokens_per_rank must be nonnegative, "
                f"got {self.num_max_dispatch_tokens_per_rank}"
            )
        self._log_callback = log
        self._logged_dispatch_allocation_shapes: set[tuple[int, str, tuple[int, ...]]] = set()
        self._emit_log(
            self._format_topology_log(
                layout=self.layout,
                rank=self.rank,
                world_size=self.world_size,
                role=self.role,
            )
        )
        self.buffer = DeepEPMoeElasticBuffer(
            group=group,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            hidden_size=self.hidden_size,
            num_experts=self.layout.deepep_num_experts,
            top_k=self.top_k,
            log=log,
        )
        self._emit_log(
            self._format_deepep_path_log(
                self.buffer,
                rank=self.rank,
                role=self.role,
            )
        )
        self._destroyed = False

    @property
    def role(self) -> str:
        return "ag" if self.is_ag else "eg"

    @property
    def topk_idx_dtype(self) -> torch.dtype:
        return self.buffer.topk_idx_dtype

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        hidden_states_scale: torch.Tensor | None = None,
        expert_alignment: int | None = None,
        num_max_dispatch_tokens_per_rank: int | None = None,
        do_cpu_sync: bool = False,
        router_logits: torch.Tensor | None = None,
        renormalize: bool = False,
        topk_ids_are_deepep: bool = False,
    ) -> DeepEPM2NDispatchOutput:
        self._check_not_destroyed()
        effective_max_tokens = (
            self.num_max_dispatch_tokens_per_rank
            if num_max_dispatch_tokens_per_rank is None
            else int(num_max_dispatch_tokens_per_rank)
        )
        self._validate_hidden_states(hidden_states)
        local_tokens = int(hidden_states.shape[0])
        if effective_max_tokens < local_tokens:
            raise RuntimeError(
                "DeepEP M2N dispatch bucket is smaller than local token count: "
                f"bucket={effective_max_tokens} tokens={local_tokens}"
            )
        if self.is_eg and local_tokens != 0:
            raise RuntimeError(
                "DeepEP M2N EG ranks must not source tokens in Phase 3A: "
                f"rank={self.rank} tokens={local_tokens}"
            )
        if router_logits is not None:
            deepep_topk_ids, real_topk_weights = self._route_to_deepep_topk(
                router_logits,
                hidden_states,
                renormalize=bool(renormalize),
            )
        elif topk_ids_are_deepep:
            deepep_topk_ids = self._normalize_topk_ids(topk_ids, hidden_states).to(
                device=hidden_states.device,
                dtype=self.topk_idx_dtype,
            )
            real_topk_weights = self._normalize_topk_weights(topk_weights, hidden_states)
        else:
            real_topk_ids = self._normalize_topk_ids(topk_ids, hidden_states)
            real_topk_weights = self._normalize_topk_weights(topk_weights, hidden_states)
            deepep_topk_ids = remap_real_to_deepep_experts(
                real_topk_ids,
                self.layout,
            ).to(device=hidden_states.device, dtype=self.topk_idx_dtype)
        (
            recv_x,
            recv_x_scale,
            recv_topk_ids,
            recv_topk_weights,
            handle,
        ) = self.buffer.dispatch(
            hidden_states.contiguous(),
            deepep_topk_ids.contiguous(),
            real_topk_weights.float().contiguous(),
            hidden_states_scale=hidden_states_scale,
            expert_alignment=(
                self._resolve_expert_alignment(expert_alignment)
                if self.use_expanded_layout
                else 1
            ),
            num_max_dispatch_tokens_per_rank=effective_max_tokens,
            do_cpu_sync=bool(do_cpu_sync),
            use_expanded_layout=self.use_expanded_layout,
        )
        if recv_topk_weights is not None and int(recv_topk_weights.shape[0]) > int(recv_x.shape[0]):
            recv_topk_weights = recv_topk_weights[: recv_x.shape[0]]
        dispatch_output = DeepEPM2NDispatchOutput(
            hidden_states=recv_x,
            handle=handle,
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            recv_count=handle.handle.psum_num_recv_tokens_per_expert,
            layout=self.layout,
            rank=self.rank,
            is_ag=self.is_ag,
            is_eg=self.is_eg,
            hidden_states_scale=recv_x_scale,
        )
        self._log_dispatch_allocation(dispatch_output)
        return dispatch_output

    def combine(
        self,
        expert_output: torch.Tensor,
        dispatch_output: DeepEPM2NDispatchOutput | Any,
        *,
        topk_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_not_destroyed()
        if isinstance(dispatch_output, DeepEPM2NDispatchOutput):
            if topk_weights is not None:
                raise RuntimeError(
                    "DeepEP M2N combine must not receive topk_weights; "
                    "the expert runner applies routed top-k weights before combine"
                )
            handle = dispatch_output.handle
        else:
            handle = dispatch_output
        return self.buffer.combine(expert_output, topk_weights, handle)

    @staticmethod
    def num_recv_tokens(dispatch_output: DeepEPM2NDispatchOutput) -> int:
        runtime_handle = dispatch_output.handle.handle
        return int(runtime_handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())

    @staticmethod
    def valid_expanded_slots(dispatch_output: DeepEPM2NDispatchOutput) -> torch.Tensor:
        runtime_handle = dispatch_output.handle.handle
        num_recv_tokens = DeepEPM2NAdapter.num_recv_tokens(dispatch_output)
        metadata = runtime_handle.recv_src_metadata[:num_recv_tokens]
        if metadata.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=metadata.device)
        slots = metadata[:, 2:].reshape(-1)
        slots = slots[slots >= 0]
        return slots.to(dtype=torch.long)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.buffer.destroy()
        self._destroyed = True

    close = destroy

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"hidden_states must be 2D, got shape={tuple(hidden_states.shape)}"
            )
        if int(hidden_states.shape[1]) != self.hidden_size:
            raise ValueError(
                "hidden_states hidden size mismatch: "
                f"got={int(hidden_states.shape[1])} expected={self.hidden_size}"
            )
        if not hidden_states.is_cuda:
            raise ValueError("DeepEPM2NAdapter requires CUDA hidden_states")

    def _normalize_topk_ids(
        self,
        topk_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        local_tokens = int(hidden_states.shape[0])
        if topk_ids is None:
            if local_tokens != 0:
                raise ValueError("topk_ids is required for non-empty source ranks")
            topk_ids = torch.empty(
                (0, self.top_k),
                dtype=torch.int64,
                device=hidden_states.device,
            )
        if topk_ids.ndim != 2:
            raise ValueError(f"topk_ids must be 2D, got shape={tuple(topk_ids.shape)}")
        if tuple(topk_ids.shape) != (local_tokens, self.top_k):
            raise ValueError(
                "topk_ids shape mismatch: "
                f"got={tuple(topk_ids.shape)} expected={(local_tokens, self.top_k)}"
            )
        return topk_ids.to(device=hidden_states.device)

    def _normalize_topk_weights(
        self,
        topk_weights: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        local_tokens = int(hidden_states.shape[0])
        if topk_weights is None:
            if local_tokens != 0:
                raise ValueError("topk_weights is required for non-empty source ranks")
            topk_weights = torch.empty(
                (0, self.top_k),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        if topk_weights.ndim != 2:
            raise ValueError(
                f"topk_weights must be 2D, got shape={tuple(topk_weights.shape)}"
            )
        if tuple(topk_weights.shape) != (local_tokens, self.top_k):
            raise ValueError(
                "topk_weights shape mismatch: "
                f"got={tuple(topk_weights.shape)} expected={(local_tokens, self.top_k)}"
        )
        return topk_weights.to(device=hidden_states.device, dtype=torch.float32)

    def route_to_deepep_topk(
        self,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._route_to_deepep_topk(
            router_logits,
            hidden_states,
            renormalize=bool(renormalize),
        )

    def _route_to_deepep_topk(
        self,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = int(hidden_states.shape[0])
        if router_logits.ndim != 2:
            raise ValueError(f"router_logits must be 2D, got shape={tuple(router_logits.shape)}")
        expected = (local_tokens, self.layout.real_num_experts)
        if tuple(router_logits.shape) != expected:
            raise ValueError(
                "router_logits shape mismatch: "
                f"got={tuple(router_logits.shape)} expected={expected}"
            )
        if not router_logits.is_cuda:
            raise ValueError("DeepEP M2N fused topk requires CUDA router_logits")
        if router_logits.device != hidden_states.device:
            raise ValueError(
                "router_logits and hidden_states must be on the same device: "
                f"{router_logits.device} vs {hidden_states.device}"
            )
        if not router_logits.is_contiguous():
            router_logits = router_logits.contiguous()

        topk_weights = torch.empty(
            (local_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        deepep_topk_ids = torch.empty(
            (local_tokens, self.top_k),
            dtype=self.topk_idx_dtype,
            device=hidden_states.device,
        )
        from minisgl.kernel import topk_softmax_group_local

        topk_softmax_group_local(
            topk_weights,
            deepep_topk_ids,
            router_logits,
            self._deepep_expert_map(hidden_states.device),
            renormalize=bool(renormalize),
        )
        return deepep_topk_ids, topk_weights

    def _deepep_expert_map(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index, str(self.topk_idx_dtype))
        cached = self._deepep_expert_map_by_device.get(key)
        if cached is not None:
            return cached
        expert_map = torch.arange(
            self.layout.real_num_experts,
            dtype=self.topk_idx_dtype,
            device=device,
        )
        offset = int(self.layout.dummy_expert_offset)
        if offset:
            expert_map = expert_map + offset
        expert_map = expert_map.contiguous()
        self._deepep_expert_map_by_device[key] = expert_map
        return expert_map

    def _check_not_destroyed(self) -> None:
        if self._destroyed:
            raise RuntimeError("DeepEPM2NAdapter has been destroyed")

    def _emit_log(self, message: str) -> None:
        if self._log_callback is not None:
            self._log_callback(f"deepep_m2n:{message}")

    def _log_dispatch_allocation(self, dispatch_output: DeepEPM2NDispatchOutput) -> None:
        if self._log_callback is None:
            return
        if self._is_cuda_graph_capture_active(dispatch_output.hidden_states):
            return
        recv_x_shape = tuple(int(dim) for dim in dispatch_output.hidden_states.shape)
        signature = (
            int(dispatch_output.rank),
            "ag" if dispatch_output.is_ag else "eg",
            recv_x_shape,
        )
        logged_shapes = getattr(self, "_logged_dispatch_allocation_shapes", None)
        if logged_shapes is None:
            logged_shapes = set()
            self._logged_dispatch_allocation_shapes = logged_shapes
        if signature in logged_shapes:
            return
        logged_shapes.add(signature)
        real_recv_tokens = self.num_recv_tokens(dispatch_output)
        real_recv_expanded_slots = int(self.valid_expanded_slots(dispatch_output).numel())
        self._emit_log(
            self._format_dispatch_allocation_log(
                dispatch_output,
                real_recv_tokens=real_recv_tokens,
                real_recv_expanded_slots=real_recv_expanded_slots,
            )
        )

    @staticmethod
    def _is_cuda_graph_capture_active(tensor: torch.Tensor) -> bool:
        if not tensor.is_cuda or not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            return False

    @staticmethod
    def _format_topology_log(
        *,
        layout: DeepEPUnionM2NLayout,
        rank: int,
        world_size: int,
        role: str,
    ) -> str:
        ag_ranks = ",".join(str(i) for i in range(layout.ag_size))
        eg_ranks = ",".join(str(layout.union_rank_for_eg_rank(i)) for i in range(layout.eg_size))
        dummy_ranges = ",".join(
            f"ag{i}:[{rng.start},{rng.stop})"
            for i in range(layout.ag_size)
            for rng in (layout.dummy_expert_range_for_ag_rank(i),)
        )
        real_ranges = ",".join(
            f"eg{i}:[{rng.start},{rng.stop})"
            for i in range(layout.eg_size)
            for rng in (layout.real_expert_range_for_eg_rank(i),)
        )
        return (
            "topology "
            f"rank={int(rank)} role={role} world_size={int(world_size)} "
            f"ag_size={layout.ag_size} eg_size={layout.eg_size} "
            f"union_size={layout.union_size} ag_ranks=[{ag_ranks}] eg_ranks=[{eg_ranks}] "
            f"real_experts={layout.real_num_experts} "
            f"experts_per_union_rank={layout.experts_per_union_rank} "
            f"dummy_offset={layout.dummy_expert_offset} "
            f"virtual_experts={layout.deepep_num_experts} "
            f"dummy_ranges={dummy_ranges} real_ranges={real_ranges}"
        )

    @staticmethod
    def _format_deepep_path_log(
        buffer: Any,
        *,
        rank: int | None = None,
        role: str | None = None,
    ) -> str:
        def field(name: str) -> Any:
            return getattr(buffer, name, "unknown")

        def bool_as_int(value: Any) -> Any:
            if value == "unknown":
                return value
            return int(bool(value))

        scaleout = field("num_scaleout_ranks")
        scaleup = field("num_scaleup_ranks")
        domain = field("domain")
        if domain == "unknown":
            domain = f"scaleout{scaleout}_scaleup{scaleup}"
        num_qps = field("num_qps_per_rank")
        if num_qps == "unknown":
            num_qps = field("num_allocated_qps")
        return (
            "new_afd_deepep "
            f"rank={field('rank') if rank is None else int(rank)} "
            f"role={field('role') if role is None else role} "
            f"allow_hybrid={bool_as_int(field('allow_hybrid_mode'))} "
            f"num_scaleup_ranks={scaleup} "
            f"num_scaleout_ranks={scaleout} "
            f"num_qps_per_rank={num_qps} "
            f"num_sms={field('num_sms')} "
            f"domain={domain} "
            f"allow_multi_reduce={bool_as_int(field('allow_multiple_reduction'))} "
            f"prefer_overlap={bool_as_int(field('prefer_overlap_with_compute'))} "
            f"num_rdma_ranks={field('num_rdma_ranks')} "
            f"num_nvlink_ranks={field('num_nvlink_ranks')}"
        )

    @staticmethod
    def _format_dispatch_allocation_log(
        dispatch_output: DeepEPM2NDispatchOutput,
        *,
        real_recv_tokens: int,
        real_recv_expanded_slots: int,
    ) -> str:
        recv_x = dispatch_output.hidden_states
        recv_x_shape = tuple(int(dim) for dim in recv_x.shape)
        recv_x_bytes = int(recv_x.numel() * recv_x.element_size())
        return (
            "new_afd_m2n_alloc "
            f"rank={dispatch_output.rank} role={'ag' if dispatch_output.is_ag else 'eg'} "
            f"step_id=unknown layer=unknown mb=unknown "
            f"recv_x_shape={recv_x_shape} recv_x_nbytes={recv_x_bytes} "
            f"real_recv_tokens={int(real_recv_tokens)} "
            f"real_recv_expanded_slots={int(real_recv_expanded_slots)} "
            f"virtual_experts={dispatch_output.layout.deepep_num_experts} "
            f"experts_per_rank={dispatch_output.layout.experts_per_union_rank}"
        )

    @staticmethod
    def _resolve_expert_alignment(expert_alignment: int | None) -> int:
        from minisgl.kernel import deepgemm as deep_gemm

        if expert_alignment is not None:
            alignment = int(expert_alignment)
        else:
            alignment = int(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
        if alignment <= 0:
            raise ValueError(f"expert_alignment must be positive, got {alignment}")
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        return alignment


__all__ = [
    "DeepEPM2NAdapter",
    "DeepEPM2NDispatchOutput",
]
