"""AFD expert worker actor."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import time
from typing import Any

import torch
import torch.distributed
from .afd_protocol import (
    AfdCommand,
    AfdFlushStepCmd,
    AfdProfilerCmd,
    AfdStopCmd,
    AfdEGStepPlan,
    AfdEGStepReply,
    AfdRunEGStepCmd,
)
from minisgl.core import Context, get_global_ctx, set_global_ctx
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _init_tp_communication
from minisgl.utils import nvtx_range

from .cuda_graph_utils import capture_cuda_graph
from .afd_support import AfdRuntimeConfig, log_line
from .afd_worker_base import BaseAfdWorker, ensure_afd_parallel_info

__all__ = ["AfdExpertWorker", "AfdEgDecodeGraphRunner"]


class AfdExpertState:
    """Expert-side CUDA state: device, stream, TP comms, and global context."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.dtype = config.dtype
        device_env = os.environ.get("MINISGL_DEVICE_INDEX")
        device_index = (
            int(device_env) if device_env is not None else int(torch.cuda.current_device())
        )
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        ensure_afd_parallel_info(config)
        self.tp_cpu_group = (
            torch.distributed.group.WORLD
            if torch.distributed.is_initialized()
            else _init_tp_communication(config, self.dtype)
        )
        try:
            self.ctx = get_global_ctx()
        except AssertionError:
            self.ctx = Context(config.page_size)
            set_global_ctx(self.ctx)
        self.stream = torch.cuda.Stream(device=self.device)


@dataclass
class _AfdEgInFlight:
    """EG step handle: experts launched on the compute stream; `done` is
    synchronized when the fixed two-step pipeline retires it."""

    step_id: int
    sent_ns: int
    done: torch.cuda.Event


@dataclass(frozen=True)
class _AfdEgEmptyPayload:
    hidden: torch.Tensor
    scale: torch.Tensor | None


def _eg_uses_megamoe(adapter) -> bool:
    return getattr(adapter, "backend", "deepep") == "megamoe_m2n"


def _make_eg_empty_payload(
    *,
    hidden_size: int,
    device: torch.device,
    expert_stage,
) -> _AfdEgEmptyPayload:
    if bool(getattr(expert_stage, "dispatch_uses_fp8", lambda: True)()):
        return _AfdEgEmptyPayload(
            hidden=torch.empty(
                (0, hidden_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            scale=torch.empty(
                (0, hidden_size // 128 // 4),
                dtype=torch.int32,
                device=device,
            ),
        )
    return _AfdEgEmptyPayload(
        hidden=torch.empty(
            (0, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        scale=None,
    )


def _run_eg_deepep_pipeline_body(
    *,
    num_layers: int,
    num_mb: int,
    bucket: int,
    payload: _AfdEgEmptyPayload,
    expert_stage,
    adapters,
    engine_stream: torch.cuda.Stream,
    lane_streams: tuple[torch.cuda.Stream, ...],
    dispatch_done: list[list[torch.cuda.Event]],
    expert_done: list[list[torch.cuda.Event]],
    phase: str,
    retire_async_objects: bool,
) -> None:
    """DeepEP EG pipeline: empty dispatch -> local experts -> combine."""
    num_layers = int(num_layers)
    num_mb = max(1, int(num_mb))
    bucket = int(bucket)
    dispatches: list[list[Any | None]] = [
        [None for _ in range(num_mb)] for _ in range(num_layers)
    ]
    expert_outputs: list[list[Any | None]] = [
        [None for _ in range(num_mb)] for _ in range(num_layers)
    ]
    retired: list[tuple[torch.cuda.Event, Any, Any]] = []

    def retire(force: bool = False) -> None:
        if not retired:
            return
        keep: list[tuple[torch.cuda.Event, Any, Any]] = []
        for event, disp_obj, out_obj in retired:
            if force:
                event.synchronize()
            elif not event.query():
                keep.append((event, disp_obj, out_obj))
        retired[:] = keep

    def launch_dispatch(layer: int, mb: int) -> None:
        adapter_mb = adapters[mb % len(adapters)]
        lane_stream = lane_streams[mb % len(lane_streams)]
        with torch.cuda.stream(lane_stream):
            disp = adapter_mb.dispatch(
                payload.hidden,
                None,
                None,
                hidden_states_scale=payload.scale,
                num_max_dispatch_tokens_per_rank=bucket,
                do_cpu_sync=BaseAfdWorker._deepep_do_cpu_sync(phase),
            )
            dispatch_done[layer][mb].record(lane_stream)
        dispatches[layer][mb] = disp

    def launch_expert(layer: int, mb: int) -> None:
        disp = dispatches[layer][mb]
        if disp is None:
            raise RuntimeError(f"Missing EG dispatch for layer={layer} mb={mb}")
        with torch.cuda.stream(engine_stream):
            engine_stream.wait_event(dispatch_done[layer][mb])
            expert_out = expert_stage.run_experts(layer, disp)
            expert_done[layer][mb].record(engine_stream)
        expert_outputs[layer][mb] = expert_out

    def launch_combine(layer: int, mb: int) -> None:
        disp = dispatches[layer][mb]
        expert_out = expert_outputs[layer][mb]
        if disp is None or expert_out is None:
            raise RuntimeError(f"Missing EG combine input for layer={layer} mb={mb}")
        adapter_mb = adapters[mb % len(adapters)]
        lane_stream = lane_streams[mb % len(lane_streams)]
        with torch.cuda.stream(lane_stream):
            lane_stream.wait_event(expert_done[layer][mb])
            if isinstance(expert_out, torch.Tensor):
                expert_out.record_stream(lane_stream)
            adapter_mb.combine(expert_out, disp)
            done = torch.cuda.Event()
            done.record(lane_stream)
        if retire_async_objects:
            retired.append((done, disp, expert_out))
            if len(retired) >= 4:
                retire(force=True)
        # During graph capture, release each layer's Python objects as soon as
        # combine has been enqueued. Dispatch storage is created and retired on
        # the lane stream; expert_out.record_stream above covers its
        # engine-to-lane handoff. Retaining every layer until capture ends
        # prevents the graph-private allocator from reusing those buffers.
        dispatches[layer][mb] = None
        expert_outputs[layer][mb] = None

    # EG collective order:
    #   D(L0, mb0), D(L0, mb1), ...
    #   expert/combine each MB for layer L, then immediately post D(L+1, mb).
    # This mirrors the AG pipeline order without a separate schedule object.
    for mb in range(num_mb):
        launch_dispatch(0, mb)
    for layer in range(num_layers):
        for mb in range(num_mb):
            launch_expert(layer, mb)
            launch_combine(layer, mb)
            if layer + 1 < num_layers:
                launch_dispatch(layer + 1, mb)

    if retire_async_objects:
        retire(force=True)


def _run_eg_megamoe_pipeline_body(
    *,
    num_layers: int,
    num_mb: int,
    bucket: int,
    adapters,
    lane_streams: tuple[torch.cuda.Stream, ...],
) -> None:
    """Per-layer MegaMoE EG pipeline used when the persistent kernel is disabled."""
    num_layers = int(num_layers)
    num_mb = max(1, int(num_mb))
    bucket = int(bucket)
    adapter0 = adapters[0]

    for layer in range(num_layers):
        if adapter0.use_dual_lane(num_mb):
            with torch.cuda.stream(lane_streams[0]):
                adapter0.eg_moe_dual(
                    layer,
                    expected_tokens_per_rank=bucket,
                )
            continue
        for mb in range(num_mb):
            adapter_mb = adapters[mb % len(adapters)]
            lane_stream = lane_streams[mb % len(lane_streams)]
            with torch.cuda.stream(lane_stream):
                adapter_mb.eg_moe(
                    layer,
                    lane=mb,
                    expected_tokens_per_rank=bucket,
                )


class AfdEgDecodeGraphRunner:
    """Per-bucket decode graph for the EG forward path.

    Capture always uses the shared per-layer EG pipeline body. MegaMoE persistent
    is intentionally not captured here; it is selected as a separate top-level
    runtime path before replay/eager fallback.
    """

    def __init__(
        self,
        *,
        runtime_state: Any,
        expert_stage: Any,
        adapters: Any,
        num_mb: int = 1,
        comm_stream: Any = None,
        lane_comm_streams: tuple[Any, ...] = (),
        num_layers: int,
        hidden_size: int,
        bs_list: tuple[int, ...],
        log_path: str,
        label: str,
    ) -> None:
        self.state = runtime_state
        self.expert_stage = expert_stage
        self.adapters = list(adapters)
        self.num_mb = max(1, int(num_mb))
        if len(self.adapters) < self.num_mb:
            raise RuntimeError(
                f"EG decode graph requires >= num_mb adapters, "
                f"got adapters={len(self.adapters)} num_mb={self.num_mb}"
            )
        self.comm_stream = comm_stream
        self.lane_comm_streams = (
            tuple(lane_comm_streams or ()) if self.num_mb > 1 else ()
        )
        self.device = runtime_state.device
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.bs_list = tuple(sorted(int(b) for b in bs_list if int(b) > 0))
        self.log_path = log_path
        self.label = label
        self._pool = None
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if _eg_uses_megamoe(self.adapters[0]):
            self._dispatch_done = []
            self._expert_done = []
        else:
            self._dispatch_done = [
                [torch.cuda.Event() for _ in range(self.num_mb)]
                for _ in range(self.num_layers)
            ]
            self._expert_done = [
                [torch.cuda.Event() for _ in range(self.num_mb)]
                for _ in range(self.num_layers)
            ]

    def has_bucket(self, bs: int) -> bool:
        return int(bs) in self.graphs

    def warmup(self, bs: int) -> None:
        # `bs` is PER-MB. The graph is keyed by total dispatch bucket so replay
        # can look up with the runtime AfdEGStepPlan.dispatch_bucket.
        bs = int(bs)
        num_mb = int(self.num_mb)
        mb_bs = bs
        total = bs * num_mb
        if total in self.graphs:
            return

        expert_stage = self.expert_stage
        adapters = self.adapters
        comm_stream = self.comm_stream if self.comm_stream is not None else self.state.stream
        lane_streams = (
            self.lane_comm_streams
            if self.lane_comm_streams
            else tuple(comm_stream for _ in range(num_mb))
        )
        engine_stream = self.state.stream
        is_megamoe_backend = _eg_uses_megamoe(adapters[0])
        payload = (
            None
            if is_megamoe_backend
            else _make_eg_empty_payload(
                hidden_size=self.hidden_size,
                device=self.device,
                expert_stage=expert_stage,
            )
        )

        def _fn() -> None:
            if is_megamoe_backend:
                adapter = adapters[0]
                if num_mb > 2 and adapter.use_persistent(num_mb):
                    with torch.cuda.stream(lane_streams[0]):
                        adapter.eg_moe_persistent(
                            self.num_layers,
                            num_mb=num_mb,
                            expected_tokens_per_rank=mb_bs,
                        )
                else:
                    _run_eg_megamoe_pipeline_body(
                        num_layers=self.num_layers,
                        num_mb=num_mb,
                        bucket=mb_bs,
                        adapters=adapters,
                        lane_streams=lane_streams,
                    )
            else:
                if payload is None:
                    raise RuntimeError("DeepEP EG graph capture requires payload")
                _run_eg_deepep_pipeline_body(
                    num_layers=self.num_layers,
                    num_mb=num_mb,
                    bucket=mb_bs,
                    payload=payload,
                    expert_stage=expert_stage,
                    adapters=adapters,
                    engine_stream=engine_stream,
                    lane_streams=lane_streams,
                    dispatch_done=self._dispatch_done,
                    expert_done=self._expert_done,
                    phase="decode",
                    retire_async_objects=False,
                )

        with torch.cuda.stream(engine_stream):
            _fn()
        engine_stream.synchronize()
        for stream in lane_streams:
            stream.synchronize()
        graph, self._pool = capture_cuda_graph(
            device=self.device,
            engine_stream=engine_stream,
            comm_stream=comm_stream,
            extra_streams=lane_streams,
            overlap_comm=self.comm_stream is not None,
            pool=self._pool,
            fn=_fn,
        )
        self.graphs[total] = graph
        allocated_gib = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved_gib = torch.cuda.memory_reserved(self.device) / (1024**3)
        log_line(
            self.log_path,
            f"[{self.label}] afd_eg_decode_graph warmup:done total={total} "
            f"mb_bs={mb_bs} schedule=paper_pipeline lanes={len(lane_streams)} "
            f"allocated_gib={allocated_gib:.2f} reserved_gib={reserved_gib:.2f}",
            flush=True,
        )

    def replay(self, bs: int) -> None:
        with torch.cuda.stream(self.state.stream):
            self.graphs[int(bs)].replay()


class AfdExpertWorker(BaseAfdWorker):
    """Per-GPU AFD expert/EG worker actor."""

    def __init__(self, **kwargs):
        kwargs.pop("role", None)
        super().__init__(role="mlp", **kwargs)

    # ------------------------------------------------------------------
    # Hot loops
    # ------------------------------------------------------------------

    def _run_hot_rpc_loop(self) -> None:
        self._prepare_hot_rpc_runtime()
        if self.disable_overlap:
            self.afd_normal_loop()
        else:
            self.afd_overlap_loop()

    def _run_afd_eg_step(
        self,
        plan: AfdEGStepPlan,
        *,
        sent_ns: int = 0,
    ) -> _AfdEgInFlight:
        """Run one EG step and return the delayed ack handle."""
        runtime_state = self.runtime_state
        bucket = int(plan.dispatch_bucket)
        profile = self._control_profile_enabled(int(plan.step_id))
        total_start_ns = time.perf_counter_ns()
        mode = "eager"

        with torch.cuda.stream(runtime_state.stream):
            launch_start_ns = time.perf_counter_ns()
            if self._run_eg_persistent(plan, bucket):
                mode = "persistent"
            else:
                eg_graph = self._afd_eg_graph
                if (
                    self.enable_decode_graph
                    and plan.use_decode_graph
                    and eg_graph is not None
                    and plan.phase == "decode"
                    and eg_graph.has_bucket(bucket)
                ):
                    log_line(
                        self.log_path,
                        f"[{self.role} dp={self.mlp_dp_rank} rank={self.tp_rank}] "
                        f"afd_eg_decode_graph:replay step_id={int(plan.step_id)} "
                        f"bucket={int(bucket)} num_mb={int(plan.num_mb)}",
                        flush=True,
                    )
                    eg_graph.replay(bucket)
                    mode = "graph"
                else:
                    self._run_eg_pipeline_eager(plan, bucket)
                    mode = "eager"
            launch_done_ns = time.perf_counter_ns()
            event_start_ns = launch_done_ns
            done = torch.cuda.Event()
            done.record(runtime_state.stream)
            event_done_ns = time.perf_counter_ns()
        if profile:
            log_line(
                self.log_path,
                f"[{self.role} dp={self.mlp_dp_rank} rank={self.tp_rank}] control_eg_step "
                f"step_id={int(plan.step_id)} phase={plan.phase} "
                f"bucket={int(bucket)} num_mb={int(plan.num_mb)} mode={mode} "
                f"launch_enqueue_ms={(launch_done_ns - launch_start_ns) / 1_000_000.0:.3f} "
                f"event_record_ms={(event_done_ns - event_start_ns) / 1_000_000.0:.3f} "
                f"total_enqueue_ms={(event_done_ns - total_start_ns) / 1_000_000.0:.3f}",
            )
        return _AfdEgInFlight(
            step_id=int(plan.step_id),
            sent_ns=int(sent_ns),
            done=done,
        )

    def _run_eg_persistent(self, plan: AfdEGStepPlan, bucket: int) -> bool:
        """Launch the single-kernel EG MegaMoE path when enabled.

        This path is independent from both graph replay and pipeline eager mode.
        The AG side may still replay its graph; the EG persistent kernel services
        the same per-layer collectives from one long-running launch.
        """
        adapter = self._afd_adapters[0]
        if not (_eg_uses_megamoe(adapter) and adapter.use_persistent(int(plan.num_mb))):
            return False
        lane_streams = tuple(self._afd_comm_streams) or (self.runtime_state.stream,)
        with torch.cuda.stream(lane_streams[0]):
            adapter.eg_moe_persistent(
                int(self.num_layers),
                num_mb=int(plan.num_mb),
                expected_tokens_per_rank=int(bucket),
            )
        self.runtime_state.stream.wait_stream(lane_streams[0])
        return True

    def _run_eg_pipeline_eager(self, plan: AfdEGStepPlan, bucket: int) -> None:
        """Eager EG fallback used for prefill, graph-off debug, or bucket miss.

        The same pipeline scheduler handles num_mb=1 and num_mb>1. It contains
        both the per-layer MegaMoE and legacy DeepEP implementations, so the eager
        path has one dispatch/expert/combine ordering contract.
        """
        adapter = self._afd_adapter
        adapters = self._afd_adapters
        expert_stage = self._afd_eg_model
        is_megamoe_backend = _eg_uses_megamoe(adapter)
        lane_streams = tuple(self._afd_comm_streams) or (self.runtime_state.stream,)
        if len(adapters) < max(1, int(plan.num_mb)):
            raise RuntimeError(
                f"EG afd requires >= num_mb adapters, "
                f"got adapters={len(adapters)} num_mb={int(plan.num_mb)}"
        )
        num_mb = max(1, int(plan.num_mb))
        if is_megamoe_backend:
            _run_eg_megamoe_pipeline_body(
                num_layers=int(self.num_layers),
                num_mb=num_mb,
                bucket=bucket,
                adapters=adapters,
                lane_streams=lane_streams,
            )
            for stream in lane_streams:
                self.runtime_state.stream.wait_stream(stream)
            return

        payload = _make_eg_empty_payload(
            hidden_size=int(self.runtime.config.model_config.hidden_size),
            device=self.runtime_state.device,
            expert_stage=expert_stage,
        )
        dispatch_done = [
            [torch.cuda.Event() for _ in range(num_mb)]
            for _ in range(int(self.num_layers))
        ]
        expert_done = [
            [torch.cuda.Event() for _ in range(num_mb)]
            for _ in range(int(self.num_layers))
        ]
        _run_eg_deepep_pipeline_body(
            num_layers=int(self.num_layers),
            num_mb=num_mb,
            bucket=bucket,
            payload=payload,
            expert_stage=expert_stage,
            adapters=adapters,
            engine_stream=self.runtime_state.stream,
            lane_streams=lane_streams,
            dispatch_done=dispatch_done,
            expert_done=expert_done,
            phase=str(plan.phase),
            retire_async_objects=True,
        )
        for stream in lane_streams:
            self.runtime_state.stream.wait_stream(stream)

    def _process_last_afd_eg(self, last: "_AfdEgInFlight | None") -> None:
        """Finalize the PREVIOUS EG step: wait its done event, then ack."""
        if last is None:
            return
        profile = self._control_profile_enabled(int(last.step_id))
        start_ns = time.perf_counter_ns()
        last.done.synchronize()
        sync_done_ns = time.perf_counter_ns()
        self._send_reply(
            AfdEGStepReply(
                worker_rank=self.worker_rank,
                step_id=last.step_id,
                sent_ns=int(last.sent_ns),
            )
        )
        send_done_ns = time.perf_counter_ns()
        if profile:
            log_line(
                self.log_path,
                f"[{self.role} dp={self.mlp_dp_rank} rank={self.tp_rank}] control_eg_retire "
                f"step_id={int(last.step_id)} "
                f"done_sync_ms={(sync_done_ns - start_ns) / 1_000_000.0:.3f} "
                f"reply_send_ms={(send_done_ns - sync_done_ns) / 1_000_000.0:.3f} "
                f"total_ms={(send_done_ns - start_ns) / 1_000_000.0:.3f}",
            )

    def afd_normal_loop(self) -> None:
        """Synchronous EG loop: run one step, wait for it, then ack."""
        while True:
            with nvtx_range("AFD_ExpertHotLoop_AfdNormalIter"):
                cmd = self._recv_hot_command(blocking=True)
                if cmd is None:
                    continue
                if isinstance(cmd, AfdFlushStepCmd):
                    continue
                if isinstance(cmd, AfdStopCmd):
                    return
                if isinstance(cmd, AfdProfilerCmd):
                    self.handle_profiler_cmd(cmd)
                    continue
                if not isinstance(cmd, AfdRunEGStepCmd):
                    raise RuntimeError(f"Unsupported EG command: {type(cmd).__name__}")
                self._process_last_afd_eg(
                    self._run_afd_eg_step(cmd.plan, sent_ns=cmd.sent_ns)
                )

    def afd_overlap_loop(self) -> None:
        """Finite-window step-ahead EG loop.

        Match the AG worker's configured pending window before retiring the
        oldest ack handle. AfdFlushStepCmd drains the pipeline.
        """
        pending: deque[_AfdEgInFlight] = deque()

        def retire(force: bool = False) -> None:
            while pending and (force or len(pending) > self.async_retire_depth):
                self._process_last_afd_eg(pending.popleft())

        while True:
            with nvtx_range("AFD_ExpertHotLoop_AfdOverlapIter"):
                cmd = self._recv_hot_command(blocking=True)
                if cmd is None:
                    continue
                if isinstance(cmd, AfdFlushStepCmd):
                    retire(force=True)
                    continue
                if isinstance(cmd, AfdStopCmd):
                    retire(force=True)
                    return
                if isinstance(cmd, AfdProfilerCmd):
                    self.handle_profiler_cmd(cmd)
                    continue
                if not isinstance(cmd, AfdRunEGStepCmd):
                    raise RuntimeError(f"Unsupported EG command: {type(cmd).__name__}")
                pending.append(
                    self._run_afd_eg_step(cmd.plan, sent_ns=cmd.sent_ns)
                )
                retire()

    def _log_plan_recv(
        self,
        cmd: AfdCommand,
        *,
        blocking: bool,
        recv_wait_ms: float,
    ) -> None:
        if not isinstance(cmd, AfdRunEGStepCmd):
            return
        if not self._control_profile_enabled(int(cmd.plan.step_id)):
            return
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] plan_recv kind=mlp "
            f"step_id={cmd.plan.step_id} worker_rank={self.worker_rank} "
            f"blocking={int(blocking)} recv_wait_ms={float(recv_wait_ms):.3f}",
        )

    # ------------------------------------------------------------------
    # BaseAfdWorker hooks
    # ------------------------------------------------------------------

    def _create_runtime(self, config: AfdRuntimeConfig):
        return AfdExpertState(config)
