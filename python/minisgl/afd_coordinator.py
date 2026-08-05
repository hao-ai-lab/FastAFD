from __future__ import annotations

import gc
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray
import zmq

from minisgl.core import Batch, Req
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeBatchMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from .afd_protocol import (
    AfdModelPlan,
    AfdCommand,
    AfdReply,
    AfdStopCmd,
    AfdTopology,
    AfdFlushStepCmd,
    AfdAGStepReply,
    AfdEGStepPlan,
    AfdRunAGStepCmd,
    AfdRunEGStepCmd,
    build_afd_ag_plan,
)
from minisgl.scheduler.scheduler import (
    _prepare_user_msg,
)
from minisgl.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    init_logger,
    init_nvtx_cpu_trace,
    load_tokenizer,
    nvtx_label,
    nvtx_range,
    serialize_zmq_payload,
    shutdown_nvtx_cpu_trace,
)

from .afd_support import (
    build_runtime_sizing,
    flush_log_lines,
    log_line,
    normalize_attention_backend_page_size,
)
from .afd_profiler import (
    maybe_start_nsys_runtime_capture_for_step,
    maybe_stop_nsys_runtime_capture_after_step,
    stop_nsys_runtime_capture,
)
from .afd_scheduler import CentralizedAfdDpScheduler
from .afd_worker_launcher import startup_afd_workers

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from minisgl.server.args import ServerArgs


logger = init_logger(__name__, "afd-coordinator")


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


class _AfdForwardRemain:
    """State left by completed forwards and consumed by later forward inputs."""

    def __init__(self) -> None:
        self._generation_by_table: dict[int, int] = {}

    def generation(self, table_idx: int) -> int:
        return self._generation_by_table.get(int(table_idx), 0)

    def empty(self) -> bool:
        return not self._generation_by_table

    def forget_tables(self, table_indices: tuple[int, ...]) -> None:
        for table_idx in table_indices:
            table_idx = int(table_idx)
            self._generation_by_table[table_idx] = self._generation_by_table.get(table_idx, 0) + 1


@dataclass(frozen=True)
class _AfdDpLaunchBatch:
    dp_rank: int
    model_plan: AfdModelPlan
    batch: Batch
    is_dummy: bool


_MAX_DRAIN_PER_SOCKET = 256


class AfdCoordinator:
    """
    AFD coordinator: the single Ray actor that owns the request queue and
    drives each inference step by orchestrating AFD worker actors on the
    attention and model nodes.
    """

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.max_batched_tokens = int(server_args.afd_max_batched_tokens)
        if self.max_batched_tokens < 1:
            raise RuntimeError("--afd-max-batched-tokens must be >= 1")
        self.max_seq_len = int(server_args.max_seq_len)
        self.device_comm_num_sms = max(1, int(server_args.afd_device_comm_num_sms))
        self.async_pending_steps = max(
            2, int(server_args.afd_async_pending_steps)
        )
        if str(server_args.cache_type) != "naive":
            raise RuntimeError(
                "AFD serve uses the centralized scheduler and currently supports only cache_type='naive'"
            )
        self._rpc_timeout_ms = max(30_000, int(float(server_args.distributed_timeout) * 1000))
        self.attn_dp_size = int(getattr(server_args, "afd_attn_dp_size", 1))
        self.mlp_dp_size = int(getattr(server_args, "afd_mlp_dp_size", 1))
        self.attn_tp_size = int(server_args.afd_attn_tp_size)
        self.mlp_tp_size = int(server_args.afd_mlp_tp_size)
        self._layout = self._validate_dp_layout()
        self._requested_attention_backend = str(server_args.attention_backend)
        self._requested_page_size = int(server_args.page_size)
        self._needs_gpu_backend_resolution = (
            self._requested_attention_backend == "auto" and self._requested_page_size != 1
        )
        if self._needs_gpu_backend_resolution:
            self.attention_backend = self._requested_attention_backend
            page_size = self._requested_page_size
        else:
            backend, page_size = normalize_attention_backend_page_size(
                self._requested_attention_backend,
                self._requested_page_size,
            )
            self.attention_backend = str(backend)
        self.page_size = int(page_size)
        self.decode_graph_bs: tuple[int, ...] = ()
        self.max_graph_bs = 0
        self.max_running_req = 1
        self.attn_max_running_req = 1
        self.num_page_override: int | None = 2
        self.max_comm_tokens = 1
        self.afd_num_mb = 1
        self.enable_attention_decode_graph = bool(server_args.afd_enable_attention_decode_graph)
        self.enable_model_decode_graph = bool(server_args.afd_enable_model_decode_graph)
        self._control_profile_start_step = _env_int(
            "MINISGL_AFD_CONTROL_PROFILE_START_STEP",
            0,
        )
        self._control_profile_stop_step = _env_int(
            "MINISGL_AFD_CONTROL_PROFILE_STOP_STEP",
            0,
        )
        self._log_afd_steps = bool(_env_int("MINISGL_AFD_LOG_STEPS", 0))
        self._log_plan_send_groups = bool(
            _env_int("MINISGL_AFD_LOG_PLAN_SEND_GROUPS", 0)
        )
        self._refresh_runtime_sizing()
        self._init_report_logging()
        self._init_frontend_queues()
        self._init_runtime_state()

    def _profile_control_step(self, step_id: int) -> bool:
        start = int(self._control_profile_start_step)
        stop = int(self._control_profile_stop_step)
        if start <= 0 and stop <= 0:
            return False
        step_id = int(step_id)
        return (start <= 0 or step_id >= start) and (stop <= 0 or step_id <= stop)

    def _init_report_logging(self) -> None:
        server_args = self.server_args
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if server_args.afd_report_dir:
            self.report_dir = Path(server_args.afd_report_dir).resolve()
        else:
            self.report_dir = (PROJECT_ROOT / "reports" / f"afd_online_{timestamp}").resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = str(self.report_dir / "afd_coordinator.log")
        self._cpu_trace_path = init_nvtx_cpu_trace(
            output_dir=self.report_dir,
            process_label="coordinator",
            enabled=bool(server_args.ray_nsys),
            metadata={
                "kind": "afd-coordinator",
                "ray_nsys": bool(server_args.ray_nsys),
                "attn_dp_size": self.attn_dp_size,
                "mlp_dp_size": self.mlp_dp_size,
                "attn_tp_size": self.attn_tp_size,
                "mlp_tp_size": self.mlp_tp_size,
                "max_seq_len": self.max_seq_len,
                "max_batched_tokens": self.max_batched_tokens,
            },
        )

    def _init_frontend_queues(self) -> None:
        server_args = self.server_args
        self._tokenizer_inbox = ZmqPullQueue(
            server_args.zmq_backend_addr,
            create=True,
            decoder=BaseBackendMsg.decoder,
        )
        self._send_into_detokenizer: ZmqPushQueue[BaseTokenizerMsg] | None = ZmqPushQueue(
            server_args.zmq_detokenizer_addr,
            create=bool(server_args.backend_create_detokenizer_link),
            encoder=BaseTokenizerMsg.encoder,
        )

    def _init_runtime_state(self) -> None:
        self.attn_workers: list[ray.actor.ActorHandle] = []
        self.mlp_workers: list[ray.actor.ActorHandle] = []
        self._worker_hot_loop_refs: list[ray.ObjectRef] = []
        self._cmd_to_workers: list[ZmqPushQueue[AfdCommand]] = []
        self._worker_inbox: ZmqPullQueue[AfdReply] | None = None
        self._worker_cmd_addrs: list[str] = []
        self._worker_inbox_addr: str | None = None
        self._next_attn_dp_rr = 0
        self._uid_to_attn_dp: dict[int, int] = {}
        self._central_schedulers: list[CentralizedAfdDpScheduler] = []
        self._afd_reply_seen: dict[int, int] = {}
        self._afd_reply_tokens_by_dp: dict[int, dict[int, list[int]]] = {}
        self._shutdown_requested = False
        self._nsys_runtime_started = False
        self._nsys_runtime_finished = False
        self._nsys_start_step = max(0, _env_int("MINISGL_RAY_NSYS_START_STEP", 0))
        self._nsys_stop_step = max(0, _env_int("MINISGL_RAY_NSYS_STOP_STEP", 0))
        self._shutdown_timeout_s = max(
            5.0,
            float(_env_int("MINISGL_AFD_SHUTDOWN_TIMEOUT_S", 30)),
        )
        self._thread: threading.Thread | None = None
        self._failure: str | None = None

    def _refresh_runtime_sizing(self) -> None:
        sizing = build_runtime_sizing(
            explicit_decode_graph_bs=tuple(self.server_args.afd_decode_graph_bs),
            cuda_graph_max_bs=0 if self.server_args.cuda_graph_max_bs is None else int(self.server_args.cuda_graph_max_bs),
            attention_backend=self.attention_backend,
            page_size=self.page_size,
            cache_type=self.server_args.cache_type,
            afd_batch_size=int(self.server_args.afd_batch_size),
            afd_max_running_req=int(self.server_args.afd_max_running_req),
            max_batched_tokens=self.max_batched_tokens,
            afd_num_mb=int(self.server_args.afd_num_mb),
        )
        self.decode_graph_bs = sizing.decode_graph_bs
        if sizing.requested_graph_bs and not self.decode_graph_bs:
            if sizing.num_mb > 1:
                logger.warning(
                    "AFD split decode graph has no usable per-MB bucket for afd_num_mb=%d; "
                    "falling back to eager-only split-step execution.",
                    sizing.num_mb,
                )
            else:
                logger.warning(
                    "AFD decode graph is currently disabled for backend=%s page_size=%d cache_type=%s; "
                    "falling back to eager-only execution.",
                    self.attention_backend,
                    self.page_size,
                    self.server_args.cache_type,
                )
        self.max_graph_bs = sizing.max_graph_bs
        self.max_running_req = sizing.max_running_req
        # `max_running_req` is the global server capacity.  Each attention-DP
        # scheduler owns only its local round-robin shard, and its table_idx
        # namespace is local to that DP.  Using the global capacity per AG rank
        # makes per-GPU page tables/token pools grow with the number of AG nodes.
        self.attn_max_running_req = max(
            1,
            (int(self.max_running_req) + int(self._layout.attn_dp_size) - 1)
            // int(self._layout.attn_dp_size),
        )
        self.num_page_override = sizing.num_page_override
        self.max_comm_tokens = sizing.max_comm_tokens
        # A single MLP-DP lane may merge multiple attention-DP lanes.  The
        # shared GIN slot layout must cover the largest merged MLP microbatch
        # even though each attention DP still schedules only its local
        # max_batched_tokens budget.
        self.max_comm_tokens *= self._layout.attn_fanin_per_mlp_dp
        self.afd_num_mb = sizing.num_mb

    def _validate_dp_layout(self) -> AfdTopology:
        try:
            return AfdTopology(
                attn_dp_size=self.attn_dp_size,
                mlp_dp_size=self.mlp_dp_size,
                attn_tp_size=self.attn_tp_size,
                mlp_tp_size=self.mlp_tp_size,
                ep_size=int(getattr(self.server_args, "afd_mlp_ep_size", 1)),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def _init_worker_queues(self) -> None:
        num_workers = len(self._all_workers())
        self._close_worker_queues()
        bind_addr = f"tcp://{ray.util.get_node_ip_address()}:*"
        worker_cmd_addrs: list[str] = []
        cmd_to_workers: list[ZmqPushQueue[AfdCommand]] = []
        worker_inbox: ZmqPullQueue[AfdReply] | None = None
        try:
            for _ in range(num_workers):
                queue = ZmqPushQueue(
                    bind_addr,
                    create=True,
                    encoder=AfdCommand.encoder,
                    serde="pickle",
                )
                cmd_to_workers.append(queue)
                worker_cmd_addrs.append(queue.addr)
            worker_inbox = ZmqPullQueue(
                bind_addr,
                create=True,
                decoder=AfdReply.decoder,
                serde="pickle",
            )
        except zmq.error.ZMQError as exc:
            if worker_inbox is not None:
                try:
                    worker_inbox.stop()
                except Exception:
                    pass
            for queue in cmd_to_workers:
                try:
                    queue.stop()
                except Exception:
                    pass
            raise RuntimeError("Failed to bind AFD worker ZMQ queues") from exc

        self._worker_cmd_addrs = worker_cmd_addrs
        self._worker_inbox_addr = worker_inbox.addr
        self._cmd_to_workers = cmd_to_workers
        self._worker_inbox = worker_inbox

    def _close_worker_queues(self) -> None:
        if self._worker_inbox is not None:
            try:
                self._worker_inbox.stop()
            except Exception:
                pass
            self._worker_inbox = None
        for queue in self._cmd_to_workers:
            try:
                queue.stop()
            except Exception:
                pass
        self._cmd_to_workers = []
        self._worker_cmd_addrs = []
        self._worker_inbox_addr = None

    def _broadcast_cmd_to_workers(self, cmd: AfdCommand) -> None:
        self._send_cmd_to_workers(tuple(range(len(self._cmd_to_workers))), cmd)

    def _drain_queue(
        self,
        queue: Any,
        label_prefix: str,
        *,
        max_items: int = _MAX_DRAIN_PER_SOCKET,
    ) -> list[Any]:
        with nvtx_range(
            nvtx_label(
                f"{label_prefix}_First",
            )
        ):
            items = [queue.get()]
        if max_items < 1:
            return items
        if not queue.empty() and len(items) < max_items:
            with nvtx_range(
                nvtx_label(
                    f"{label_prefix}_Drain",
                )
            ):
                while len(items) < max_items and not queue.empty():
                    items.append(queue.get())
        return items

    def _recv_worker_inbox(self) -> list[AfdReply]:
        if self._worker_inbox is None:
            raise RuntimeError("Worker reply queue is not initialized")
        start_ns = time.perf_counter_ns()
        replies = self._drain_queue(
            self._worker_inbox,
            "AFD_Coordinator_RecvWorkerReply",
        )
        done_ns = time.perf_counter_ns()
        recv_wall_ns = time.time_ns()
        self._log_worker_reply_receives(
            replies,
            drain_ms=(done_ns - start_ns) / 1_000_000.0,
            recv_wall_ns=recv_wall_ns,
        )
        return replies

    def _log_worker_reply_receives(
        self,
        replies: list[AfdReply],
        *,
        drain_ms: float,
        recv_wall_ns: int,
    ) -> None:
        profiled = [
            reply
            for reply in replies
            if reply.step_id is not None and self._profile_control_step(int(reply.step_id))
        ]
        if not profiled:
            return
        steps = sorted({int(reply.step_id) for reply in profiled if reply.step_id is not None})
        kinds: dict[str, int] = {}
        latencies_by_kind: dict[str, list[float]] = {}
        for reply in profiled:
            key = type(reply).__name__
            kinds[key] = kinds.get(key, 0) + 1
            sent_ns = int(getattr(reply, "sent_ns", 0) or 0)
            if sent_ns > 0:
                latencies_by_kind.setdefault(key, []).append(
                    (int(recv_wall_ns) - sent_ns) / 1_000_000.0
                )
        latency_parts: list[str] = []
        for key in sorted(latencies_by_kind):
            vals = latencies_by_kind[key]
            if not vals:
                continue
            latency_parts.append(
                f"{key}:avg={sum(vals) / len(vals):.3f},"
                f"min={min(vals):.3f},max={max(vals):.3f},n={len(vals)}"
            )
        latency_text = (
            " cmd_to_reply_ms={" + "; ".join(latency_parts) + "}"
            if latency_parts
            else ""
        )
        log_line(
            self.log_path,
            f"[afd-coordinator] reply_recv_batch count={len(profiled)} "
            f"steps={steps[:6]} kinds={kinds} drain_ms={float(drain_ms):.3f}"
            f"{latency_text}",
        )

    def _recv_tokenizer_msgs(self) -> list[BaseBackendMsg]:
        return self._drain_queue(self._tokenizer_inbox, "AFD_Coordinator_RecvTokenizer")

    # ------------------------------------------------------------------
    # Ray actor interface (used by AfdBackendSupervisor)
    # ------------------------------------------------------------------

    def _log_ready_banner(self) -> None:
        log_line(
            self.log_path,
            f"[afd-coordinator] ready max_seq_len={self.max_seq_len} "
            f"attn_dp={self.attn_dp_size} mlp_dp={self.mlp_dp_size} "
            f"attn_tp={self.attn_tp_size} mlp_tp={self.mlp_tp_size} "
            f"max_running_req={self.max_running_req} "
            f"page_size={self.page_size} "
            f"attention_backend={self.attention_backend} "
            f"max_batched_tokens={self.max_batched_tokens} "
            f"decode_graph_bs={list(self.decode_graph_bs)} "
            f"disable_overlap={bool(self.server_args.afd_disable_overlap)} "
            f"async_pending_steps={self.async_pending_steps} "
            f"centralized_scheduler=True",
            flush=True,
        )

    def _shutdown_call(self, step: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            log_line(
                self.log_path,
                f"[afd-coordinator] shutdown {step} failed: {exc!r}",
                flush=True,
            )

    def start(self) -> None:
        """Ray actor entry point.

        Blocks until all AFD workers are warmed up, then starts the event loop
        in a daemon thread and returns — so the supervisor's
        ``ray.get(actor.start.remote())`` unblocks exactly when the cluster is
        ready to serve requests.
        """
        self._redirect_stdio_to_ray_log()
        self._log_node_scoped_cache_mode()
        startup_afd_workers(self)
        self._log_ready_banner()
        self._start_loop_thread()

    def _redirect_stdio_to_ray_log(self) -> None:
        if not self.server_args.ray_log_dir:
            return
        log_path = os.path.join(self.server_args.ray_log_dir, "afd-coordinator.log")
        os.makedirs(self.server_args.ray_log_dir, exist_ok=True)
        sys.stdout.flush()
        sys.stderr.flush()
        handle = open(log_path, "a", buffering=1, encoding="utf-8")
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        sys.stdout = sys.stderr = handle

    def _log_node_scoped_cache_mode(self) -> None:
        # Worker actors apply their own node-scoped cache paths. Mutating the
        # coordinator env here would leak this node's cache path to remote actors.
        if not self.server_args.ray_node_scoped_cache:
            return
        node_ip = ray.util.get_node_ip_address()
        print(
            f"[afd-coordinator] node-scoped cache enabled; coordinator_node={node_ip}",
            flush=True,
        )

    def _start_loop_thread(self) -> None:
        def _target() -> None:
            try:
                self._run_loop()
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                self._failure = f"{type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=_target, name="minisgl-afd-event-loop", daemon=True)
        self._thread.start()

    def get_failure(self) -> str | None:
        if self._failure is not None:
            return self._failure
        if self._worker_hot_loop_refs and not self._shutdown_requested:
            ready_refs, _ = ray.wait(self._worker_hot_loop_refs, num_returns=1, timeout=0)
            if ready_refs:
                try:
                    result = ray.get(ready_refs[0])
                except Exception as exc:
                    return f"AFD worker hot loop failed: {exc}"
                return f"AFD worker hot loop exited unexpectedly: {result}"
        if self._thread is not None and not self._thread.is_alive():
            return "AfdCoordinator loop exited unexpectedly"
        return None

    def shutdown(self) -> None:
        self._shutdown_requested = True
        if self._cmd_to_workers:
            self._shutdown_call(
                "broadcast_stop",
                lambda: self._broadcast_cmd_to_workers(AfdStopCmd()),
            )
        if self._thread is not None:
            self._thread.join(timeout=self._shutdown_timeout_s)
        if self._worker_hot_loop_refs:
            self._shutdown_call(
                "wait_worker_hot_loops",
                self._wait_for_worker_hot_loops,
            )
            self._worker_hot_loop_refs = []
        if self._nsys_runtime_started:
            self._shutdown_call(
                "stop_cuda_profiler",
                lambda: stop_nsys_runtime_capture(self, sync=True, reason="shutdown"),
            )
        self._shutdown_call("stop_tokenizer_inbox", self._tokenizer_inbox.stop)
        if self._send_into_detokenizer is not None:
            self._shutdown_call(
                "stop_detokenizer_outbox",
                self._send_into_detokenizer.stop,
            )
            self._send_into_detokenizer = None
        self._close_worker_queues()
        shutdown_refs = []
        for worker_index, worker in enumerate(self._all_workers()):
            self._shutdown_call(
                f"submit_worker_shutdown[{worker_index}]",
                lambda worker=worker: shutdown_refs.append(worker.shutdown.remote()),
            )
        if shutdown_refs:
            self._shutdown_call("wait_worker_shutdown", lambda: ray.get(shutdown_refs))
        for worker_index, worker in enumerate(self._all_workers()):
            self._shutdown_call(
                f"kill_worker[{worker_index}]",
                lambda worker=worker: ray.kill(worker, no_restart=True),
            )
        trace_path = shutdown_nvtx_cpu_trace()
        if trace_path:
            log_line(self.log_path, f"[afd-coordinator] cpu_trace flushed path={trace_path}", flush=True)
        flush_log_lines(self.log_path)

    def _wait_for_worker_hot_loops(self) -> None:
        ray.get(
            self._worker_hot_loop_refs,
            timeout=self._shutdown_timeout_s,
        )

    def _init_centralized_schedulers(self) -> None:
        if not self.attn_workers:
            raise RuntimeError("Cannot initialize centralized scheduler without attention workers")
        info = ray.get(self.attn_workers[0].get_runtime_info.remote())
        num_pages = int(info["num_pages"])
        max_seq_len = int(info["max_seq_len"])
        eos_token_id = int(load_tokenizer(self.server_args.model_path).eos_token_id)
        mlp_fanout = self._layout.mlp_fanout_per_attn_dp
        # AFD decode is graph-first: fixed bucket padding is required so the
        # captured graph's static shapes match replay.
        _decode_graph_bs = tuple(self.decode_graph_bs)
        self._central_schedulers = [
            CentralizedAfdDpScheduler(
                dp_rank=dp_rank,
                max_running_req=self.attn_max_running_req,
                max_seq_len=max_seq_len,
                num_pages=num_pages,
                page_size=self.page_size,
                prefill_budget=self.max_batched_tokens,
                decode_graph_bs=_decode_graph_bs,
                cuda_graph_max_bs=self.max_graph_bs,
                afd_num_mb=self.afd_num_mb,
                mlp_fanout=mlp_fanout,
                eos_token_id=eos_token_id,
            )
            for dp_rank in range(self._layout.attn_dp_size)
        ]
        log_line(
            self.log_path,
            f"[afd-coordinator] centralized_scheduler ready "
            f"num_pages={num_pages} max_seq_len={max_seq_len} "
            f"cache_type={self.server_args.cache_type} "
            f"global_max_running_req={self.max_running_req} "
            f"attn_local_max_running_req={self.attn_max_running_req}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Internal: scheduler loops
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        if self._worker_inbox is None:
            raise RuntimeError("AFD worker queues are not initialized")
        # afd AG/EG drive loop is the only AFD loop on this branch.
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            self._afd_run_loop()
            return
        finally:
            if gc_was_enabled:
                gc.enable()

    def _has_local_scheduler_work(self) -> bool:
        return any(scheduler.has_runnable_reqs() for scheduler in self._central_schedulers)

    def receive_msg(self, blocking: bool = False) -> list[BaseBackendMsg]:
        """Receive tokenizer/backend messages, matching Scheduler.receive_msg."""
        if blocking:
            with nvtx_range("AFD_Coordinator_ReceiveMsg_Blocking"):
                return self._recv_tokenizer_msgs()
        if self._tokenizer_inbox.empty():
            return []
        with nvtx_range("AFD_Coordinator_ReceiveMsg_Drain"):
            return self._recv_tokenizer_msgs()

    def _process_one_msg(
        self,
        msg: BaseBackendMsg,
        forward_remains: list[_AfdForwardRemain],
    ) -> bool:
        with nvtx_range(
            nvtx_label(
                "AFD_Coordinator_ProcessMsg",
                msg_type=type(msg).__name__,
            )
        ):
            if isinstance(msg, BatchBackendMsg):
                return self._process_batch_msg(msg.data, forward_remains)
            if isinstance(msg, ExitMsg):
                self._shutdown_requested = True
                self._broadcast_cmd_to_workers(AfdStopCmd())
                return False
            if isinstance(msg, UserMsg):
                self._process_user_msgs([msg])
                return True
            if isinstance(msg, AbortBackendMsg):
                self._process_abort_msg(msg, forward_remains)
                return True
            raise NotImplementedError(f"Unsupported tokenizer msg: {type(msg)}")

    def _process_batch_msg(
        self,
        msgs: list[BaseBackendMsg],
        forward_remains: list[_AfdForwardRemain],
    ) -> bool:
        pending_users: list[UserMsg] = []

        def flush_users() -> None:
            nonlocal pending_users
            if pending_users:
                self._process_user_msgs(pending_users)
                pending_users = []

        for msg in msgs:
            if isinstance(msg, BatchBackendMsg):
                flush_users()
                if not self._process_batch_msg(msg.data, forward_remains):
                    return False
                continue
            if isinstance(msg, UserMsg):
                pending_users.append(msg)
                continue
            flush_users()
            if not self._process_one_msg(msg, forward_remains):
                return False
        flush_users()
        return True

    def _process_user_msgs(self, msgs: list[UserMsg]) -> None:
        by_dp: dict[int, list[UserMsg]] = {}
        for msg in msgs:
            if not _prepare_user_msg(
                msg=msg,
                max_seq_len=self.max_seq_len,
                warn=logger.warning,
            ):
                continue
            dp_rank = self._next_attn_dp_rr % self._layout.attn_dp_size
            self._next_attn_dp_rr += 1
            self._uid_to_attn_dp[int(msg.uid)] = int(dp_rank)
            by_dp.setdefault(int(dp_rank), []).append(msg)

        for dp_rank in sorted(by_dp):
            self._central_schedulers[dp_rank].add_reqs(by_dp[dp_rank])

    def _process_abort_msg(
        self,
        msg: AbortBackendMsg,
        forward_remains: list[_AfdForwardRemain],
    ) -> None:
        uid = int(msg.uid)
        dp_rank = self._uid_to_attn_dp.pop(uid, None)
        if dp_rank is not None:
            dp = int(dp_rank)
            freed = tuple(self._central_schedulers[dp].abort_req(uid))
            forward_remains[dp].forget_tables(freed)
        else:
            for fallback_dp_rank, scheduler in enumerate(self._central_schedulers):
                freed = tuple(scheduler.abort_req(uid))
                forward_remains[fallback_dp_rank].forget_tables(freed)

    def send_result(self, replies: list[DetokenizeMsg]) -> None:
        if not replies:
            return
        queue = self._send_into_detokenizer
        if queue is None:
            raise RuntimeError("AFD centralized detokenizer queue is not initialized")
        if len(replies) == 1:
            queue.put(replies[0])
        else:
            queue.put(BatchTokenizerMsg(data=replies))

    def send_detokenize_batch(
        self,
        uids: list[int],
        next_tokens: list[int],
        finished: list[bool],
    ) -> None:
        if not uids:
            return
        queue = self._send_into_detokenizer
        if queue is None:
            raise RuntimeError("AFD centralized detokenizer queue is not initialized")
        if len(uids) == 1:
            queue.put(
                DetokenizeMsg(
                    uid=int(uids[0]),
                    next_token=int(next_tokens[0]),
                    finished=bool(finished[0]),
                )
            )
        else:
            queue.put(
                DetokenizeBatchMsg(
                    uids=uids,
                    next_tokens=next_tokens,
                    finished=finished,
                )
            )

    def _send_cmd_to_workers(self, worker_ranks: tuple[int, ...], cmd: AfdCommand) -> None:
        if isinstance(cmd, (AfdRunAGStepCmd, AfdRunEGStepCmd)):
            kind = type(cmd).__name__
            step_id = int(cmd.plan.step_id)
            cmd.sent_ns = time.time_ns()
        else:
            kind = type(cmd).__name__
            step_id = -1
        serialize_start_ns = time.perf_counter_ns()
        raw = serialize_zmq_payload(cmd, AfdCommand.encoder, "pickle")
        serialize_done_ns = time.perf_counter_ns()
        put_start_ns = serialize_done_ns
        for worker_rank in worker_ranks:
            self._cmd_to_workers[int(worker_rank)].put_raw(raw)
        put_done_ns = time.perf_counter_ns()
        if self._log_plan_send_groups:
            log_line(
                self.log_path,
                f"[afd-coordinator] plan_send_group kind={kind} step_id={step_id} "
                f"worker_ranks={list(worker_ranks)} bytes={len(raw)} "
                f"serialize_ms={(serialize_done_ns - serialize_start_ns) / 1_000_000.0:.3f} "
                f"put_ms={(put_done_ns - put_start_ns) / 1_000_000.0:.3f}",
            )

    def _all_workers(self) -> list[ray.actor.ActorHandle]:
        return self.attn_workers + self.mlp_workers

    # ===== afd AG/EG drive loop (no merge/split) =====

    @property
    def _afd_expert_worker_ranks(self) -> tuple[int, ...]:
        layout = self._layout
        return tuple(
            layout.mlp_worker_rank(dp, tp)
            for dp in range(layout.mlp_dp_size)
            for tp in range(layout.mlp_tp_size)
        )

    def _afd_dispatch_bucket(self, model_plan: AfdModelPlan) -> int:
        """Per-rank DeepEP dispatch bucket.

        afd's AG step dispatches the WHOLE batch in one collective (no
        microbatch split on the comm path), so the bucket must be an upper
        bound on the full per-rank token count, not the per-MB slice. With
        attn_tp the sequence is replicated (not sharded) across TP ranks, so
        every AG rank dispatches all tokens.

        The AG worker materializes padded_reqs up to microbatch_offsets[-1]
        (the padded REQUEST count) and dispatches one token per decode req, so
        for a padded decode batch the dispatched token count equals the padded
        req count — which can exceed the token-offset total / graph_token_count
        computed from the unpadded batch. Include the padded req count so the
        bucket never under-sizes (else DeepEP asserts bucket < local tokens).
        """
        return max(1, int(model_plan.graph_token_count), int(model_plan.padded_size))

    def _afd_use_decode_graph(self, model_plan: AfdModelPlan, bucket: int) -> bool:
        if str(model_plan.phase) != "decode":
            return False
        if not (self.enable_attention_decode_graph and self.enable_model_decode_graph):
            return False
        if not self.decode_graph_bs:
            return False
        num_mb = max(1, len(tuple(model_plan.microbatch_offsets)) - 1)
        if int(bucket) % num_mb != 0:
            return False
        per_mb_bucket = int(bucket) // num_mb
        return any(int(bs) == per_mb_bucket for bs in self.decode_graph_bs)

    def _log_afd_step(
        self,
        *,
        step_id: int,
        model_plan: AfdModelPlan,
        batch: Batch,
        dp_rank: int,
        is_dummy: bool,
    ) -> None:
        if not self._log_afd_steps:
            return
        logger.info(
            "[afd_step] step=%d dp=%d phase=%s bs=%d padded=%d dummy=%d "
            "mb_real=%s mb_offsets=%s",
            step_id,
            int(dp_rank),
            model_plan.phase,
            batch.size,
            len(getattr(batch, "padded_reqs", batch.reqs)),
            int(bool(is_dummy)),
            tuple(int(x) for x in model_plan.microbatch_real_token_counts),
            tuple(int(x) for x in model_plan.microbatch_offsets),
        )

    @staticmethod
    def _afd_filter_valid_completion_tokens(
        batch: Batch,
        gens: tuple[int, ...],
        next_tokens: list[int],
        forward_remain: _AfdForwardRemain,
    ) -> list[int]:
        if forward_remain.empty():
            req_count = len(batch.reqs)
            if len(next_tokens) == req_count:
                return next_tokens
            return [int(next_tokens[i]) if i < len(next_tokens) else 0 for i in range(req_count)]
        valid_reqs: list[Req] = []
        valid_tokens: list[int] = []
        for i, (req, generation) in enumerate(zip(batch.reqs, gens)):
            if forward_remain.generation(int(req.table_idx)) == int(generation):
                valid_reqs.append(req)
                valid_tokens.append(int(next_tokens[i]) if i < len(next_tokens) else 0)
        if len(valid_reqs) != len(batch.reqs):
            batch.reqs = valid_reqs
        return valid_tokens

    def _afd_prepare_step_launch(
        self,
        step_id: int,
        scheds: list[CentralizedAfdDpScheduler],
        global_phase: str,
    ) -> tuple[int, list[_AfdDpLaunchBatch], str, int] | None:
        profile = self._profile_control_step(step_id)
        total_start_ns = time.perf_counter_ns()
        schedule_ns = 0
        bucket_ns = 0
        dummy_ns = 0
        real_results: list[tuple[int, AfdModelPlan, Batch]] = []
        results_by_dp: dict[int, tuple[AfdModelPlan, Batch]] = {}
        with nvtx_range(
            nvtx_label("AFD_Coordinator_ScheduleDps", step=step_id, phase=global_phase)
        ):
            for dp, sched in enumerate(scheds):
                sched_start_ns = time.perf_counter_ns()
                result = sched.schedule_step(step_id, phase=global_phase)
                schedule_ns += time.perf_counter_ns() - sched_start_ns
                if result is None:
                    continue
                model_plan, batch = result
                real_results.append((dp, model_plan, batch))
                results_by_dp[dp] = (model_plan, batch)
        if not real_results:
            if profile:
                log_line(
                    self.log_path,
                    f"[afd-control] prepare step_id={step_id} phase={global_phase} "
                    f"empty=1 schedule_ms={schedule_ns / 1_000_000.0:.3f} "
                    f"total_ms={(time.perf_counter_ns() - total_start_ns) / 1_000_000.0:.3f}",
                )
            return None

        with nvtx_range(
            nvtx_label("AFD_Coordinator_BuildDpLaunches", step=step_id, phase=global_phase)
        ):
            bucket_start_ns = time.perf_counter_ns()
            bucket = max(self._afd_dispatch_bucket(mp) for _, mp, _ in real_results)
            bucket_ns += time.perf_counter_ns() - bucket_start_ns
            dummy_padded_size = (
                int(bucket)
                if global_phase == "decode"
                else max(1, int(self.server_args.afd_num_mb))
            )
            dp_batches: list[_AfdDpLaunchBatch] = []
            for dp, sched in enumerate(scheds):
                result = results_by_dp.get(dp)
                is_dummy = result is None
                if result is None:
                    dummy_start_ns = time.perf_counter_ns()
                    result = sched.dummy_decode_step(
                        step_id,
                        padded_size=dummy_padded_size,
                    )
                    dummy_ns += time.perf_counter_ns() - dummy_start_ns
                dp_batches.append(
                    _AfdDpLaunchBatch(
                        dp_rank=dp,
                        model_plan=result[0],
                        batch=result[1],
                        is_dummy=is_dummy,
                    )
                )
        num_mbs = {int(item.model_plan.num_mb) for item in dp_batches}
        if len(num_mbs) != 1:
            raise RuntimeError(
                "AFD requires all attention DP lanes to use the same num_mb "
                f"within one collective step, got {sorted(num_mbs)}"
            )
        if profile:
            total_ns = time.perf_counter_ns() - total_start_ns
            log_line(
                self.log_path,
                f"[afd-control] prepare step_id={step_id} phase={global_phase} "
                f"real_dps={len(real_results)} dummy_dps={len(dp_batches) - len(real_results)} "
                f"bucket={int(bucket)} num_mb={next(iter(num_mbs))} "
                f"schedule_ms={schedule_ns / 1_000_000.0:.3f} "
                f"bucket_ms={bucket_ns / 1_000_000.0:.3f} "
                f"dummy_ms={dummy_ns / 1_000_000.0:.3f} "
                f"total_ms={total_ns / 1_000_000.0:.3f}",
            )
        return bucket, dp_batches, str(global_phase), next(iter(num_mbs))

    def _afd_launch_step(
        self,
        *,
        step_id: int,
        attn_tp: int,
        scheds: list[CentralizedAfdDpScheduler],
        forward_remains: list[_AfdForwardRemain],
        pending_free: dict[int, tuple[int, ...]],
        bucket: int,
        dp_batches: list[_AfdDpLaunchBatch],
        phase: str,
        num_mb: int,
    ) -> list[tuple[int, Batch, tuple[int, ...]]]:
        profile = self._profile_control_step(step_id)
        total_start_ns = time.perf_counter_ns()
        ag_build_ns = 0
        ag_send_ns = 0
        state_ns = 0
        eg_build_ns = 0
        eg_send_ns = 0
        use_graph_ns = 0
        gens_ns = 0
        step_log_ns = 0
        rank_ns = 0
        append_ns = 0
        ag_loop_ns = 0
        eg_loop_ns = 0
        state_updates: list[tuple[int, Batch]] = []
        use_graph_start_ns = time.perf_counter_ns()
        use_decode_graph = all(
            self._afd_use_decode_graph(item.model_plan, bucket)
            for item in dp_batches
        )
        use_graph_ns = time.perf_counter_ns() - use_graph_start_ns
        items: list[tuple[int, Batch, tuple[int, ...]]] = []
        with nvtx_range(nvtx_label("AFD_Coordinator_LaunchAG", step=step_id, phase=phase)):
            ag_loop_start_ns = time.perf_counter_ns()
            for item in dp_batches:
                dp = int(item.dp_rank)
                batch = item.batch
                gens_start_ns = time.perf_counter_ns()
                gens = tuple(
                    forward_remains[dp].generation(int(req.table_idx))
                    for req in batch.reqs
                )
                gens_ns += time.perf_counter_ns() - gens_start_ns
                step_log_start_ns = time.perf_counter_ns()
                self._log_afd_step(
                    step_id=step_id,
                    dp_rank=dp,
                    model_plan=item.model_plan,
                    batch=batch,
                    is_dummy=item.is_dummy,
                )
                step_log_ns += time.perf_counter_ns() - step_log_start_ns
                ag_build_start_ns = time.perf_counter_ns()
                ag_plan = build_afd_ag_plan(
                    step_id,
                    item.model_plan,
                    batch,
                    dispatch_bucket=bucket,
                    free_table_indices=pending_free[dp],
                    use_decode_graph=use_decode_graph,
                )
                ag_build_ns += time.perf_counter_ns() - ag_build_start_ns
                rank_start_ns = time.perf_counter_ns()
                ranks = tuple(range(dp * attn_tp, (dp + 1) * attn_tp))
                rank_ns += time.perf_counter_ns() - rank_start_ns
                ag_send_start_ns = time.perf_counter_ns()
                self._send_cmd_to_workers(ranks, AfdRunAGStepCmd(plan=ag_plan))
                ag_send_ns += time.perf_counter_ns() - ag_send_start_ns
                pending_free[dp] = ()
                if not item.is_dummy:
                    state_updates.append((dp, batch))
                append_start_ns = time.perf_counter_ns()
                items.append((dp, batch, gens))
                append_ns += time.perf_counter_ns() - append_start_ns
            ag_loop_ns = time.perf_counter_ns() - ag_loop_start_ns

        with nvtx_range(nvtx_label("AFD_Coordinator_LaunchEG", step=step_id, phase=phase)):
            eg_loop_start_ns = time.perf_counter_ns()
            eg_build_start_ns = time.perf_counter_ns()
            eg_plan = AfdEGStepPlan(
                step_id=step_id,
                phase=phase,  # type: ignore[arg-type]
                dispatch_bucket=bucket,
                num_mb=num_mb,
                use_decode_graph=use_decode_graph,
            )
            eg_build_ns = time.perf_counter_ns() - eg_build_start_ns
            eg_send_start_ns = time.perf_counter_ns()
            self._send_cmd_to_workers(
                self._afd_expert_worker_ranks,
                AfdRunEGStepCmd(plan=eg_plan),
            )
            eg_send_ns = time.perf_counter_ns() - eg_send_start_ns
            eg_loop_ns = time.perf_counter_ns() - eg_loop_start_ns
        if state_updates:
            with nvtx_range(
                nvtx_label("AFD_Coordinator_PostLaunchState", step=step_id, phase=phase)
            ):
                state_start_ns = time.perf_counter_ns()
                for dp, batch in state_updates:
                    for req in batch.reqs:
                        req.complete_one()
                    scheds[dp].decode_manager.filter_reqs(batch.reqs)
                state_ns += time.perf_counter_ns() - state_start_ns
        if profile:
            total_ns = time.perf_counter_ns() - total_start_ns
            tracked_ns = (
                use_graph_ns
                + gens_ns
                + step_log_ns
                + ag_build_ns
                + rank_ns
                + ag_send_ns
                + state_ns
                + append_ns
                + eg_build_ns
                + eg_send_ns
            )
            log_line(
                self.log_path,
                f"[afd-control] launch step_id={step_id} phase={phase} "
                f"dps={len(dp_batches)} bucket={int(bucket)} num_mb={int(num_mb)} "
                f"use_graph={int(bool(use_decode_graph))} "
                f"use_graph_ms={use_graph_ns / 1_000_000.0:.3f} "
                f"ag_loop_ms={ag_loop_ns / 1_000_000.0:.3f} "
                f"gens_ms={gens_ns / 1_000_000.0:.3f} "
                f"step_log_ms={step_log_ns / 1_000_000.0:.3f} "
                f"ag_build_ms={ag_build_ns / 1_000_000.0:.3f} "
                f"rank_ms={rank_ns / 1_000_000.0:.3f} "
                f"ag_send_ms={ag_send_ns / 1_000_000.0:.3f} "
                f"state_ms={state_ns / 1_000_000.0:.3f} "
                f"append_ms={append_ns / 1_000_000.0:.3f} "
                f"eg_loop_ms={eg_loop_ns / 1_000_000.0:.3f} "
                f"eg_build_ms={eg_build_ns / 1_000_000.0:.3f} "
                f"eg_send_ms={eg_send_ns / 1_000_000.0:.3f} "
                f"untracked_ms={(total_ns - tracked_ns) / 1_000_000.0:.3f} "
                f"total_ms={total_ns / 1_000_000.0:.3f}",
            )
        return items

    def _afd_run_loop(self) -> None:
        """Drive all attention-DP schedulers in lockstep. The DeepEP dispatch is a
        collective over ALL AG+EG ranks, so every global step processes every active DP
        lane together: a per-DP AfdRunAGStepCmd to that lane's AG worker ranks
        (dp*attn_tp..) + ONE shared AfdRunEGStepCmd with a UNIFORM bucket (the collective
        requires all ranks to agree on num_max_dispatch_tokens_per_rank). Per-DP
        scheduling/completion + per-DP forward_remain (table indices are per-lane).
        For n_dp=1 this is the old single-DP path without a separate implementation."""
        if self.server_args.afd_disable_overlap:
            self._afd_normal_loop()
        else:
            self._afd_overlap_loop()

    def _afd_normal_loop(self) -> None:
        """Synchronous AFD loop: launch one global step and complete it immediately."""
        n_dp = len(self._central_schedulers)
        scheds = self._central_schedulers
        attn_tp = int(self.attn_tp_size)
        forward_remains = [_AfdForwardRemain() for _ in range(n_dp)]
        pending_free = {dp: () for dp in range(n_dp)}
        step_id = 1
        log_line(
            self.log_path,
            f"[afd-overlap] mode=dp_lockstep_normal n_dp={n_dp} attn_tp={attn_tp}",
        )
        self._afd_reply_seen = {}
        self._afd_reply_tokens_by_dp = {}
        while True:
            blocking = not self._has_local_scheduler_work()
            for msg in self.receive_msg(blocking=blocking):
                if not self._process_one_msg(msg, forward_remains):
                    return

            global_phase = self._afd_next_global_phase(scheds)
            prepared = self._afd_prepare_step_launch(step_id, scheds, global_phase)
            if prepared is None:
                continue
            bucket, dp_batches, phase, num_mb = prepared
            maybe_start_nsys_runtime_capture_for_step(self, step_id)
            items = self._afd_launch_step(
                step_id=step_id,
                attn_tp=attn_tp,
                scheds=scheds,
                forward_remains=forward_remains,
                pending_free=pending_free,
                bucket=bucket,
                dp_batches=dp_batches,
                phase=phase,
                num_mb=num_mb,
            )
            for dp, freed in self._afd_complete_global(
                (step_id, items),
                scheds,
                forward_remains,
                attn_tp,
            ).items():
                pending_free[dp] = tuple(pending_free[dp]) + tuple(freed)
            maybe_stop_nsys_runtime_capture_after_step(self, step_id)
            step_id += 1

    def _afd_overlap_loop(self) -> None:
        """AFD overlap loop with a fixed finite command lead.

        AFD decode does not need next-token CPU replies to enqueue the next
        step: AG workers write sampled tokens back to their local token_pool.
        Keep a small fixed number of global steps in flight so worker queues
        stay fed, but do not pre-feed the full generation window.
        """
        n_dp = len(self._central_schedulers)
        scheds = self._central_schedulers
        attn_tp = int(self.attn_tp_size)
        forward_remains = [_AfdForwardRemain() for _ in range(n_dp)]
        max_pending = self.async_pending_steps
        log_line(
            self.log_path,
            f"[afd-overlap] mode=dp_lockstep_fixed_lead n_dp={n_dp} "
            f"attn_tp={attn_tp} max_pending={max_pending}",
        )
        self._afd_reply_seen = {}
        self._afd_reply_tokens_by_dp = {}
        pending: deque = deque()  # (global_step_id, [(dp, batch, gens), ...])
        pending_free = {dp: () for dp in range(n_dp)}
        step_id = 1
        workers_flushed = False

        def flush_pending_workers(reason: str) -> None:
            nonlocal workers_flushed
            if workers_flushed:
                return
            log_line(
                self.log_path,
                f"[afd-overlap] flush_pending_workers reason={reason} "
                f"pending={len(pending)}",
            )
            self._broadcast_cmd_to_workers(AfdFlushStepCmd())
            workers_flushed = True

        def complete_pending(pending_item) -> None:
            gsid = pending_item[0]
            for dp, freed in self._afd_complete_global(
                pending_item,
                scheds,
                forward_remains,
                attn_tp,
            ).items():
                # Keep worker KV/table state in lockstep with the central
                # scheduler's immediately reusable table manager. Per-worker
                # command FIFO means this free reaches the AG worker only after
                # all older in-flight commands that might still reference it.
                pending_free[dp] = tuple(pending_free[dp]) + tuple(freed)
            maybe_stop_nsys_runtime_capture_after_step(self, gsid)

        while True:
            blocking = not pending and not self._has_local_scheduler_work()
            for msg in self.receive_msg(blocking=blocking):
                if not self._process_one_msg(msg, forward_remains):
                    if pending:
                        flush_pending_workers("shutdown")
                    while pending:
                        complete_pending(pending.popleft())
                    return

            scheduled_none = False
            while len(pending) < max_pending:
                prepared = self._afd_prepare_step_launch(
                    step_id,
                    scheds,
                    self._afd_next_global_phase(scheds),
                )
                if prepared is None:
                    scheduled_none = True
                    break
                bucket, dp_batches, phase, num_mb = prepared
                maybe_start_nsys_runtime_capture_for_step(self, step_id)
                items = self._afd_launch_step(
                    step_id=step_id,
                    attn_tp=attn_tp,
                    scheds=scheds,
                    forward_remains=forward_remains,
                    pending_free=pending_free,
                    bucket=bucket,
                    dp_batches=dp_batches,
                    phase=phase,
                    num_mb=num_mb,
                )
                pending.append((step_id, items))
                step_id += 1
                workers_flushed = False

            if scheduled_none and pending:
                flush_pending_workers("no_schedulable_step")
                while pending:
                    complete_pending(pending.popleft())
                continue

            if pending and len(pending) >= max_pending:
                complete_pending(pending.popleft())

    @staticmethod
    def _afd_next_global_phase(scheds: list[CentralizedAfdDpScheduler]) -> str:
        # Prefill stays globally prioritized so AG/EG collectives never mix
        # prefill and decode shapes in the same step.
        if any(
            s.prefill_manager.runnable and s.table_manager.available_size > 0
            for s in scheds
        ):
            return "prefill"
        if any(s.decode_manager.runnable for s in scheds):
            return "decode"
        return "prefill" if any(s.prefill_manager.runnable for s in scheds) else "decode"

    def _afd_collect_global(self, step_id, active_dps, attn_tp):
        """Collect all AG (active lanes x attn_tp) + EG replies for the global step;
        map each AG reply's tokens to its DP lane (lane = worker_rank // attn_tp)."""
        expected = len(active_dps) * attn_tp + len(self._afd_expert_worker_ranks)
        seen = self._afd_reply_seen
        store = self._afd_reply_tokens_by_dp
        while seen.get(step_id, 0) < expected:
            for reply in self._recv_worker_inbox():
                sid = int(reply.step_id)
                seen[sid] = seen.get(sid, 0) + 1
                if isinstance(reply, AfdAGStepReply) and reply.next_tokens:
                    lane = int(reply.worker_rank) // attn_tp
                    store.setdefault(sid, {})[lane] = list(reply.next_tokens)
        seen.pop(step_id, None)
        return store.pop(step_id, {})

    def _afd_complete_global(self, pending_item, scheds, forward_remains, attn_tp):
        step_id, items = pending_item
        profile = self._profile_control_step(int(step_id))
        total_start_ns = time.perf_counter_ns()
        collect_start_ns = total_start_ns
        active_dps = [dp for dp, _, _ in items]
        with nvtx_range(nvtx_label("AFD_Coordinator_CompleteCollect", step=step_id)):
            toks_by_dp = self._afd_collect_global(step_id, active_dps, attn_tp)
        collect_done_ns = time.perf_counter_ns()

        freed_by_dp = {}
        filter_ms_sum = 0.0
        complete_ms_sum = 0.0
        send_ms_sum = 0.0
        forget_ms_sum = 0.0
        dp_total_ms_sum = 0.0
        total_valid_tokens = 0
        total_replies = 0
        total_freed = 0
        slowest_dp: tuple[int, float, int, int, int, int] | None = None
        # Accumulate all DP replies and send them once. Per-DP sends are pure
        # coordinator overhead in decode and scale poorly with attention DP.
        all_replies: list[DetokenizeMsg] = []
        all_reply_uids: list[int] = []
        all_reply_next_tokens: list[int] = []
        all_reply_finished: list[bool] = []

        for (dp, batch, gens) in items:
            dp_start_ns = time.perf_counter_ns()
            next_tokens = toks_by_dp.get(dp, [])
            fr = forward_remains[dp]
            filter_start_ns = dp_start_ns
            valid_tokens = self._afd_filter_valid_completion_tokens(
                batch,
                gens,
                next_tokens,
                fr,
            )
            filter_done_ns = time.perf_counter_ns()
            complete_start_ns = filter_done_ns
            with nvtx_range(
                nvtx_label("AFD_Coordinator_CompleteResults", step=step_id, dp=dp)
            ):
                completion = scheds[dp].complete_results(batch, valid_tokens)
            complete_done_ns = time.perf_counter_ns()
            send_start_ns = complete_done_ns
            if completion.reply_uids is not None:
                compact_tokens = completion.reply_next_tokens
                compact_finished = completion.reply_finished
                if compact_tokens is None or compact_finished is None:
                    raise RuntimeError("Incomplete compact AFD completion payload")
                reply_count = len(completion.reply_uids)
                if len(compact_tokens) != reply_count or len(compact_finished) != reply_count:
                    raise RuntimeError(
                        "Mismatched compact AFD completion payload lengths: "
                        f"uids={reply_count} tokens={len(compact_tokens)} "
                        f"finished={len(compact_finished)}"
                    )
                all_reply_uids.extend(completion.reply_uids)
                all_reply_next_tokens.extend(compact_tokens)
                all_reply_finished.extend(compact_finished)
            else:
                replies = list(completion.replies)
                reply_count = len(replies)
                all_replies.extend(replies)
            send_done_ns = time.perf_counter_ns()
            forget_start_ns = send_done_ns
            freed = tuple(completion.freed_table_indices)
            if freed:
                fr.forget_tables(freed)
            forget_done_ns = time.perf_counter_ns()

            filter_ms = (filter_done_ns - filter_start_ns) / 1_000_000.0
            complete_ms = (complete_done_ns - complete_start_ns) / 1_000_000.0
            send_ms = (send_done_ns - send_start_ns) / 1_000_000.0
            forget_ms = (forget_done_ns - forget_start_ns) / 1_000_000.0
            dp_total_ms = (forget_done_ns - dp_start_ns) / 1_000_000.0
            filter_ms_sum += filter_ms
            complete_ms_sum += complete_ms
            send_ms_sum += send_ms
            forget_ms_sum += forget_ms
            dp_total_ms_sum += dp_total_ms
            total_valid_tokens += len(valid_tokens)
            total_replies += reply_count
            total_freed += len(freed)
            if slowest_dp is None or dp_total_ms > slowest_dp[1]:
                slowest_dp = (
                    int(dp),
                    float(dp_total_ms),
                    int(len(next_tokens)),
                    int(len(valid_tokens)),
                    int(reply_count),
                    int(len(freed)),
                )
            if profile:
                log_line(
                    self.log_path,
                    f"[afd-coordinator] control_complete_dp "
                    f"step_id={int(step_id)} dp={int(dp)} "
                    f"batch_real={int(batch.size)} padded={len(batch.padded_reqs)} "
                    f"next_tokens={len(next_tokens)} valid_tokens={len(valid_tokens)} "
                    f"replies={reply_count} freed={len(freed)} "
                    f"filter_ms={filter_ms:.3f} "
                    f"complete_results_ms={complete_ms:.3f} "
                    f"send_result_ms={send_ms:.3f} "
                    f"forget_ms={forget_ms:.3f} "
                    f"total_ms={dp_total_ms:.3f}",
                )
            freed_by_dp[dp] = freed

        send_start_ns = time.perf_counter_ns()
        self.send_result(all_replies)
        self.send_detokenize_batch(
            all_reply_uids,
            all_reply_next_tokens,
            all_reply_finished,
        )
        send_done_ns = time.perf_counter_ns()
        send_result_ms = (send_done_ns - send_start_ns) / 1_000_000.0
        send_ms_sum += send_result_ms
        total_done_ns = time.perf_counter_ns()
        if profile:
            slowest_text = ""
            if slowest_dp is not None:
                slowest_text = (
                    f" slowest_dp={slowest_dp[0]} "
                    f"slowest_dp_ms={slowest_dp[1]:.3f} "
                    f"slowest_next_tokens={slowest_dp[2]} "
                    f"slowest_valid_tokens={slowest_dp[3]} "
                    f"slowest_replies={slowest_dp[4]} "
                    f"slowest_freed={slowest_dp[5]}"
                )
            log_line(
                self.log_path,
                f"[afd-coordinator] control_complete_global "
                f"step_id={int(step_id)} active_dps={len(items)} attn_tp={int(attn_tp)} "
                f"collect_ms={(collect_done_ns - collect_start_ns) / 1_000_000.0:.3f} "
                f"filter_sum_ms={filter_ms_sum:.3f} "
                f"complete_results_sum_ms={complete_ms_sum:.3f} "
                f"send_result_sum_ms={send_ms_sum:.3f} "
                f"forget_sum_ms={forget_ms_sum:.3f} "
                f"dp_total_sum_ms={dp_total_ms_sum:.3f} "
                f"total_valid_tokens={total_valid_tokens} "
                f"total_replies={total_replies} "
                f"total_freed={total_freed} "
                f"total_ms={(total_done_ns - total_start_ns) / 1_000_000.0:.3f}"
                f"{slowest_text}",
            )
        return freed_by_dp
