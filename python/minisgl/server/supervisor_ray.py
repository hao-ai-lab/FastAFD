from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import threading
import traceback
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

from .supervisor import (
    _format_failures,
    _run_scheduler,
    _send_exit_msg,
    _shutdown_processes,
    _spawn_tokenizer_processes,
    _wait_for_acks,
)

if TYPE_CHECKING:
    from .args import ServerArgs


# ---------------------------------------------------------------------------
# Ray environment helpers
# ---------------------------------------------------------------------------

_RAY_ENV_KEYS = (
    "PYTHONPATH",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "MINISGL_RAY_NSYS_BIN",
    "MINISGL_RAY_NSYS_CUDA_GRAPH_TRACE",
    "MINISGL_RAY_NSYS_START_STEP",
    "MINISGL_RAY_NSYS_STOP_STEP",
    "MINISGL_AFD_CONTROL_PROFILE_START_STEP",
    "MINISGL_AFD_CONTROL_PROFILE_STOP_STEP",
    "MINISGL_NVTX_CPU_TRACE",
    "NSYS",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_IB_SL",
    "NCCL_NET_GDR_LEVEL",
    "MINISGL_AFD_ATTN_NODES",
    "MINISGL_AFD_MLP_NODES",
    "MINISGL_DEEPEP_BUILD_DIR",
    "MINISGL_DEEPGEMM_BUILD_DIR",
    "MINISGL_FP8_QUANT_BACKEND",
    "MINISGL_QK_NORM_ROPE_BACKEND",
    "MINISGL_AFD_MOE_BACKEND",
    "MINISGL_AFD_MAX_COMM_TOKENS",
    "MINISGL_AFD_DISABLE_DEEPEP_CPU_SYNC",
    "MINISGL_MEGAMOE_AG_SMS",
    "MINISGL_MEGAMOE_EG_EXPECTED_TOKENS",
    "DG_PRINT_CONFIGS",
    "MINISGL_AFD_STARTUP_TIMEOUT_S",
    "EP_NCCL_ROOT_DIR",
    "EP_SUPPRESS_NCCL_CHECK",
    "EP_JIT_CACHE_DIR",
    "N2M_M2N_GIN_BUILD_DIR",
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_NVCC_EXECUTABLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "MINISGL_QWEN35_CONV1D_EAGER",
    "MINISGL_QWEN35_GATE_EAGER",
    "DG_JIT_CACHE_DIR",
    "DG_JIT_NVCC_COMPILER",
    "DG_JIT_USE_NVRTC",
    "TRITON_PTXAS_BLACKWELL_PATH",
    "FLASHINFER_WORKSPACE_BASE",
    "TVM_FFI_CACHE_DIR",
    "TORCH_CUDA_ARCH_LIST",
    "CC",
    "CXX",
    "MAX_JOBS",
    "LOG_LEVEL",
    "LOG_PID",
)

_RAY_NSYS_BIN_ENV = "MINISGL_RAY_NSYS_BIN"
_RAY_STDIO_HANDLE = None
_NODE_SCOPED_CACHE_ENV_KEYS = (
    "FLASHINFER_WORKSPACE_BASE",
    "TVM_FFI_CACHE_DIR",
    "EP_JIT_CACHE_DIR",
    "N2M_M2N_GIN_BUILD_DIR",
    "MINISGL_DEEPEP_BUILD_DIR",
    "MINISGL_DEEPGEMM_BUILD_DIR",
    "DG_JIT_CACHE_DIR",
)


def _get_ray_node_resource_key(ray_mod, node_ip: str) -> str:
    for node in ray_mod.nodes():
        if not node.get("Alive", False):
            continue
        if node.get("NodeManagerAddress") != node_ip:
            continue
        for resource_name in node.get("Resources", {}):
            if resource_name.startswith("node:"):
                return resource_name
    raise RuntimeError(f"Could not find a live Ray node resource for {node_ip}")


def _build_ray_rank_plan(ray_mod, current_ip: str, tp_size: int) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str, int]] = []
    for node in ray_mod.nodes():
        if not node.get("Alive", False):
            continue
        gpu_count = int(node.get("Resources", {}).get("GPU", 0))
        if gpu_count <= 0:
            continue
        node_ip = node.get("NodeManagerAddress")
        if not node_ip:
            continue
        node_resource = _get_ray_node_resource_key(ray_mod, node_ip)
        candidates.append((node_ip, node_resource, gpu_count))

    if not candidates:
        raise RuntimeError("No live Ray nodes with GPU resources were found")

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (item[0] != current_ip, item[0]),
    )
    if ordered_candidates[0][0] != current_ip:
        raise RuntimeError(
            "Ray backend requires the driver node to have at least one GPU so rank0 can host control sockets"
        )

    rank_plan: list[tuple[str, str]] = []
    for node_ip, node_resource, gpu_count in ordered_candidates:
        rank_plan.extend((node_ip, node_resource) for _ in range(gpu_count))

    if len(rank_plan) < tp_size:
        raise RuntimeError(
            f"Requested tp_size={tp_size}, but Ray only exposes {len(rank_plan)} total GPU slots"
        )

    return rank_plan[:tp_size]


def _prepend_path_entry(path_value: str, entry: str) -> str:
    parts = [part for part in path_value.split(os.pathsep) if part]
    if entry in parts:
        parts.remove(entry)
    parts.insert(0, entry)
    return os.pathsep.join(parts)


def _build_ray_runtime_env(nsys_bin: str | None = None) -> dict[str, dict[str, str]]:
    env_vars = {
        key: value
        for key in _RAY_ENV_KEYS
        if (value := os.environ.get(key))
    }
    python_dir = str(Path(__file__).resolve().parents[2])
    env_vars["PYTHONPATH"] = _prepend_path_entry(
        env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", "")),
        python_dir,
    )
    if nsys_bin is not None:
        env_vars[_RAY_NSYS_BIN_ENV] = nsys_bin
        nsys_dir = os.path.dirname(nsys_bin)
        current_path = env_vars.get("PATH", os.environ.get("PATH", ""))
        env_vars["PATH"] = _prepend_path_entry(current_path, nsys_dir)
    return {"env_vars": env_vars}


def _resolve_nsys_bin(env: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    env = env or os.environ
    explicit_path = env.get(_RAY_NSYS_BIN_ENV, "")
    if not explicit_path:
        return None, f"{_RAY_NSYS_BIN_ENV} is not set"
    if not (os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK)):
        return None, f"{_RAY_NSYS_BIN_ENV}={explicit_path!r} is not an executable file"
    return os.path.realpath(explicit_path), None


def _sanitize_node_tag(node_ip: str) -> str:
    return node_ip.replace(":", "_").replace(".", "_")


def _scope_cache_dir(base_dir: str, node_ip: str) -> str:
    return os.path.join(base_dir, f"ray-node-{_sanitize_node_tag(node_ip)}")


def _prepare_ray_node_scoped_cache(server_args: ServerArgs) -> None:
    if not server_args.ray_node_scoped_cache:
        return

    import ray

    node_ip = ray.util.get_node_ip_address()
    scoped_env = {}
    for key in _NODE_SCOPED_CACHE_ENV_KEYS:
        base_dir = os.environ.get(key, "")
        if not base_dir:
            continue
        scoped_dir = _scope_cache_dir(base_dir, node_ip)
        os.makedirs(scoped_dir, exist_ok=True)
        os.environ[key] = scoped_dir
        scoped_env[key] = scoped_dir

    if scoped_env:
        print(
            f"[minisgl-ray] node-scoped cache enabled for {node_ip}: {scoped_env}",
            flush=True,
        )


def _get_ray_nsys_output_prefix(server_args: ServerArgs) -> str:
    if server_args.ray_nsys_output_prefix:
        return server_args.ray_nsys_output_prefix
    if server_args.ray_log_dir:
        return os.path.join(server_args.ray_log_dir, "nsys", "minisgl")
    return "minisgl-ray"


def _build_ray_nsight_options(
    server_args: ServerArgs,
    rank: int,
    *,
    actor_kind: str = "worker",
) -> dict[str, str] | None:
    if not server_args.ray_nsys:
        return None
    output_prefix = _get_ray_nsys_output_prefix(server_args)
    if actor_kind == "coordinator":
        return {
            "trace": "cuda,nvtx,osrt",
            "cuda-event-trace": "false",
            "kill": "none",
            "sample": "none",
            "capture-range": "none",
            "stop-on-exit": "true",
            "o": f"{output_prefix}_coordinator_%p",
        }
    if actor_kind != "worker":
        raise ValueError(f"Unknown Ray nsight actor_kind={actor_kind!r}")
    worker_options = {
        "trace": "cuda,nvtx",
        # CUDA event completion tracing can create false stream dependencies.
        # AFD model graph overlap intentionally uses CUDA events between the
        # engine and comm streams, so keep graph-node visibility but avoid
        # perturbing those dependencies under NSYS.
        "cuda-event-trace": "false",
        "kill": "none",
        "sample": "none",
        "capture-range": "cudaProfilerApi",
        "stop-on-exit": "true",
        "o": f"{output_prefix}_rank{rank}_%p",
    }
    worker_options["cuda-memory-usage"] = "true"
    graph_trace = os.environ.get("MINISGL_RAY_NSYS_CUDA_GRAPH_TRACE", "graph").strip()
    if graph_trace and graph_trace.lower() not in {"0", "false", "none", "off"}:
        worker_options["cuda-graph-trace"] = graph_trace
    return worker_options


def _build_ray_runtime_env_for_rank(
    server_args: ServerArgs, rank: int, nsys_bin: str | None = None
) -> dict[str, Any]:
    runtime_env: dict[str, Any] = _build_ray_runtime_env(nsys_bin=nsys_bin)
    nsight = _build_ray_nsight_options(server_args, rank)
    if nsight is not None:
        runtime_env["nsight"] = nsight
    return runtime_env


def _get_ray_rank_log_path(server_args: ServerArgs) -> str:
    host = socket.gethostname()
    filename = f"ray-scheduler-rank{server_args.tp_info.rank}-{host}.log"
    return os.path.join(server_args.ray_log_dir, filename)


def _redirect_stdio_to_log_file(log_path: str) -> None:
    global _RAY_STDIO_HANDLE

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()

    handle = open(log_path, "a", buffering=1, encoding="utf-8")
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    sys.stdout = handle
    sys.stderr = handle
    _RAY_STDIO_HANDLE = handle
    print(f"[minisgl-ray] redirecting stdout/stderr to {log_path}", flush=True)


# ---------------------------------------------------------------------------
# Ray actor wrapping Scheduler
# ---------------------------------------------------------------------------

def _make_ray_actor():
    import ray

    class _RaySchedulerActor:
        def __init__(self):
            self.thread: threading.Thread | None = None
            self.failure: str | None = None
            self.ready_queue: Queue[str] = Queue()

        def start(self, server_args: ServerArgs) -> None:
            if self.thread is not None:
                raise RuntimeError("Scheduler actor has already started")

            if server_args.ray_log_dir:
                _redirect_stdio_to_log_file(_get_ray_rank_log_path(server_args))
            _prepare_ray_node_scoped_cache(server_args)
            os.environ["MINISGL_DEVICE_INDEX"] = "0"

            def _target() -> None:
                try:
                    _run_scheduler(server_args, self.ready_queue)
                except BaseException as exc:
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    self.failure = f"{type(exc).__name__}: {exc}"

            self.thread = threading.Thread(
                target=_target,
                name=f"minisgl-ray-TP{server_args.tp_info.rank}",
                daemon=True,
            )
            self.thread.start()

            while True:
                try:
                    _ = self.ready_queue.get(timeout=5)
                    return
                except Empty:
                    if self.failure is not None:
                        print(
                            f"[minisgl-ray] scheduler rank {server_args.tp_info.rank} failed before ready: "
                            f"{self.failure}",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise RuntimeError(self.failure)
                    if self.thread is not None and not self.thread.is_alive():
                        print(
                            f"[minisgl-ray] scheduler rank {server_args.tp_info.rank} exited before ready",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise RuntimeError("Scheduler thread exited before ready")

        def get_failure(self) -> str | None:
            if self.failure is not None:
                return self.failure
            if self.thread is not None and not self.thread.is_alive():
                return "Scheduler thread exited unexpectedly"
            return None

        def shutdown(self, timeout_s: float = 5.0) -> str | None:
            if self.thread is not None:
                self.thread.join(timeout=timeout_s)
            return self.get_failure()

    return ray.remote(max_restarts=0, max_task_retries=0)(_RaySchedulerActor)


# ---------------------------------------------------------------------------
# RayBackendSupervisor
# ---------------------------------------------------------------------------

class RayBackendSupervisor:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.logger = init_logger(__name__, "ray-backend-supervisor")
        self.ray = None
        self.scheduler_actors: list[tuple[int, Any]] = []
        self.tokenizer_processes: list[mp.Process] = []
        self.ack_queue: mp.Queue[str] | None = None
        self.owns_ray = False

    def start(self) -> None:
        import ray

        mp.set_start_method("spawn", force=True)
        self.ray = ray
        if not ray.is_initialized():
            try:
                ray.init(address=self.server_args.ray_address)
            except Exception as exc:
                if self.server_args.ray_address == "auto":
                    raise RuntimeError(
                        "Ray backend requires an existing external Ray cluster when "
                        "--ray-address is left at the default 'auto'. Start one with "
                        "`ray start --head`, or override with --ray-address local to "
                        "let minisgl create a local Ray runtime."
                    ) from exc
                raise
            self.owns_ray = True

        current_ip = ray.util.get_node_ip_address()
        rank_plan = _build_ray_rank_plan(ray, current_ip, self.server_args.tp_info.size)
        multi_node = len({node_ip for node_ip, _ in rank_plan}) > 1
        actor_cls = _make_ray_actor()
        driver_nsys_path = None
        if self.server_args.ray_nsys:
            driver_nsys_path, driver_nsys_error = _resolve_nsys_bin()
            if driver_nsys_path is None:
                raise RuntimeError(
                    "Ray Nsight Systems profiling requires "
                    f"{_RAY_NSYS_BIN_ENV} to point to a valid `nsys` binary. "
                    f"Driver check failed: {driver_nsys_error}"
                )
        base_runtime_env = _build_ray_runtime_env(nsys_bin=driver_nsys_path)
        env_keys = sorted(base_runtime_env["env_vars"])
        self.logger.info("Ray actor env passthrough keys: %s", env_keys or "<none>")
        if multi_node:
            self.logger.info("Ray multi-node execution detected; enabling node-scoped JIT caches")
        if self.server_args.ray_log_dir:
            os.makedirs(self.server_args.ray_log_dir, exist_ok=True)
            self.logger.info("Ray scheduler logs directory: %s", self.server_args.ray_log_dir)
        if self.server_args.ray_nsys:
            nsys_prefix = _get_ray_nsys_output_prefix(self.server_args)
            nsys_dir = os.path.dirname(nsys_prefix)
            if nsys_dir:
                os.makedirs(nsys_dir, exist_ok=True)
            self.logger.info("Ray scheduler driver resolved nsys from: %s", driver_nsys_path)
            self.logger.info(
                "Ray scheduler will prepend %s to actor PATH for Nsight Systems",
                os.path.dirname(driver_nsys_path),
            )
            self.logger.info(
                "Ray scheduler Nsight Systems profiling enabled with output prefix: %s",
                nsys_prefix,
            )

        self.ack_queue = mp.Queue()
        self.tokenizer_processes = _spawn_tokenizer_processes(self.server_args, self.ack_queue)

        try:
            start_refs = []
            for rank in range(self.server_args.tp_info.size):
                node_ip, node_resource = rank_plan[rank]
                rank_args = replace(
                    self.server_args,
                    tp_info=DistributedInfo(rank, self.server_args.tp_info.size),
                    ray_node_scoped_cache=multi_node,
                )
                runtime_env = _build_ray_runtime_env_for_rank(
                    self.server_args, rank, nsys_bin=driver_nsys_path
                )
                actor = actor_cls.options(
                    num_cpus=1,
                    num_gpus=1,
                    resources={node_resource: 0.001},
                    runtime_env=runtime_env,
                ).remote()
                self.scheduler_actors.append((rank, actor))
                start_refs.append(actor.start.remote(rank_args))

            ray.get(start_refs)

            assert self.ack_queue is not None
            _wait_for_acks(
                expected_acks=self.server_args.num_tokenizer + 1,
                ack_queue=self.ack_queue,
                processes=self.tokenizer_processes,
                logger=self.logger,
            )
        except Exception:
            self.shutdown()
            raise

    def get_failures(self) -> list[str]:
        failures = _format_failures(self.tokenizer_processes)
        if self.ray is None:
            return failures
        for rank, actor in self.scheduler_actors:
            try:
                failure = self.ray.get(actor.get_failure.remote())
            except Exception as exc:
                failures.append(f"minisgl-TP{rank}-scheduler(ray): {exc}")
                continue
            if failure:
                failures.append(f"minisgl-TP{rank}-scheduler(ray): {failure}")
        return failures

    def shutdown(self) -> None:
        if self.scheduler_actors and self.ray is not None:
            try:
                _send_exit_msg(self.server_args)
            except Exception:
                pass

            shutdown_refs = [actor.shutdown.remote() for _, actor in self.scheduler_actors]
            try:
                self.ray.get(shutdown_refs)
            except Exception:
                pass

            for _, actor in self.scheduler_actors:
                try:
                    self.ray.kill(actor, no_restart=True)
                except Exception:
                    pass
            self.scheduler_actors = []

        if self.tokenizer_processes:
            _shutdown_processes(self.tokenizer_processes)
            self.tokenizer_processes = []
        if self.ack_queue is not None:
            self.ack_queue.close()
            self.ack_queue = None

        if self.owns_ray and self.ray is not None and self.ray.is_initialized():
            self.ray.shutdown()
            self.owns_ray = False
