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
from minisgl.ray_env_keys import RAY_ACTOR_ENV_KEYS, RAY_ACTOR_NODE_SCOPED_CACHE_KEYS
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

_RAY_ENV_KEYS = RAY_ACTOR_ENV_KEYS

_RAY_NSYS_BIN_ENV = "MINISGL_RAY_NSYS_BIN"
_RAY_STDIO_HANDLE = None
_NODE_SCOPED_CACHE_ENV_KEYS = RAY_ACTOR_NODE_SCOPED_CACHE_KEYS


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


def _probe_nsys_on_node() -> dict[str, Any]:
    """Look up `nsys` the way Ray's nsight plugin does: a bare name on this process's PATH."""
    import shutil
    import subprocess

    result: dict[str, Any] = {
        "host": socket.gethostname(),
        "which_nsys": shutil.which("nsys"),
        "version_rc": None,
        "version": "",
    }
    if result["which_nsys"] is None:
        return result
    try:
        proc = subprocess.run(["nsys", "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["version"] = f"{type(exc).__name__}: {exc}"
        return result
    result["version_rc"] = proc.returncode
    result["version"] = (proc.stdout.strip() or proc.stderr.strip())[:200]
    return result


def _nsys_preflight_error(results: dict[str, dict[str, Any]], nsys_bin: str) -> str | None:
    """Return an actionable error if any node's Ray daemons cannot run `nsys`, else None.

    Args:
        results: Node IP -> `_probe_nsys_on_node()` result, or `{"error": ...}` when the probe task failed.
        nsys_bin: The driver's resolved `MINISGL_RAY_NSYS_BIN`.
    """
    failed = {ip for ip, result in results.items() if result.get("version_rc") != 0}
    if not failed:
        return None
    lines = [
        f"Ray Nsight Systems preflight failed on {len(failed)} of {len(results)} node(s). "
        "Ray's nsight runtime_env plugin starts a bare `nsys` inside each node's runtime-env agent, "
        f"which uses the Ray daemon (raylet) PATH; {_RAY_NSYS_BIN_ENV}={nsys_bin} only reaches this "
        "driver and the actor env_vars, not that agent.",
        f"Fix: append the nsys directory (for example {os.path.dirname(nsys_bin)}) to PATH in the "
        "environment that runs `ray start` on the failing node(s), restart those Ray daemons, and retry; "
        "or run without Nsight profiling (NSYS=0).",
        "Per-node result (`which nsys` / `nsys --version` as the Ray daemons see it):",
    ]
    for ip in sorted(results):
        result = results[ip]
        status = "FAIL" if ip in failed else "OK"
        if "error" in result:
            detail = f"probe task failed: {result['error']}"
        else:
            detail = f"which nsys={result.get('which_nsys')}"
            if result.get("which_nsys") is not None:
                detail += f", nsys --version rc={result.get('version_rc')}: {result.get('version')}"
        lines.append(f"  {status} {ip} ({result.get('host', '?')}): {detail}")
    return "\n".join(lines)


def _preflight_ray_nsys(ray_mod, node_ips: list[str], nsys_bin: str) -> dict[str, dict[str, Any]]:
    """Fail fast, before any actor with an `nsight` runtime_env exists, if a target node lacks `nsys`.

    Ray's nsight plugin only reports "nsight is not installed" after the actor is scheduled. This probe runs
    one CPU-only task per node with a runtime_env that sets PYTHONPATH only: actor env_vars forward the
    driver's PATH, so forwarding it here would hide exactly the raylet PATH the plugin sees.

    Returns:
        Node IP -> probe result, when every node passes.

    Raises:
        RuntimeError: With every node's result and the fix, when any node fails.
    """
    probe = ray_mod.remote(num_cpus=0, max_retries=0)(_probe_nsys_on_node)
    runtime_env = {"env_vars": {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}}
    refs = {
        ip: probe.options(
            resources={_get_ray_node_resource_key(ray_mod, ip): 0.001},
            runtime_env=runtime_env,
        ).remote()
        for ip in sorted(set(node_ips))
    }
    results: dict[str, dict[str, Any]] = {}
    for ip, ref in refs.items():
        try:
            results[ip] = ray_mod.get(ref, timeout=120)
        except ray_mod.exceptions.RayError as exc:  # report every node, not only the first failure
            results[ip] = {"error": f"{type(exc).__name__}: {exc}"}
    error = _nsys_preflight_error(results, nsys_bin)
    if error is not None:
        raise RuntimeError(error)
    return results


# Package versions an AFD worker must share with its driver. The worker runs the Python of the Ray
# daemons (raylet) on its node, while the driver's PATH, LD_LIBRARY_PATH, CUDA_HOME and CONDA_PREFIX
# reach it through runtime_env (ray_env_keys.py), so two envs with different versions would be mixed.
_RUNTIME_VERSION_DISTS = ("torch", "flashinfer-python", "flashinfer-cubin", "triton")
_RAY_PY_EXECUTABLE_ENV = "MINISGL_RAY_PY_EXECUTABLE"


def _probe_runtime_versions() -> dict[str, Any]:
    """Report this interpreter's prefix, Python patch version and the `_RUNTIME_VERSION_DISTS` versions.

    Reads package metadata only, so it neither imports torch nor initializes CUDA.
    """
    import importlib.metadata
    import platform

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for dist in _RUNTIME_VERSION_DISTS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return {"host": socket.gethostname(), "prefix": sys.prefix, "versions": versions}


def _format_versions(versions: dict[str, str | None]) -> str:
    return " ".join(f"{name}={value}" for name, value in versions.items())


def _runtime_version_check(
    driver: dict[str, Any], results: dict[str, dict[str, Any]]
) -> tuple[str | None, str | None]:
    """Compare each node's `_probe_runtime_versions()` result with the driver's.

    Args:
        driver: This driver's `_probe_runtime_versions()` result.
        results: Node IP -> probe result, or `{"error": ...}` when the probe task failed.

    Returns:
        (error, warning): an actionable error when any node's versions differ or its probe failed;
        otherwise a one-line warning when only the env prefix differs (a private clone of the same env),
        else (None, None).
    """
    mismatched = {
        ip for ip, result in results.items() if "error" in result or result["versions"] != driver["versions"]
    }
    if mismatched:
        lines = [
            f"FastAFD driver and Ray worker environments differ on {len(mismatched)} of {len(results)} node(s). "
            "AFD workers run the Python of the Ray daemons (raylet) on their node but receive this driver's "
            "PATH, LD_LIBRARY_PATH, CUDA_HOME and CONDA_PREFIX (python/minisgl/ray_env_keys.py), so each "
            "worker would mix two environments.",
            f"Driver: {driver['prefix']} {_format_versions(driver['versions'])}",
            "Fix: run the driver from the conda env the Ray daemons were started from (the env given to "
            "`ray start`, for example through scripts/enter_clean.sh; on a cluster started from a cluster.env, "
            "its CONDA_ENV), or restart the daemons from the driver's env.",
            "Per node, as its Ray daemons run it:",
        ]
        for ip in sorted(results):
            result = results[ip]
            if "error" in result:
                lines.append(f"  FAIL {ip} (?): probe task failed: {result['error']}")
                continue
            detail = f"{result['prefix']} {_format_versions(result['versions'])}"
            if ip in mismatched:
                diffs = ", ".join(
                    f"{name} {result['versions'].get(name)} (driver {value})"
                    for name, value in driver["versions"].items()
                    if result["versions"].get(name) != value
                )
                lines.append(f"  MISMATCH {ip} ({result['host']}): {detail}; differs: {diffs}")
            else:
                lines.append(f"  OK {ip} ({result['host']}): {detail}")
        return "\n".join(lines), None
    other_prefix = sorted(ip for ip, result in results.items() if result["prefix"] != driver["prefix"])
    if other_prefix:
        prefixes = sorted({results[ip]["prefix"] for ip in other_prefix})
        return None, (
            f"FastAFD driver env {driver['prefix']} is not the Ray daemons' env on {', '.join(other_prefix)} "
            f"({', '.join(prefixes)}); package versions match ({_format_versions(driver['versions'])}), continuing."
        )
    return None, None


def _preflight_runtime_versions(ray_mod, node_ips: list[str]) -> str | None:
    """Fail fast, before any AFD actor exists, if a node's Ray daemons run other package versions than this driver.

    One CPU-only task per node with a runtime_env that sets PYTHONPATH only, so it runs the interpreter
    the raylet gives an AFD worker.

    Returns:
        A warning to log (env prefix differs but versions match, or the check was skipped), else None.

    Raises:
        RuntimeError: With the driver's and every node's versions and the fix, when any node differs.
    """
    if os.environ.get(_RAY_PY_EXECUTABLE_ENV, "").strip():
        return (
            f"FastAFD driver/worker version check skipped: {_RAY_PY_EXECUTABLE_ENV} replaces the workers' "
            "interpreter, which the probe cannot reproduce."
        )
    probe = ray_mod.remote(num_cpus=0, max_retries=0)(_probe_runtime_versions)
    runtime_env = {"env_vars": {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}}
    refs = {
        ip: probe.options(
            resources={_get_ray_node_resource_key(ray_mod, ip): 0.001},
            runtime_env=runtime_env,
        ).remote()
        for ip in sorted(set(node_ips))
    }
    results: dict[str, dict[str, Any]] = {}
    for ip, ref in refs.items():
        try:
            results[ip] = ray_mod.get(ref, timeout=120)
        except ray_mod.exceptions.RayError as exc:  # report every node, not only the first failure
            results[ip] = {"error": f"{type(exc).__name__}: {exc}"}
    error, warning = _runtime_version_check(_probe_runtime_versions(), results)
    if error is not None:
        raise RuntimeError(error)
    return warning


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
            preflight = _preflight_ray_nsys(ray, [node_ip for node_ip, _ in rank_plan], driver_nsys_path)
            self.logger.info("Ray nsys preflight passed: %s", {ip: r["which_nsys"] for ip, r in preflight.items()})
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
