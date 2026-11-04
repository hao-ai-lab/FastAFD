from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from minisgl.utils import init_logger

from .supervisor import (
    _format_failures,
    _send_exit_msg,
    _shutdown_processes,
    _spawn_tokenizer_processes,
    _wait_for_acks,
)
from .supervisor_ray import (
    _RAY_NSYS_BIN_ENV,
    _build_ray_nsight_options,
    _build_ray_runtime_env,
    _get_ray_node_resource_key,
    _get_ray_nsys_output_prefix,
    _preflight_ray_nsys,
    _preflight_runtime_versions,
    _resolve_nsys_bin,
)

if TYPE_CHECKING:
    from .args import ServerArgs


# ---------------------------------------------------------------------------
# AfdBackendSupervisor
# ---------------------------------------------------------------------------

class AfdBackendSupervisor:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.logger = init_logger(__name__, "afd-backend-supervisor")
        self.ray = None
        self.backend_actor: Any = None
        self.tokenizer_processes: list[mp.Process] = []
        self.ack_queue: mp.Queue[str] | None = None
        self.owns_ray = False

    def start(self) -> None:
        import os

        import ray

        from minisgl.afd_coordinator import AfdCoordinator

        mp.set_start_method("spawn", force=True)
        self.ray = ray
        if not ray.is_initialized():
            try:
                ray.init(address=self.server_args.ray_address)
            except Exception as exc:
                if self.server_args.ray_address == "auto":
                    raise RuntimeError(
                        "AFD backend requires an existing external Ray cluster when "
                        "--ray-address is left at the default 'auto'. Start one with "
                        "`ray start --head`, or override with --ray-address local to "
                        "let minisgl create a local Ray runtime."
                    ) from exc
                raise
            self.owns_ray = True

        # (c) multi-node detection
        live_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
        multi_node = len(live_nodes) > 1
        if multi_node:
            self.logger.info("AFD multi-node execution detected; enabling node-scoped JIT caches")

        # (a) nsys support
        driver_nsys_path = None
        if self.server_args.ray_nsys:
            driver_nsys_path, driver_nsys_error = _resolve_nsys_bin()
            if driver_nsys_path is None:
                raise RuntimeError(
                    "AFD backend requires "
                    f"{_RAY_NSYS_BIN_ENV} to point to a valid `nsys` binary. "
                    f"Driver check failed: {driver_nsys_error}"
                )
            # Nodes that will host an nsight actor: the coordinator (driver node, see (e)) and every AG/EG
            # worker, placed by the same function the coordinator uses (afd_worker_launcher.py).
            from minisgl.afd_support import afd_dp_node_layout

            placement = afd_dp_node_layout(
                attn_dp_size=self.server_args.afd_attn_dp_size,
                attn_tp_size=self.server_args.afd_attn_tp_size,
                mlp_dp_size=self.server_args.afd_mlp_dp_size,
                mlp_tp_size=self.server_args.afd_mlp_tp_size,
            )
            nsight_nodes: list[str] = []
            for rank_nodes in (*placement.attn_rank_nodes, *placement.mlp_rank_nodes):
                nsight_nodes.extend(rank_nodes)
            driver_ip = ray.util.get_node_ip_address()
            if any(n.get("NodeManagerAddress") == driver_ip for n in live_nodes):
                nsight_nodes.append(driver_ip)
            preflight = _preflight_ray_nsys(ray, nsight_nodes, driver_nsys_path)
            self.logger.info("AFD nsys preflight passed: %s", {ip: r["which_nsys"] for ip, r in preflight.items()})
            nsys_prefix = _get_ray_nsys_output_prefix(self.server_args)
            nsys_dir = os.path.dirname(nsys_prefix)
            if nsys_dir:
                os.makedirs(nsys_dir, exist_ok=True)
            self.logger.info("AFD backend Nsight Systems profiling enabled; output prefix: %s", nsys_prefix)

        # (d) driver/worker env guard: every node that will host an AFD actor (the coordinator on this
        # driver's node, see (e); the AG/EG workers where afd_worker_launcher.py places them) must run
        # the driver's package versions, or its workers mix two envs (see _preflight_runtime_versions).
        from minisgl.afd_support import afd_dp_node_layout

        layout = afd_dp_node_layout(
            attn_dp_size=self.server_args.afd_attn_dp_size,
            attn_tp_size=self.server_args.afd_attn_tp_size,
            mlp_dp_size=self.server_args.afd_mlp_dp_size,
            mlp_tp_size=self.server_args.afd_mlp_tp_size,
        )
        actor_nodes: list[str] = []
        for rank_nodes in (*layout.attn_rank_nodes, *layout.mlp_rank_nodes):
            actor_nodes.extend(rank_nodes)
        this_ip = ray.util.get_node_ip_address()
        if any(n.get("NodeManagerAddress") == this_ip for n in live_nodes):
            actor_nodes.append(this_ip)
        env_warning = _preflight_runtime_versions(ray, actor_nodes)
        if env_warning is not None:
            self.logger.warning(env_warning)

        # (b) log dir
        if self.server_args.ray_log_dir:
            os.makedirs(self.server_args.ray_log_dir, exist_ok=True)
            self.logger.info("AFD backend log directory: %s", self.server_args.ray_log_dir)

        # Build coordinator runtime env
        coordinator_runtime_env: dict[str, Any] = _build_ray_runtime_env(nsys_bin=driver_nsys_path)
        coordinator_runtime_env.setdefault("env_vars", {})["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        if self.server_args.ray_nsys:
            nsight = _build_ray_nsight_options(
                self.server_args,
                rank=0,
                actor_kind="coordinator",
            )
            if nsight is not None:
                coordinator_runtime_env["nsight"] = nsight

        # (c) propagate multi_node so AfdCoordinator.start() can set up node-scoped cache
        actor_args = replace(self.server_args, ray_node_scoped_cache=multi_node)

        # (e) bind coordinator to the driver node for network proximity with tokenizers
        current_ip = ray.util.get_node_ip_address()
        actor_options: dict[str, Any] = dict(
            num_cpus=1,
            num_gpus=0,
            runtime_env=coordinator_runtime_env,
        )
        try:
            head_resource = _get_ray_node_resource_key(ray, current_ip)
            actor_options["resources"] = {head_resource: 0.001}
        except RuntimeError:
            self.logger.warning("Could not resolve head node resource key; coordinator will be placed freely")

        # AfdCoordinator IS the Ray actor — no extra wrapper needed
        actor_cls = ray.remote(
            num_cpus=1,
            num_gpus=0,
            max_restarts=0,
            max_task_retries=0,
        )(AfdCoordinator)

        self.ack_queue = mp.Queue()
        self.tokenizer_processes = _spawn_tokenizer_processes(self.server_args, self.ack_queue)

        try:
            self.backend_actor = actor_cls.options(**actor_options).remote(actor_args)
            # start() blocks inside the actor until AFD workers finish startup
            ray.get(self.backend_actor.start.remote())

            assert self.ack_queue is not None
            _wait_for_acks(
                expected_acks=self.server_args.num_tokenizer,
                ack_queue=self.ack_queue,
                processes=self.tokenizer_processes,
                logger=self.logger,
            )
        except Exception:
            self.shutdown()
            raise

    def get_failures(self) -> list[str]:
        failures = _format_failures(self.tokenizer_processes)
        if self.ray is None or self.backend_actor is None:
            return failures
        try:
            failure = self.ray.get(self.backend_actor.get_failure.remote())
        except Exception as exc:
            failures.append(f"minisgl-afd-backend(ray): {exc}")
            return failures
        if failure:
            failures.append(f"minisgl-afd-backend(ray): {failure}")
        return failures

    def shutdown(self) -> None:
        if self.backend_actor is not None and self.ray is not None:
            try:
                _send_exit_msg(self.server_args)
            except Exception:
                pass
            try:
                self.ray.get(self.backend_actor.shutdown.remote())
            except Exception:
                pass
            try:
                self.ray.kill(self.backend_actor, no_restart=True)
            except Exception:
                pass
            self.backend_actor = None

        if self.tokenizer_processes:
            _shutdown_processes(self.tokenizer_processes)
            self.tokenizer_processes = []

        if self.ack_queue is not None:
            self.ack_queue.close()
            self.ack_queue = None

        if self.owns_ray and self.ray is not None and self.ray.is_initialized():
            self.ray.shutdown()
            self.owns_ray = False
