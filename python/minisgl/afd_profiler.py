from __future__ import annotations

from typing import Any

import ray

from .afd_protocol import AfdProfilerCmd
from .afd_support import log_line


def start_nsys_runtime_capture(
    coord: Any,
    *,
    sync: bool,
    reason: str,
    via_worker_queue: bool = False,
) -> None:
    if (
        not coord.server_args.ray_nsys
        or coord._nsys_runtime_started
        or getattr(coord, "_nsys_runtime_finished", False)
    ):
        return
    if via_worker_queue:
        coord._broadcast_cmd_to_workers(
            AfdProfilerCmd(action="start", sync=bool(sync))
        )
    else:
        ray.get(
            [
                worker.start_cuda_profiler.remote(sync=bool(sync))
                for worker in coord._all_workers()
            ]
        )
    coord._nsys_runtime_started = True
    log_line(
        coord.log_path,
        f"[afd-coordinator] nsys profiler:start reason={reason} "
        f"sync={int(bool(sync))} via_worker_queue={int(bool(via_worker_queue))}",
        flush=True,
    )


def stop_nsys_runtime_capture(
    coord: Any,
    *,
    sync: bool,
    reason: str,
    via_worker_queue: bool = False,
) -> None:
    if not coord.server_args.ray_nsys or not coord._nsys_runtime_started:
        return
    if via_worker_queue:
        coord._broadcast_cmd_to_workers(
            AfdProfilerCmd(action="stop", sync=bool(sync))
        )
    else:
        ray.get(
            [
                worker.stop_cuda_profiler.remote(sync=bool(sync))
                for worker in coord._all_workers()
            ]
        )
    coord._nsys_runtime_started = False
    coord._nsys_runtime_finished = True
    log_line(
        coord.log_path,
        f"[afd-coordinator] nsys profiler:stop reason={reason} "
        f"sync={int(bool(sync))} via_worker_queue={int(bool(via_worker_queue))}",
        flush=True,
    )


def maybe_start_nsys_runtime_capture_for_step(coord: Any, step_id: int) -> None:
    if coord._nsys_start_step <= 0:
        return
    if int(step_id) >= coord._nsys_start_step:
        start_nsys_runtime_capture(
            coord,
            sync=False,
            reason=f"step={int(step_id)}",
            via_worker_queue=True,
        )


def maybe_stop_nsys_runtime_capture_after_step(coord: Any, step_id: int) -> None:
    if coord._nsys_stop_step <= 0:
        return
    if int(step_id) >= coord._nsys_stop_step:
        stop_nsys_runtime_capture(
            coord,
            sync=False,
            reason=f"step={int(step_id)}",
            via_worker_queue=True,
        )


__all__ = [
    "maybe_start_nsys_runtime_capture_for_step",
    "maybe_stop_nsys_runtime_capture_after_step",
    "start_nsys_runtime_capture",
    "stop_nsys_runtime_capture",
]
