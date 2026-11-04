"""CAD node runtime: one GPU's decode steps (attention or model side), eager or as one CUDA graph per bucket.

The program per GPU is fixed (cad/program.py), so the graph key is the per-mb bucket alone, and there is no program
cache, no planner, no clock stamps, no task markers.

  iteration()     every task of the program on the current stream (`getattr(tasks, task.kind)(task.layer, task.mb)`),
                  then ops.bump_epoch(window.epoch) as the step's last kernel (graph replays stay flag-balanced:
                  every flag is epoch + 1 of the step that released it)
  warm(buckets)   collective, the same bucket order on every GPU of the CAD world: per bucket, bind it, one eager step
                  with dummy inputs together with every peer (real cross-GPU flags; every kernel the step launches is
                  loaded before any graph replays it), check the abort word, then capture one graph of iteration()
  run_step(cmd)   bind cmd.bucket, fill the bucket's fixed buffers from the step command (tasks.fill), replay the
                  bucket's graph when cmd.use_graph and one was captured, else the same iteration() eagerly; returns
                  (done event, finish): finish() after the step synchronized reads the abort word (copied to a host slot
                  on the step's stream, so it never waits for later steps) and raises CadStepAborted if a wait timed out

The lazy-load hazard: the first eager launch of a not-yet-loaded kernel behind a pending flag wait can
block in the launch; CAD workers run with CUDA_MODULE_LOADING=EAGER, checked at warm().

Task executors (cad/attn_side.py CadAttnTasks, cad/model_side.py CadModelTasks) provide: bind(bucket), fill(cmd),
fill_dummy(bucket) and one method per task kind; every op stream-ordered with no host sync (graph-capturable). The GPU
primitives sit behind `backend` (TorchCadBackend here), the hop kernels behind `ops`
(kernel/cad_hop.py).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence

from minisgl.cad.program import Task

MODULE_LOADING_ENV = "CUDA_MODULE_LOADING"


class CadStepAborted(RuntimeError):
    """A flag wait of the step timed out (the window's abort word is set): the step's results are not valid."""


class CadRuntimeError(RuntimeError):
    pass


def check_module_loading(environ: Mapping[str, str] | None = None) -> None:
    """CUDA_MODULE_LOADING=EAGER must be set before CUDA init."""
    env = os.environ if environ is None else environ
    raw = env.get(MODULE_LOADING_ENV, "")
    if raw.strip().upper() != "EAGER":
        raise CadRuntimeError(f"CAD needs {MODULE_LOADING_ENV}=EAGER before CUDA init, this worker has {raw or 'unset'} "
                              "(a lazily loaded kernel behind a pending flag wait can hang the step)")


class TorchCadBackend:
    """The GPU side of the runtime: graphs on one capture stream, replays, events, host barriers (gloo)."""

    def __init__(self, *, device, cpu_group=None, capture_stream=None):
        self.device, self.cpu_group, self.capture_stream = device, cpu_group, capture_stream
        self._pool = None

    def capture(self, fn: Callable[[], None]):
        import torch

        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=self.device)
        stream = self.capture_stream
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, pool=self._pool, stream=stream):
            fn()
        self._pool = graph.pool()
        torch.cuda.current_stream().wait_stream(stream)
        return graph

    def replay(self, graph) -> None:
        graph.replay()

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize(self.device)

    def barrier(self) -> None:
        if self.cpu_group is not None:
            import torch.distributed as dist

            dist.barrier(group=self.cpu_group)

    def event(self):
        import torch

        ev = torch.cuda.Event()
        ev.record()
        return ev

    def host_slot(self, like):
        import torch

        return torch.zeros(1, dtype=like.dtype, pin_memory=like.device.type == "cuda")


class CadNodeRuntime:
    def __init__(self, tasks, program: Sequence[Task], window, *, ops=None, backend=None,
                 log: Callable[[str], None] = print, label: str = "cad", check_loading: bool = True):
        self.tasks, self.program, self.win = tasks, tuple(program), window
        if ops is None:
            from minisgl.kernel import cad_hop as ops
        self.ops, self.backend, self.log, self.label = ops, backend, log, label
        self.check_loading = bool(check_loading)
        self.graphs: dict[int, object] = {}
        self.warmed: list[int] = []
        self._err_slots: list = []
        kinds = {t.kind for t in self.program}
        missing = sorted(k for k in kinds if not callable(getattr(tasks, k, None)))
        if missing:
            raise CadRuntimeError(f"{label}: the task executor has no {missing} for its program")

    # ------------------------------------------------------------------ one step's ops
    def iteration(self) -> None:
        """Enqueue one decode step on the current stream (eager or under capture)."""
        tasks = self.tasks
        for t in self.program:
            getattr(tasks, t.kind)(t.layer, t.mb)
        self.ops.bump_epoch(self.win.epoch)

    # ------------------------------------------------------------------ start-up
    def _check_bucket(self, bucket: int) -> int:
        bucket = int(bucket)
        t_max = int(self.win.layout.t_max)
        if not 0 <= bucket <= t_max:
            raise CadRuntimeError(f"{self.label}: bucket {bucket} outside the window's 0..{t_max} rows")
        return bucket

    def warm(self, buckets: Sequence[int], *, graphs: bool = True) -> list[int]:
        """Collective (every GPU of the CAD world, the same buckets in the same order): per bucket one paired eager step
        with dummy inputs, the abort word checked, then (graphs) one capture. Returns the captured buckets."""
        if self.check_loading:
            check_module_loading()
        order = sorted({self._check_bucket(b) for b in buckets})
        for bucket in order:
            self.tasks.bind(bucket)
            self.tasks.fill_dummy(bucket)
            self.iteration()
            self.backend.synchronize()
            self._raise_if_aborted(int(self.win.err.item()), f"warm-up step of bucket {bucket}")
            self.warmed.append(bucket)
            if graphs:
                self.graphs[bucket] = self.backend.capture(self.iteration)
        self.backend.barrier()
        self.log(f"[{self.label}] warmed buckets {order}, graphs {sorted(self.graphs)}")
        return sorted(self.graphs)

    # ------------------------------------------------------------------ steps
    def run_step(self, cmd):
        """Enqueue one decode step (AfdRunCadStepCmd) on the current stream; returns (done event, finish)."""
        bucket = self._check_bucket(cmd.bucket)
        graph = self.graphs.get(bucket) if cmd.use_graph else None
        if cmd.use_graph and graph is None:
            raise CadRuntimeError(f"{self.label}: step {cmd.step_id} asks for the graph of bucket {bucket}, "
                                  f"captured {sorted(self.graphs)}")
        if not cmd.use_graph and bucket not in self.warmed:
            raise CadRuntimeError(f"{self.label}: step {cmd.step_id} runs bucket {bucket} eagerly, never warmed "
                                  f"(warmed {sorted(self.warmed)}): its first kernels would load behind a pending wait")
        self.tasks.bind(bucket)
        self.tasks.fill(cmd)
        if graph is not None:
            self.backend.replay(graph)
        else:
            self.iteration()
        err_host = self._abort_word_copy()
        done = self.backend.event()
        step_id = int(cmd.step_id)

        def finish() -> dict:
            done.synchronize()
            err = int(err_host[0])
            self._err_slots.append(err_host)
            self._raise_if_aborted(err, f"step {step_id}")
            return {"step_id": step_id, "source": "graph" if graph is not None else "eager", "err": err}

        return done, finish

    def _abort_word_copy(self):
        err = self.win.err
        host = self._err_slots.pop() if self._err_slots else self.backend.host_slot(err)
        host.copy_(err, non_blocking=getattr(err, "device", None) is not None and err.device.type == "cuda")
        return host

    def _raise_if_aborted(self, err: int, what: str) -> None:
        if int(err):
            raise CadStepAborted(f"[{self.label}] {what}: a CAD flag wait timed out (abort word {int(err)}); "
                                 "the step is not valid")


__all__ = ["CadNodeRuntime", "CadRuntimeError", "CadStepAborted", "MODULE_LOADING_ENV", "TorchCadBackend",
           "check_module_loading"]
