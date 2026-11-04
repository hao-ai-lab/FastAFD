"""
Shared AFD Ray worker runtime.

BaseAfdWorker owns the common per-GPU runtime used by AFD attention/model actors.
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import torch
from minisgl.distributed import (
    DistributedInfo,
    set_ep_info,
    set_tp_info,
    try_get_ep_info,
    try_get_tp_info,
)
from minisgl.env import afd_env
from minisgl.message import BaseTokenizerMsg
from minisgl.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    div_even,
    init_logger,
    init_nvtx_cpu_trace,
    nvtx_label,
    nvtx_range,
    shutdown_nvtx_cpu_trace,
)

from .afd_protocol import (
    AfdCommand,
    AfdProfilerCmd,
    AfdReply,
    AfdTopology,
    AfdWorkerReadyReply,
)
from .afd_support import AfdRuntimeConfig, flush_log_lines, log_line

__all__ = ["BaseAfdWorker"]

# The M2N collective's cross-launch barrier chain, traced per step on both halves
# of the union. Off by default; see BaseAfdWorker.log_m2n_barrier_head.
_M2N_BARRIER_TRACE = os.environ.get("MINISGL_AFD_M2N_BARRIER_TRACE", "0") == "1"


def ensure_afd_parallel_info(config: Any) -> None:
    """Install TP/EP rank metadata for the current AFD worker process."""
    tp_info = try_get_tp_info()
    if tp_info is None:
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
    elif tp_info.rank != config.tp_info.rank or tp_info.size != config.tp_info.size:
        raise RuntimeError(
            f"TP info is already set to {tp_info.rank}/{tp_info.size}, "
            f"cannot reset to {config.tp_info.rank}/{config.tp_info.size}"
        )

    ep_size = int(getattr(config, "ep_size", 1))
    if ep_size <= 1:
        return
    per_replica_size = config.tp_info.size
    cross_replica_size = max(1, int(getattr(config, "dp_size", 1))) * config.tp_info.size
    if ep_size not in (per_replica_size, cross_replica_size):
        raise RuntimeError(
            "AFD MoE currently supports EP only as per-MLP-DP EP or full-world "
            "EP, so ep_size must be in "
            f"{{tp_size={per_replica_size}, dp_size*tp_size={cross_replica_size}}} "
            "when EP is enabled. Got "
            f"ep_size={ep_size} (dp_size={getattr(config, 'dp_size', 1)})"
        )
    if ep_size == cross_replica_size and ep_size != per_replica_size:
        ep_rank = int(getattr(config, "dp_rank", 0)) * config.tp_info.size + config.tp_info.rank
    else:
        ep_rank = config.tp_info.rank
    ep_info = try_get_ep_info()
    if ep_info is None:
        set_ep_info(rank=ep_rank, size=ep_size)
        return
    if ep_info.rank != ep_rank or ep_info.size != ep_size:
        raise RuntimeError(
            f"EP info is already set to {ep_info.rank}/{ep_info.size}, "
            f"cannot reset to {ep_rank}/{ep_size}"
        )



logger = init_logger(__name__)


class BaseAfdWorker:
    """Shared per-GPU worker runtime for AFD attention/model actors."""

    def __init__(
        self,
        *,
        role: Literal["attention", "mlp"],
        tp_rank: int,
        tp_size: int,
        comm_rank: int,
        node_ip: str,
        distributed_host: str,
        distributed_port: int,
        model_path: str,
        attention_backend: str,
        max_seq_len: int,
        max_comm_tokens: int,
        max_extend_tokens: int | None,
        max_running_req: int,
        num_page_override: int | None,
        cuda_graph_max_bs: int,
        decode_graph_bs: tuple[int, ...],
        enable_attention_decode_graph: bool,
        enable_model_decode_graph: bool,
        page_size: int,
        cache_type: str,
        log_dir: str,
        use_dummy_weight: bool,
        device_comm_num_sms: int,
        ep_size: int = 1,
        moe_a2a_backend: str = "none",
        moe_runner_backend: str = "auto",
        ray_nsys: bool = False,
        disable_overlap: bool = False,
        async_pending_steps: int = 3,
        afd_num_mb: int = 1,
        rpc_timeout_ms: int = 300_000,
        attn_dp_rank: int = 0,
        attn_dp_size: int = 1,
        mlp_dp_rank: int = 0,
        mlp_dp_size: int = 1,
        attn_tp_size: int | None = None,
        mlp_tp_size: int | None = None,
        memory_ratio: float = 0.8,
        distributed_rank: int | None = None,
        distributed_world_size: int | None = None,
        distributed_tp_groups: tuple[tuple[int, ...], ...] | None = None,
    ):
        import ray.util
        self.role = role
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.attn_dp_rank = int(attn_dp_rank)
        self.attn_dp_size = max(1, int(attn_dp_size))
        self.mlp_dp_rank = int(mlp_dp_rank)
        self.mlp_dp_size = max(1, int(mlp_dp_size))
        self.attn_tp_size = int(attn_tp_size if attn_tp_size is not None else tp_size)
        self.mlp_tp_size = int(mlp_tp_size if mlp_tp_size is not None else tp_size)
        self._topology = AfdTopology(
            attn_dp_size=self.attn_dp_size,
            mlp_dp_size=self.mlp_dp_size,
            attn_tp_size=self.attn_tp_size,
            mlp_tp_size=self.mlp_tp_size,
            ep_size=int(ep_size),
        )
        self.comm_rank = comm_rank
        self.worker_rank = comm_rank
        self.node_ip = ray.util.get_node_ip_address()
        assert self.node_ip == node_ip, (self.node_ip, node_ip)
        if self.role == "mlp" and self.mlp_dp_size > 1:
            self.log_path = str(Path(log_dir) / f"{role}_dp{self.mlp_dp_rank}_rank{tp_rank}.log")
        elif self.attn_dp_size > 1:
            self.log_path = str(Path(log_dir) / f"{role}_dp{self.attn_dp_rank}_rank{tp_rank}.log")
        else:
            self.log_path = str(Path(log_dir) / f"{role}_rank{tp_rank}.log")
        ray_gpu_ids = tuple(ray.get_gpu_ids())
        initial_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        use_megamoe_device_binding = afd_env("MOE_BACKEND", "deepep") == "megamoe_m2n"
        device_index = 0
        if use_megamoe_device_binding:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "symmetric-memory device binding requires CUDA_VISIBLE_DEVICES to be "
                    "cleared before CUDA is initialized"
                )
            if ray_gpu_ids:
                device_index = int(ray_gpu_ids[0])
            # torch symmetric memory exchanges CUDA device ordinals.  If Ray
            # remaps each actor to a single visible cuda:0 device, same-node
            # ranks look overlapping even when they are on different physical
            # GPUs.  Match the standalone MegaMoE harness: expose the full node
            # and bind this process to Ray's assigned physical GPU.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["MINISGL_DEVICE_INDEX"] = str(device_index)
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] cuda_binding "
            f"backend={afd_env('MOE_BACKEND', 'deepep')} "
            f"initial_cvd={initial_cvd!r} effective_cvd={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
            f"ray_gpu_ids={ray_gpu_ids} device={self.device}",
            flush=True,
        )
        self.dtype = torch.bfloat16
        self.attention_backend = str(attention_backend)
        self.page_size = int(page_size)
        self.max_seq_len = int(max_seq_len)
        self.max_comm_tokens = int(max_comm_tokens)
        self.max_running_req = int(max_running_req)
        self.cuda_graph_max_bs = int(cuda_graph_max_bs)
        self.decode_graph_bs = tuple(sorted(int(b) for b in decode_graph_bs))
        self.enable_attention_decode_graph = bool(enable_attention_decode_graph)
        self.enable_model_decode_graph = bool(enable_model_decode_graph)
        # AG and EG decode graphs contain matching collective calls, so capture and
        # replay must be enabled on both sides or disabled on both sides.
        self.enable_decode_graph = (
            self.enable_attention_decode_graph and self.enable_model_decode_graph
        )
        self.device_comm_num_sms = max(1, int(device_comm_num_sms))
        self.mlp_ep_size = int(ep_size)
        self.ep_size = self.mlp_ep_size if role == "mlp" else 1
        self.disable_overlap = bool(disable_overlap)
        self.async_pending_steps = max(2, int(async_pending_steps))
        self.async_retire_depth = self.async_pending_steps - 1
        self.afd_num_mb = max(1, int(afd_num_mb))
        self._control_profile_start_step = int(
            os.environ.get("MINISGL_AFD_CONTROL_PROFILE_START_STEP", "0") or 0
        )
        self._control_profile_stop_step = int(
            os.environ.get("MINISGL_AFD_CONTROL_PROFILE_STOP_STEP", "0") or 0
        )
        self._recv_timeout_ms = max(1_000, int(rpc_timeout_ms))
        self._dual_stream_microbatch = self.afd_num_mb > 1
        self._comm_stream = torch.cuda.Stream(device=self.device)
        self._afd_comm_streams = [
            torch.cuda.Stream(device=self.device) for _ in range(max(1, self.afd_num_mb))
        ]
        # Dedicated D2H stream for the one-step-ahead (overlap) AG/EG loops: the
        # sampled next_tokens copy runs here async, its copy_done_event is
        # synchronized one step LATER (mirrors the one-step-ahead EG worker), so the
        # compute stream never stalls on a gpu->cpu sync between steps.
        self._d2h_stream = torch.cuda.Stream(device=self.device)
        self._comm_ready = False
        self._recv_from_coordinator: ZmqPullQueue[AfdCommand] | None = None
        self._send_to_coordinator: ZmqPushQueue[AfdReply] | None = None
        self._send_into_detokenizer: ZmqPushQueue[BaseTokenizerMsg] | None = None
        self._hot_loop_running = False
        self._ray_nsys_enabled = bool(ray_nsys)
        self._nsys_runtime_started = False
        if self.role == "mlp" and self.mlp_dp_size > 1:
            trace_label = f"{role}_dp{self.mlp_dp_rank}_rank{tp_rank}"
        elif self.attn_dp_size > 1:
            trace_label = f"{role}_dp{self.attn_dp_rank}_rank{tp_rank}"
        else:
            trace_label = f"{role}_rank{tp_rank}"
        self._cpu_trace_path = init_nvtx_cpu_trace(
            output_dir=log_dir,
            process_label=trace_label,
            enabled=bool(ray_nsys),
            metadata={
                "kind": "afd-worker",
                "role": role,
                "attn_dp_rank": self.attn_dp_rank,
                "attn_dp_size": self.attn_dp_size,
                "mlp_dp_rank": self.mlp_dp_rank,
                "mlp_dp_size": self.mlp_dp_size,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "comm_rank": comm_rank,
                "node_ip": node_ip,
                "ray_nsys": bool(ray_nsys),
            },
        )
        if self._cpu_trace_path is not None:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] cpu_trace enabled path={self._cpu_trace_path}",
                flush=True,
            )

        # The AFD workers' process group (gloo barriers around warmup, where an EG
        # rank may still be JIT-compiling) takes the server's --distributed-timeout
        # through MINISGL_AFD_DIST_TIMEOUT; the config default is 300 s.
        dist_timeout_env = os.environ.get("MINISGL_AFD_DIST_TIMEOUT")
        extra_config = {"distributed_timeout": float(dist_timeout_env)} if dist_timeout_env else {}
        config = AfdRuntimeConfig(
            **extra_config,
            model_path=model_path,
            tp_info=DistributedInfo(tp_rank, tp_size),
            dtype=torch.bfloat16,
            max_running_req=max_running_req,
            attention_backend=attention_backend,
            moe_backend="fused",
            cuda_graph_max_bs=self.cuda_graph_max_bs,
            page_size=self.page_size,
            memory_ratio=float(memory_ratio),
            use_dummy_weight=use_dummy_weight,
            max_extend_tokens=(
                int(max_extend_tokens)
                if max_extend_tokens is not None
                else AfdRuntimeConfig.max_extend_tokens
            ),
            max_seq_len_override=max_seq_len,
            num_page_override=num_page_override,
            ep_size=self.ep_size,
            dp_size=(self.mlp_dp_size if role == "mlp" else self.attn_dp_size),
            dp_rank=(self.mlp_dp_rank if role == "mlp" else self.attn_dp_rank),
            cache_type=cache_type,
            offline_mode=True,
            distributed_host=distributed_host,
            distributed_port=distributed_port,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            distributed_tp_groups=distributed_tp_groups,
            ray_nsys=bool(ray_nsys),
            afd_num_mb=self.afd_num_mb,
            afd_attn_dp_size=self.attn_dp_size,
            afd_mlp_dp_size=self.mlp_dp_size,
            afd_attn_tp_size=self.attn_tp_size,
            afd_mlp_tp_size=self.mlp_tp_size,
            afd_moe_a2a_backend=str(moe_a2a_backend),
            afd_moe_runner_backend=str(moe_runner_backend),
        )
        self.runtime = self._create_runtime(config)
        self.runtime_state = getattr(self.runtime, "state", self.runtime)
        try:
            ctx = self.runtime_state.ctx
            ctx.server_args = config
        except AttributeError:
            pass

        model_config = config.model_config
        self.head_dim = model_config.head_dim
        # Local view: heads/dims this worker's own TP rank holds (attn or mlp).
        self.local_qo_heads = div_even(model_config.num_qo_heads, tp_size)
        self.local_qo_dim = self.local_qo_heads * self.head_dim
        self.local_kv_dim = (
            div_even(model_config.num_kv_heads, tp_size, allow_replicate=True) * self.head_dim
        )
        self.num_layers = int(model_config.num_layers)
        self.hidden_size = int(model_config.hidden_size)
        self.vocab_size = int(model_config.vocab_size)
        self.num_experts = int(getattr(model_config, "num_experts", 0) or 0)
        self.top_k = int(getattr(model_config, "num_experts_per_tok", 0) or 0)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] runtime_ready node_ip={self.node_ip} "
            f"comm_rank={self.comm_rank} "
            f"attn_dp_rank={self.attn_dp_rank} attn_dp_size={self.attn_dp_size} "
            f"mlp_dp_rank={self.mlp_dp_rank} mlp_dp_size={self.mlp_dp_size} "
            f"attn_tp_size={self.attn_tp_size} mlp_tp_size={self.mlp_tp_size} "
            f"device_comm_num_sms={self.device_comm_num_sms} "
            f"afd_num_mb={self.afd_num_mb} "
            f"disable_overlap={self.disable_overlap} "
            f"async_pending_steps={self.async_pending_steps} "
            f"async_retire_depth={self.async_retire_depth} "
            f"cuda_device_max_connections={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '<unset>')} "
            f"dual_stream_microbatch={self._dual_stream_microbatch} "
            f"rpc_timeout_ms={self._recv_timeout_ms} "
            f"decode_graph_bs={list(self.decode_graph_bs)} "
            f"attention_decode_graph={self.enable_attention_decode_graph} "
            f"model_decode_graph={self.enable_model_decode_graph}",
            flush=True,
        )

    def _create_runtime(self, config: AfdRuntimeConfig):
        raise NotImplementedError

    def _control_profile_enabled(self, step_id: int) -> bool:
        start = int(self._control_profile_start_step)
        stop = int(self._control_profile_stop_step)
        if start <= 0 and stop <= 0:
            return False
        step_id = int(step_id)
        return (start <= 0 or step_id >= start) and (stop <= 0 or step_id <= stop)

    def log_m2n_barrier_head(self, *, step_id: int, phase: object, when: str) -> None:
        """Trace the M2N cross-launch barrier chain. Off unless
        MINISGL_AFD_M2N_BARRIER_TRACE=1.

        Every rank of the AG+EG union must run the SAME number of barrier rounds
        on a given symmetric buffer, because the only agreement protocol is
        `counter & 3` (`deep_gemm/comm/barrier.cuh:53-55`) and the words are
        seeded once at construction and never reset. An odd slip, or a slip of
        exactly 2, hits the 300 s timeout; a slip of a multiple of 4 passes
        SILENTLY on a stale round and corrupts values instead. So the counters
        have to be compared across roles to see that class of defect at all.

        A step boundary is the only legal place for this: the read syncs, and a
        sync inside the collective is illegal under graph capture and lethal on
        the hot path. Both guards below are load-bearing.
        """
        if not _M2N_BARRIER_TRACE:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        adapters = getattr(self, "_afd_adapters", None)
        if not adapters:
            return
        adapter = adapters[0]
        if getattr(adapter, "backend", "") != "megamoe_m2n":
            return
        torch.cuda.synchronize(self.device)
        heads = []
        for lane in range(int(getattr(adapter, "num_lanes", 1))):
            counter, signal0, signal1 = adapter.barrier_head(lane)
            heads.append(f"lane{lane}=({counter},{signal0},{signal1})")
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] m2n_barrier {when} "
            f"step_id={int(step_id)} phase={phase} {' '.join(heads)}",
            flush=True,
        )

    @staticmethod
    def _deepep_do_cpu_sync(phase: object) -> bool:
        if os.environ.get("MINISGL_AFD_DISABLE_DEEPEP_CPU_SYNC", "0") == "1":
            return False
        return str(phase) != "decode"

    def _run_hot_rpc_loop(self) -> None:
        raise NotImplementedError

    def _prepare_hot_rpc_runtime(self) -> None:
        # nsys-safe JIT: child cuobjdump/nvcc must not inherit nsys CUDA
        # injection before DeepEP/DeepGEMM JIT compilation starts.
        self._detach_nsys_injection_from_subprocesses()
        self._init_afd_runtime()
        self._warmup_ag_dense_gemms()

        # Cold page cache can make EG stage loading minutes slower than AG
        # loading. Do not let an early AG rank enter DeepEP graph warmup while
        # another union rank is still loading weights: DeepEP's collective
        # timeout is shorter than the distributed startup timeout.
        if (
            self.enable_decode_graph
            and self.decode_graph_bs
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] "
                "afd_graph_warmup_barrier:start",
                flush=True,
            )
            torch.distributed.barrier()
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] "
                "afd_graph_warmup_barrier:done",
                flush=True,
            )

        # Runtime/model initialization can leave large, releasable blocks in the
        # CUDA caching allocator. Graph capture needs physical headroom for its
        # private pools, so mirror the ordinary engine graph lifecycle here.
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved(self.device)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd_graph_capture_memory_prepared "
            f"allocated_gib={allocated / (1024**3):.2f} "
            f"reserved_before_gib={reserved_before / (1024**3):.2f} "
            f"reserved_after_gib={reserved_after / (1024**3):.2f} "
            f"released_gib={(reserved_before - reserved_after) / (1024**3):.2f}",
            flush=True,
        )

        # AG and EG graph capture contains matching collectives, so all ranks
        # self-trigger warmup here before the coordinator starts sending steps.
        self.warmup_afd_decode_graphs(self.decode_graph_bs)
        self._send_hot_loop_ready()
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:ready_after_afd_init",
            flush=True,
        )

    def _recv_hot_command(self, *, blocking: bool) -> AfdCommand | None:
        queue = self._recv_from_coordinator
        if queue is None:
            raise RuntimeError("Hot RPC loop is not initialized")
        recv_start_ns = time.perf_counter_ns()
        if blocking:
            try:
                cmd = queue.get(timeout_ms=self._recv_timeout_ms)
            except TimeoutError:
                return None
        else:
            # true non-blocking (queue.get() with no timeout blocks on socket.recv)
            if queue.empty():
                return None
            cmd = queue.get()
        recv_done_ns = time.perf_counter_ns()
        self._log_plan_recv(
            cmd,
            blocking=blocking,
            recv_wait_ms=(recv_done_ns - recv_start_ns) / 1_000_000.0,
        )
        return cmd

    def _log_plan_recv(
        self,
        cmd: AfdCommand,
        *,
        blocking: bool,
        recv_wait_ms: float,
    ) -> None:
        pass

    def _ensure_comm_ready(self) -> None:
        if self._comm_ready:
            return
        self._comm_ready = True
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd runtime ready",
            flush=True,
        )

    def prepare_communication_runtime(self) -> dict[str, Any]:
        self._ensure_comm_ready()
        return {"role": self.role, "rank": self.tp_rank, "ready": self._comm_ready}

    def get_runtime_info(self) -> dict[str, Any]:
        info = {
            "role": self.role,
            "rank": self.tp_rank,
            "num_layers": int(self.num_layers),
        }
        if self.role == "attention":
            info["num_pages"] = int(self.runtime_state.num_pages)
            info["max_seq_len"] = int(self.runtime_state.max_seq_len)
        return info

    def _warmup_ag_dense_gemms(self) -> None:
        """Compile the attention stage's FP8 dense GEMM kernels for every prefill size
        before serving. MINISGL_DG_GEMM_WARMUP: "0" off; "1" or "auto" (default) on.

        A prefill step's dense GEMMs run on that step's token count, and a prompt's last
        chunk makes it arbitrary. DeepGEMM compiles a new kernel config on its first use
        (~2 s, holding the GIL) and every other attention rank waits for that rank in
        lockstep: on Qwen3-235B A4:F4 with random 512-8K prompts, 7 prefill steps took
        2-6 s each (15 of 32 s of prefill). Compiled kernels persist in the JIT cache, so
        later starts only replay the sweep (minisgl.layers.linear.warmup_fp8_dense_gemms).
        """
        model = getattr(self, "_afd_ag_model", None)
        if model is None or os.environ.get("MINISGL_DG_GEMM_WARMUP", "auto") == "0":
            return
        from minisgl.layers.linear import warmup_fp8_dense_gemms

        start = time.perf_counter()
        with torch.cuda.stream(self.runtime_state.stream):
            shapes, calls = warmup_fp8_dense_gemms(model, int(self.runtime.config.max_extend_tokens))
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] fp8_dense_gemm_warmup shapes={shapes} calls={calls} "
            f"seconds={time.perf_counter() - start:.1f}",
            flush=True,
        )

    def _init_afd_runtime(self) -> None:
        """afd AG/EG: build the role model + role-split weights + the directional
        DeepEP adapter over the union world. Collective (all union ranks build the
        adapter together); idempotent; called once at hot-loop start."""
        if getattr(self, "_afd_initialized", False):
            return
        import torch.distributed as dist
        from minisgl.moe.deepep_m2n_adapter import DeepEPM2NAdapter

        runtime_state = self.runtime_state
        config = runtime_state.config
        mc = config.model_config
        ag_size = self._topology.total_attn_workers
        eg_size = self._topology.total_mlp_workers
        dp_fanin = self._topology.attn_fanin_per_mlp_dp
        bucket = self._afd_dispatch_bucket()

        # Directional dispatch/combine over the union world (all AG+EG ranks).
        # Pipeline scheduling intentionally keeps multiple dispatch handles alive:
        # D(L,MB0), D(L,MB1), C(L,MB0), D(L+1,MB0), ...
        # A DeepEP ElasticBuffer owns the metadata/workspace that combine reads
        # through the dispatch handle, so each in-flight MB gets one adapter lane.
        adapter_lanes = max(1, int(self.afd_num_mb))
        self._afd_moe_backend = afd_env("MOE_BACKEND", "deepep")
        if self._afd_moe_backend == "megamoe_m2n":
            os.environ.setdefault("MINISGL_MEGAMOE_AG_SMS", "24")
            from minisgl.moe.megamoe_m2n_afd import MegaMoEM2NAfdAdapter

            mega_kwargs, remote_shared_experts = self._megamoe_m2n_adapter_kwargs(
                ag_size=ag_size, eg_size=eg_size, bucket=bucket, num_lanes=adapter_lanes
            )
            mega_adapter = MegaMoEM2NAfdAdapter(**mega_kwargs)
            # Lane view keeps the downstream `adapters[mb % len(adapters)]`
            # indexing working; every lane shares the adapter object and picks
            # its own symmetric buffer via the `lane` argument.
            self._afd_adapters = [mega_adapter] * adapter_lanes
            self._afd_adapter = mega_adapter
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_megamoe_m2n_adapter "
                f"lanes={adapter_lanes} afd_num_mb={int(self.afd_num_mb)} bucket={bucket} "
                f"remote_shared={int(remote_shared_experts)} "
                f"eg_expected_tokens_override={mega_adapter.eg_expected_tokens_override} "
                f"eg_expected_tokens_source={mega_adapter.eg_expected_tokens_source}",
                flush=True,
            )
        else:
            if adapter_lanes > 1:
                os.environ["MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM"] = "1"
            # Debug: MINISGL_AFD_LOG_M2N_ALLOC=1 logs real_recv_tokens for each new
            # eager dispatch shape (prefill only; decode graph replays are not logged).
            m2n_alloc_log = (
                (lambda msg: log_line(self.log_path, f"[{self.role} rank={self.tp_rank}] {msg}", flush=True))
                if afd_env("LOG_M2N_ALLOC", "0") == "1"
                else None
            )
            self._afd_adapters = [
                DeepEPM2NAdapter(
                    group=dist.group.WORLD,
                    ag_size=ag_size,
                    eg_size=eg_size,
                    real_num_experts=int(mc.num_experts),
                    hidden_size=int(mc.hidden_size),
                    top_k=int(mc.num_experts_per_tok),
                    num_max_dispatch_tokens_per_rank=bucket,
                    log=m2n_alloc_log,
                )
                for _lane in range(adapter_lanes)
            ]
            self._afd_adapter = self._afd_adapters[0]
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_deepep_adapters "
                f"lanes={adapter_lanes} afd_num_mb={int(self.afd_num_mb)} bucket={bucket} "
                f"hidden={self._afd_adapter.hidden_size} "
                f"per_buffer_comm_stream={os.environ.get('MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM', '0')}",
                flush=True,
            )

        if self.role == "attention":
            self._load_afd_ag_stage()
        else:
            self._load_afd_eg_stage(self._afd_adapter)

        # CUDA-graph decode runners (lockstep AG/EG). Built here; captured later
        # by warmup_afd_decode_graphs() once the coordinator drives all ranks
        # together (the per-layer dispatch/combine are collectives).
        self._afd_ag_graph = None
        self._afd_eg_graph = None
        if self.enable_decode_graph and self.decode_graph_bs:
            comm_stream = getattr(self, "_comm_stream", None)
            lane_comm_streams = tuple(getattr(self, "_afd_comm_streams", ()))
            spec_ag = getattr(self, "_afd_spec", None)
            if self.role == "attention" and spec_ag is not None:
                # Speculative decoding: decode steps are verify steps, so the AG
                # captures verify graphs (bs request slots of a block each).
                from .spec.graph import AfdAgVerifyGraphRunner

                self._afd_ag_graph = AfdAgVerifyGraphRunner(
                    runtime_state=runtime_state,
                    model=self._afd_ag_model,
                    adapters=self._afd_adapters,
                    num_mb=self.afd_num_mb,
                    comm_stream=comm_stream,
                    lane_comm_streams=lane_comm_streams,
                    runtime=self.runtime,
                    spec=spec_ag,
                    num_layers=int(mc.num_layers),
                    hidden_size=int(mc.hidden_size),
                    bs_list=tuple(self.decode_graph_bs),
                    log_path=self.log_path,
                    label=f"{self.role} rank={self.tp_rank}",
                )
            elif self.role == "attention":
                from .afd_attention_worker import AfdAgDecodeGraphRunner

                self._afd_ag_graph = AfdAgDecodeGraphRunner(
                    runtime_state=runtime_state,
                    model=self._afd_ag_model,
                    adapters=self._afd_adapters,
                    num_mb=self.afd_num_mb,
                    comm_stream=comm_stream,
                    lane_comm_streams=lane_comm_streams,
                    runtime=self.runtime,
                    num_layers=int(mc.num_layers),
                    hidden_size=int(mc.hidden_size),
                    bs_list=tuple(self.decode_graph_bs),
                    log_path=self.log_path,
                    label=f"{self.role} rank={self.tp_rank}",
                )
            else:
                from .afd_expert_worker import AfdEgDecodeGraphRunner

                self._afd_eg_graph = AfdEgDecodeGraphRunner(
                    runtime_state=runtime_state,
                    expert_stage=self._afd_eg_model,
                    adapters=self._afd_adapters,
                    num_mb=self.afd_num_mb,
                    comm_stream=comm_stream,
                    lane_comm_streams=lane_comm_streams,
                    num_layers=int(self.num_layers),
                    hidden_size=int(mc.hidden_size),
                    bs_list=tuple(b * self._spec_tokens_per_slot() for b in self.decode_graph_bs),
                    log_path=self.log_path,
                    label=f"{self.role} rank={self.tp_rank}",
                )

        self._afd_initialized = True
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd_runtime:init_done "
            f"ag={ag_size} eg={eg_size} bucket={bucket} "
            f"raw_max_comm_tokens={int(getattr(self, 'max_comm_tokens', 0))} "
            f"dp_fanin={dp_fanin}",
            flush=True,
        )

    def _afd_dispatch_bucket(self) -> int:
        """Tokens per AG source rank the union adapter sizes for. self.max_comm_tokens is
        sized conservatively for fan-in topologies, and DeepEP's
        num_max_dispatch_tokens_per_rank is per AG source rank, so undo that expansion."""
        dp_fanin = self._topology.attn_fanin_per_mlp_dp
        afd_max_dispatch_tokens = max(1, int(getattr(self, "max_comm_tokens", 0)) // dp_fanin)
        return max(int(self.cuda_graph_max_bs), afd_max_dispatch_tokens)

    def _afd_cast_full(self, pairs) -> dict[str, torch.Tensor]:
        # Keep fp8 weights and *_scale (float32) as-is. Router gates/biases
        # that need FP32 are listed below; everything else follows model dtype.
        """The loader's (name, tensor) pairs in the dtypes the AFD stages take: the
        model dtype, fp32 for scales and the per-model fp32 keys, quantized as is."""
        config = self.runtime_state.config
        mc = config.model_config
        arch0 = getattr(mc, "architectures", [""])[0]

        is_minimax_m2 = arch0 == "MiniMaxM2ForCausalLM"
        is_glm4_moe = arch0 == "Glm4MoeForCausalLM"
        is_qwen35 = arch0 == "Qwen3_5MoeForConditionalGeneration"

        def _minimax_fp32_key(key: str) -> bool:
            return is_minimax_m2 and (
                key.endswith(".mlp.gate.weight")
                or key.endswith(".mlp.e_score_correction_bias")
            )

        def _glm4_fp32_key(key: str) -> bool:
            return is_glm4_moe and (
                key.endswith(".mlp.gate.weight")
                or key.endswith(".mlp.gate.e_score_correction_bias")
            )

        def _qwen35_fp32_key(key: str) -> bool:
            return is_qwen35 and key.endswith(".linear_attn.A_log")

        def _cast_full(pairs):
            tgt = config.dtype
            return {
                k: (
                    v
                    if v.dtype in (torch.uint8, torch.float8_e4m3fn)
                    else v.to(torch.float32)
                    if k.endswith("_scale")
                    or _minimax_fp32_key(k)
                    or _glm4_fp32_key(k)
                    or _qwen35_fp32_key(k)
                    else v.to(tgt)
                )
                for k, v in pairs
            }

        return _cast_full(pairs)

    def _afd_stage_classes(self):
        """(attention stage, expert stage) classes of this architecture."""
        mc = self.runtime_state.config.model_config
        arch = (getattr(mc, "architectures", None) or [""])[0]
        if arch == "Qwen3MoeForCausalLM":
            from minisgl.models.qwen3_moe_afd import (
                Qwen3AfdDenseRouterForCausalLM,
                Qwen3AfdExpertStage,
            )

            return Qwen3AfdDenseRouterForCausalLM, Qwen3AfdExpertStage
        if arch == "Qwen3_5MoeForConditionalGeneration":
            from minisgl.models.qwen35_moe_afd import (
                Qwen3_5AfdDenseRouterForCausalLM,
                Qwen3_5AfdExpertStage,
            )

            return Qwen3_5AfdDenseRouterForCausalLM, Qwen3_5AfdExpertStage
        if arch == "MiniMaxM2ForCausalLM":
            from minisgl.models.minimax_m2_afd import (
                MiniMaxM2AfdDenseRouterForCausalLM,
                MiniMaxM2AfdExpertStage,
            )

            return MiniMaxM2AfdDenseRouterForCausalLM, MiniMaxM2AfdExpertStage
        if arch == "Glm4MoeForCausalLM":
            from minisgl.models.glm4_moe_afd import (
                Glm4AfdDenseRouterForCausalLM,
                Glm4AfdExpertStage,
            )

            return Glm4AfdDenseRouterForCausalLM, Glm4AfdExpertStage
        raise RuntimeError(f"AFD AG/EG does not support architecture {arch!r}")

    def _load_afd_ag_stage(self) -> None:
        """The attention stage: the non-expert weights and the sampler."""
        from minisgl.engine.sample import Sampler
        from minisgl.models import load_weight
        from minisgl.models.afd import extract_stage_state_dict
        from minisgl.utils import torch_dtype

        runtime_state = self.runtime_state
        config = runtime_state.config
        mc = config.model_config
        _cast_full = self._afd_cast_full
        AfdDenseRouterForCausalLM, _ = self._afd_stage_classes()
        from minisgl.layers import set_rope_device

        set_rope_device(runtime_state.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            model = AfdDenseRouterForCausalLM(mc)
        full = _cast_full(
            load_weight(config.model_path, runtime_state.device, skip_expert_weights=True)
        )
        model.load_state_dict(extract_stage_state_dict(model, full))
        del full
        self._afd_ag_model = model
        self._afd_sampler = Sampler(runtime_state.device, int(mc.vocab_size))
        from .spec.config import afd_spec_config

        spec = afd_spec_config()
        self._afd_spec = None
        if spec.enabled:
            from .spec.attn_side import AfdSpecAG

            self._afd_spec = AfdSpecAG(
                spec, model=model, runtime=self.runtime, device=runtime_state.device
            )
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_spec draft={spec.draft_path} "
                f"block={spec.block} sim_accept={spec.sim_accept} "
                f"capture_layers={sorted(self._afd_spec.capture_cols)} "
                f"cuda_alloc_gib={torch.cuda.memory_allocated() / (1024**3):.2f}",
                flush=True,
            )

    def _load_afd_eg_stage(self, adapter) -> None:
        """The expert stage on `adapter`: the experts of every layer."""
        from minisgl.models import load_weight
        from minisgl.models.afd import extract_stage_state_dict
        from minisgl.utils import torch_dtype

        runtime_state = self.runtime_state
        config = runtime_state.config
        mc = config.model_config
        _cast_full = self._afd_cast_full
        _, AfdExpertStage = self._afd_stage_classes()
        # The AFD EG stage receives already-dispatched DeepEPM2N payloads
        # from the outer adapter above. Its per-layer MoELayer objects are
        # only expert weight holders plus DeepGEMM runners; AFD EG model
        # classes explicitly disable MoELayer's internal A2A dispatcher.
        with torch.device("meta"), torch_dtype(config.dtype):
            model = AfdExpertStage(mc)
        pairs = load_weight(config.model_path, runtime_state.device, skip_non_expert_weights=True)
        full = _cast_full(pairs)
        model.load_state_dict(extract_stage_state_dict(model, full))
        del full
        remote_layers = [
            layer
            for i, layer in enumerate(getattr(model.model.layers, "op_list", []))
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
        ]
        try:
            sd = model.state_dict()
            tensor_bytes = sum(int(t.numel() * t.element_size()) for t in sd.values())
            num_state_tensors = len(sd)
            # drop the references now: MegaMoE registration below frees each layer's
            # original expert tensors as it goes, which a live state dict would defeat
            # (the peak would be every original plus every transformed weight)
            del sd
            shapes = []
            for idx in (0, len(remote_layers) - 1):
                if 0 <= idx < len(remote_layers):
                    experts = remote_layers[idx].mlp.experts
                    shapes.append(
                        "layer_index="
                        f"{idx} local_experts={int(getattr(experts, 'local_num_experts', -1))} "
                        f"gate_up={tuple(experts.gate_up_proj.shape)}:{experts.gate_up_proj.dtype} "
                        f"down={tuple(experts.down_proj.shape)}:{experts.down_proj.dtype}"
                    )
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_eg_model_memory "
                f"state_tensors={num_state_tensors} state_bytes={tensor_bytes} "
                f"state_gib={tensor_bytes / (1024**3):.2f} "
                f"cuda_alloc_gib={torch.cuda.memory_allocated() / (1024**3):.2f} "
                f"cuda_reserved_gib={torch.cuda.memory_reserved() / (1024**3):.2f} "
                + " ".join(shapes),
                flush=True,
            )
        except Exception as exc:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_eg_model_memory_log_failed "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        if self._afd_moe_backend == "megamoe_m2n":
            # One-time per-layer requant of FP8 expert weights to the mega
            # kernel's per-32 UE8M0 interleaved format.
            layers = model.model.layers.op_list
            registered_layers = 0
            for layer_id in range(len(layers)):
                eg_layer = layers[layer_id]
                experts = eg_layer.mlp.experts
                shared_experts = getattr(eg_layer.mlp, "shared_experts", None)
                adapter.register_eg_layer_weights(
                    layer_id, experts, shared_experts=shared_experts
                )
                registered_layers += 1
                # MegaMoE keeps its transformed weights in the adapter. Drop
                # the Qwen-format tensors immediately; retaining both forms
                # OOMs on the 235B expert ranks.
                experts.gate_up_proj = None
                experts.gate_up_proj_scale = None
                experts.down_proj = None
                experts.down_proj_scale = None
                if shared_experts is not None:
                    shared_experts.gate_up_proj.weight = None
                    shared_experts.gate_up_proj.weight_scale = None
                    shared_experts.down_proj.weight = None
                    shared_experts.down_proj.weight_scale = None
                if layer_id % 4 == 3:
                    torch.cuda.empty_cache()
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_megamoe_m2n_weights "
                f"registered layers={registered_layers}/{len(layers)}",
                flush=True,
            )
            # MegaMoE runtime owns the transformed weights in the adapter.
            # Drop the ExpertStage object too; the EG persistent and
            # per-layer MegaMoE paths never call run_experts().
            self._afd_eg_model = None
            del layers
            del model
            torch.cuda.empty_cache()
            self._precompile_megamoe_eg_kernels(adapter)
        else:
            self._afd_eg_model = model

    def _megamoe_m2n_adapter_kwargs(self, *, ag_size: int, eg_size: int, bucket: int,
                                    num_lanes: int) -> tuple[dict, bool]:
        """MegaMoEM2NAfdAdapter keyword arguments for this model (and whether the shared
        expert runs remotely)."""
        import torch.distributed as dist

        config = self.runtime_state.config
        mc = config.model_config
        arch0 = getattr(mc, "architectures", [""])[0]
        model_extra = getattr(mc, "model_extra", {})
        n_shared_experts = int(model_extra.get("n_shared_experts", 0) or 0)
        remote_shared_experts = arch0 == "Glm4MoeForCausalLM" and n_shared_experts > 0
        if remote_shared_experts and n_shared_experts != 1:
            raise RuntimeError(
                "megamoe_m2n remote shared expert currently supports exactly "
                f"one shared expert, got n_shared_experts={n_shared_experts}"
            )
        # Split MegaMoE M2N: one fused kernel per side per layer over the
        # union world; the adapter holds one symmetric buffer per lane.

        kwargs = dict(
            group=dist.group.WORLD,
            ag_size=ag_size,
            eg_size=eg_size,
            real_num_experts=int(mc.num_experts),
            hidden_size=int(mc.hidden_size),
            intermediate_size=int(mc.moe_intermediate_size),
            top_k=int(mc.num_experts_per_tok),
            num_max_dispatch_tokens_per_rank=bucket,
            num_lanes=num_lanes,
            gate_renormalize=bool(getattr(mc, "norm_topk_prob", True)),
            shared_experts_per_rank=1 if remote_shared_experts else 0,
            routed_scaling_factor=float(model_extra.get("routed_scaling_factor", 1.0)),
        )
        return kwargs, remote_shared_experts

    def _precompile_megamoe_eg_kernels(self, adapter=None) -> None:
        """Build every MegaMoE EG kernel instance before the first collective step.

        A first-use JIT compile inside a step stalls the EG host for seconds after
        the AG ranks have launched their half of the collective (see
        `MegaMoEM2NAfdAdapter.precompile_eg_kernels`). This runs on EG ranks only,
        right after weight registration: before the graph-warmup barrier and
        before this rank reports ready, so the AG ranks wait for it exactly as
        they already wait for the EG weights, and no collective step can start
        while it compiles.
        """
        from minisgl.moe.megamoe_m2n_afd import eg_precompile_enabled

        adapter = self._afd_adapter if adapter is None else adapter
        num_mb = int(self.afd_num_mb)
        if not eg_precompile_enabled():
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_megamoe_eg_precompile "
                "disabled by MINISGL_MEGAMOE_EG_PRECOMPILE=0: EG kernel instances "
                "compile on first use, inside a collective step",
                flush=True,
            )
            return
        started = time.perf_counter()
        instances, buckets = adapter.precompile_eg_kernels(num_mb)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd_megamoe_eg_precompile "
            f"instances={instances} buckets={buckets} num_mb={num_mb} "
            f"persistent={int(adapter.use_persistent(num_mb))} "
            f"elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )

    def warmup_afd_decode_graphs(self, bs_list) -> dict[str, Any]:
        """Capture the AG/EG decode graphs for each bucket. MUST be invoked on all
        AG+EG ranks concurrently (the coordinator ray.gets them together) so the
        per-bucket capture — which runs the DeepEP dispatch/combine COLLECTIVES —
        stays in lockstep across ranks. Buckets captured ascending, same order on
        every rank."""
        if not self.enable_decode_graph:
            return {"role": self.role, "rank": self.tp_rank, "captured": []}
        runner = self._afd_ag_graph if self.role == "attention" else self._afd_eg_graph
        if runner is None:
            return {"role": self.role, "rank": self.tp_rank, "captured": []}
        captured = []
        # Under speculative decoding a graph bucket of bs request slots carries
        # bs * block tokens to the experts.
        scale = self._spec_tokens_per_slot() if self.role != "attention" else 1
        # Speculative verify graphs are large; capturing the biggest bucket first lets
        # the smaller ones reuse its pool segments. Same order on AG and EG ranks.
        descending = self._spec_tokens_per_slot() > 1
        for bs in sorted((int(b) for b in bs_list), reverse=descending):
            runner.warmup(bs * scale)
            captured.append(bs * scale)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd_decode_graphs:captured {captured}",
            flush=True,
        )
        return {"role": self.role, "rank": self.tp_rank, "captured": captured}

    @staticmethod
    def _spec_tokens_per_slot() -> int:
        from .spec.config import afd_spec_config

        spec = afd_spec_config()
        return spec.block if spec.enabled else 1

    def _detach_nsys_injection_from_subprocesses(self) -> None:
        """Keep profiler injection in the worker and sanitize only build children.

        DeepEP/DeepGEMM builds pass sanitized envs through build_utils. Mutating
        the worker process env here breaks NVTX collection for all later ranges.
        """
        active = [
            key
            for key in (
            "CUDA_INJECTION64_PATH",
            "CUDA_INJECTION32_PATH",
            "NVTX_INJECTION64_PATH",
            "NVTX_INJECTION32_PATH",
            )
            if key in os.environ
        ]
        if active or os.environ.get("LD_PRELOAD"):
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] nsys_injection:preserved "
                f"keys={active} ld_preload_len={len(os.environ.get('LD_PRELOAD', ''))}",
                flush=True,
            )

    def start_cuda_profiler(self, sync: bool = True) -> dict[str, Any]:
        if not self._ray_nsys_enabled or self._nsys_runtime_started:
            return {"role": self.role, "rank": self.tp_rank, "started": self._nsys_runtime_started}
        if sync:
            torch.cuda.synchronize(self.device)
        torch.cuda.profiler.start()
        self._nsys_runtime_started = True
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] nsys profiler:start sync={int(bool(sync))}",
            flush=True,
        )
        return {"role": self.role, "rank": self.tp_rank, "started": True}

    def stop_cuda_profiler(self, sync: bool = True) -> dict[str, Any]:
        if not self._ray_nsys_enabled or not self._nsys_runtime_started:
            return {"role": self.role, "rank": self.tp_rank, "stopped": False}
        if sync:
            torch.cuda.synchronize(self.device)
        torch.cuda.profiler.stop()
        self._nsys_runtime_started = False
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] nsys profiler:stop sync={int(bool(sync))}",
            flush=True,
        )
        return {"role": self.role, "rank": self.tp_rank, "stopped": True}

    # ----- CAD decode (minisgl.cad, MINISGL_AFD_MODE=cad): the hot loops' dispatch of the two CAD commands; the role
    # classes implement _init_cad / _run_cad_step (AfdAttentionWorker here, minisgl.cad.model_worker.CadModelWorker)
    @staticmethod
    def _is_cad_cmd(cmd) -> bool:
        from .afd_protocol import AfdCadInitCmd, AfdRunCadStepCmd

        return isinstance(cmd, (AfdCadInitCmd, AfdRunCadStepCmd))

    def _handle_cad_cmd(self, cmd) -> bool:
        """Run a CAD command (True) or leave the command to the caller (False)."""
        from .afd_protocol import AfdCadInitCmd, AfdRunCadStepCmd

        if isinstance(cmd, AfdCadInitCmd):
            self._init_cad(cmd)
            return True
        if isinstance(cmd, AfdRunCadStepCmd):
            self._run_cad_step(cmd)
            return True
        return False

    def _init_cad(self, cmd) -> None:
        raise RuntimeError(f"{type(self).__name__} has no CAD role (MINISGL_AFD_MODE=cad: AfdAttentionWorker / CadModelWorker)")

    def _run_cad_step(self, cmd) -> None:
        raise RuntimeError(f"{type(self).__name__} has no CAD role (MINISGL_AFD_MODE=cad: AfdAttentionWorker / CadModelWorker)")

    def handle_profiler_cmd(self, cmd: AfdProfilerCmd) -> None:
        if cmd.action == "start":
            self.start_cuda_profiler(sync=bool(cmd.sync))
            return
        if cmd.action == "stop":
            self.stop_cuda_profiler(sync=bool(cmd.sync))
            return
        raise RuntimeError(f"Unsupported AFD profiler command action: {cmd.action!r}")

    def _send_reply(self, reply: AfdReply) -> None:
        queue = self._send_to_coordinator
        if queue is None:
            raise RuntimeError("Hot RPC loop is not initialized")
        step_id = None if reply.step_id is None else int(reply.step_id)
        fields = {
            "step": step_id,
            "kind": type(reply).__name__,
            "worker_rank": self.worker_rank,
            "tp_rank": self.tp_rank,
        }
        prefix = "AFD_Worker_SendReply"
        serialize_start_ns = time.perf_counter_ns()
        with nvtx_range(nvtx_label(f"{prefix}_Serialize", **fields)):
            raw = queue.serialize(reply)
        serialize_done_ns = time.perf_counter_ns()
        with nvtx_range(nvtx_label(f"{prefix}_Put", bytes=len(raw), **fields)):
            queue.put_raw(raw)
        put_done_ns = time.perf_counter_ns()
        if step_id is not None and self._control_profile_enabled(step_id):
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] reply_send kind={type(reply).__name__} "
                f"step_id={step_id} worker_rank={self.worker_rank} bytes={len(raw)} "
                f"serialize_ms={(serialize_done_ns - serialize_start_ns) / 1_000_000.0:.3f} "
                f"put_ms={(put_done_ns - serialize_done_ns) / 1_000_000.0:.3f} "
                f"total_ms={(put_done_ns - serialize_start_ns) / 1_000_000.0:.3f}",
            )

    def _send_hot_loop_ready(self) -> None:
        self._send_reply(
            AfdWorkerReadyReply(
                worker_rank=self.worker_rank,
                step_id=None,
            )
        )

    def start_hot_rpc_loop(
        self,
        cmd_addr: str,
        result_addr: str,
        detokenizer_addr: str | None = None,
        create_detokenizer_link: bool = False,
    ) -> dict[str, Any]:
        self._recv_from_coordinator = ZmqPullQueue(
            cmd_addr,
            create=False,
            decoder=AfdCommand.decoder,
            serde="pickle",
        )
        self._send_to_coordinator = ZmqPushQueue(
            result_addr,
            create=False,
            encoder=AfdReply.encoder,
            serde="pickle",
        )
        if detokenizer_addr is not None and self.role == "attention" and self.tp_rank == 0:
            self._send_into_detokenizer = ZmqPushQueue(
                detokenizer_addr,
                create=bool(create_detokenizer_link),
                encoder=BaseTokenizerMsg.encoder,
            )
        self._hot_loop_running = True
        log_line(self.log_path, f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:start", flush=True)
        try:
            self._run_hot_rpc_loop()
        except Exception:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:error\n{traceback.format_exc()}",
                flush=True,
            )
            raise
        finally:
            self._hot_loop_running = False
            self._log_cuda_memory_peak()
            # Finalize profiling here too: if the coordinator is killed before it
            # can call worker.shutdown() (observed with attn_dp>1 multi-node
            # teardown), this is the only place the per-rank cpu_trace + nsys-rep
            # get flushed.
            self._finalize_profiling()
            for attr in (
                "_recv_from_coordinator",
                "_send_to_coordinator",
                "_send_into_detokenizer",
            ):
                self._safe_close(attr)
            # The MegaMoE AG progress timers live in pinned host memory precisely
            # so they outlive a kernel trap, which makes shutdown the only place
            # they can still be read. Nothing is logged unless they were enabled.
            for adapter in getattr(self, "_afd_adapters", None) or []:
                report = getattr(adapter, "report_ag_timings", None)
                if report is None:
                    continue
                line = report()
                if line != "AG timings not enabled":
                    log_line(
                        self.log_path,
                        f"[{self.role} rank={self.tp_rank}] megamoe_ag_timings {line}",
                        flush=True,
                    )
            log_line(self.log_path, f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:stop", flush=True)
        return {"role": self.role, "rank": self.tp_rank, "stopped": True}

    def _finalize_profiling(self) -> None:
        # Flush the CPU NVTX trace + finalize the nsys capture. Idempotent so it
        # can run from either the hot-loop finally or shutdown(), whichever fires
        # first. Needed because the coordinator's graceful shutdown (which calls
        # worker.shutdown()) can be killed before it runs under attn_dp>1 /
        # multi-node teardown, leaving the per-rank cpu_trace + nsys-rep unwritten.
        if getattr(self, "_profiling_finalized", False):
            return
        self._profiling_finalized = True
        if self._cpu_trace_path is not None:
            log_line(self.log_path, f"[{self.role} rank={self.tp_rank}] cpu_trace flush:start", flush=True)
        try:
            trace_path = shutdown_nvtx_cpu_trace()
            if trace_path:
                log_line(
                    self.log_path,
                    f"[{self.role} rank={self.tp_rank}] cpu_trace flushed path={trace_path}",
                    flush=True,
                )
        except Exception:
            pass
        try:
            self.stop_cuda_profiler()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._log_cuda_memory_peak()
        self._finalize_profiling()
        for attr in (
            "_recv_from_coordinator",
            "_send_to_coordinator",
        ):
            self._safe_close(attr)
        self._safe_close("runtime", method="shutdown")
        self._destroy_afd_adapters()
        flush_log_lines(self.log_path)

    def _log_cuda_memory_peak(self) -> None:
        if getattr(self, "_cuda_memory_peak_logged", False):
            return
        self._cuda_memory_peak_logged = True
        try:
            current_allocated = int(torch.cuda.memory_allocated(self.device))
            current_reserved = int(torch.cuda.memory_reserved(self.device))
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
            _, total = torch.cuda.mem_get_info(self.device)
            gib = 1024**3
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_cuda_memory "
                f"current_allocated_bytes={current_allocated} "
                f"current_reserved_bytes={current_reserved} "
                f"peak_allocated_bytes={peak_allocated} "
                f"peak_reserved_bytes={peak_reserved} "
                f"device_total_bytes={int(total)} "
                f"current_allocated_gib={current_allocated / gib:.2f} "
                f"current_reserved_gib={current_reserved / gib:.2f} "
                f"peak_allocated_gib={peak_allocated / gib:.2f} "
                f"peak_reserved_gib={peak_reserved / gib:.2f} "
                f"device_total_gib={int(total) / gib:.2f}",
                flush=True,
            )
        except Exception as exc:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_cuda_memory_log_failed "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _destroy_afd_adapters(self) -> None:
        adapters = list(getattr(self, "_afd_adapters", []))
        self._afd_adapters = []
        self._afd_adapter = None
        for adapter in reversed(adapters):
            try:
                adapter.destroy()
            except Exception:
                pass

    def _safe_close(self, attr: str, *, method: str = "stop") -> None:
        obj = getattr(self, attr, None)
        if obj is None:
            return
        try:
            getattr(obj, method)()
        except Exception:
            pass
        setattr(self, attr, None)
