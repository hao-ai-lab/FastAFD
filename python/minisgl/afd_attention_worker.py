"""AFD attention worker actor."""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed
from minisgl.attention import create_attention_backend
from minisgl.attention.base import BaseAttnBackend, HybridBackend
from minisgl.attention.fi import FlashInferBackend
from minisgl.core import Batch, Context, Req, get_global_ctx, set_global_ctx
from minisgl.distributed import destroy_distributed
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config, _init_tp_communication
from minisgl.engine.graph import get_free_memory, mem_GB
from minisgl.kvcache import create_hybrid_gdn_state_pool, create_kvcache_pool
from minisgl.utils import div_even, init_logger, nvtx_label, nvtx_range

from .afd_attention_runtime import AfdAttentionRuntime
from .afd_protocol import (
    AfdAGStepPlan,
    AfdAGStepReply,
    AfdFlushStepCmd,
    AfdProfilerCmd,
    AfdRunAGStepCmd,
    AfdStopCmd,
)
from .afd_support import AfdRuntimeConfig, log_line
from .afd_worker_base import BaseAfdWorker, ensure_afd_parallel_info
from .cuda_graph_utils import capture_cuda_graph

logger = init_logger(__name__)

__all__ = ["AfdAttentionWorker", "AfdAttentionState", "AfdAgDecodeGraphRunner"]


# ---------------------------------------------------------------------------
# Attention-side CUDA process state
# ---------------------------------------------------------------------------


def _flashinfer_backend(backend: BaseAttnBackend) -> FlashInferBackend | None:
    if isinstance(backend, FlashInferBackend):
        return backend
    if isinstance(backend, HybridBackend) and isinstance(
        backend.decode_backend, FlashInferBackend
    ):
        return backend.decode_backend
    return None


class AfdAttentionState:
    """Attention-side CUDA state: device, TP comms, KV cache, and attn backends."""

    def __init__(self, config: EngineConfig):
        _adjust_config(config)
        ensure_afd_parallel_info(config)
        visible_device_count = torch.cuda.device_count()
        device_index = int(os.environ.get("MINISGL_DEVICE_INDEX", config.tp_info.rank))
        if device_index >= visible_device_count:
            device_index = 0
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.config = config
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)
        self.tp_cpu_group = _init_tp_communication(config, self.dtype)
        free_memory = self._sync_get_memory()[1]

        self.ctx.gdn_state_pool = create_hybrid_gdn_state_pool(
            config.model_config,
            max_slots=config.max_running_req + 1,
            dtype=self.dtype,
            device=self.device,
        )
        if self.ctx.gdn_state_pool is not None:
            logger.info_rank0(
                "AFD allocated hybrid GDN state: layers=%d slots=%d bytes_per_slot=%d",
                len(self.ctx.gdn_state_pool.layer_ids),
                self.ctx.gdn_state_pool.max_slots,
                self.ctx.gdn_state_pool.bytes_per_slot,
            )
        self.num_pages = self._determine_num_pages(free_memory, config)
        self.max_seq_len = min(config.max_seq_len, self.num_pages * config.page_size)
        aligned_max_seq_len = ((self.max_seq_len + 31) // 32) * 32
        self.ctx.page_table = self.page_table = torch.zeros(
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        self.ctx.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )
        # One attention backend per microbatch because each wrapper owns mutable
        # decode metadata. The compute stream still serializes actual launches.
        num_mb = max(1, int(getattr(config, "afd_num_mb", 1)))
        self.attn_backends: list = [
            create_attention_backend(config.attention_backend, config.model_config)
            for _ in range(num_mb)
        ]
        # FlashInfer stores pinned plan-copy state per wrapper. Sharing the last
        # event keeps H2D source lifetimes obvious without changing launch order.
        if len(self.attn_backends) > 1:
            first_fi_backend = _flashinfer_backend(self.attn_backends[0])
            if first_fi_backend is not None:
                shared_event = first_fi_backend.last_event
                for backend in self.attn_backends[1:]:
                    fi_backend = _flashinfer_backend(backend)
                    if fi_backend is not None:
                        fi_backend.last_event = shared_event
        self.ctx.attn_backend = self.attn_backends[0]

    def _sync_get_memory(self) -> tuple[int, int]:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor(
            [free_memory, -free_memory], device="cpu", dtype=torch.int64
        )
        torch.distributed.all_reduce(
            free_mem_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_cpu_group,
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        return min_free_memory, max_free_memory

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_full_attention_layers
        )
        state_memory = old_free_memory - new_free_memory
        available_memory = int(config.memory_ratio * old_free_memory) - state_memory
        memory_cap = available_memory // cache_per_page
        requested = config.num_page_override
        if requested is None:
            num_pages = memory_cap
        else:
            num_pages = min(requested, memory_cap)
            if requested > memory_cap:
                logger.warning(
                    "AfdAttentionState: num_page_override=%d clamped to memory cap=%d "
                    "(requested KV would need %s, only %s available at memory_ratio=%.2f).",
                    requested,
                    memory_cap,
                    mem_GB(requested * cache_per_page),
                    mem_GB(available_memory),
                    config.memory_ratio,
                )
        assert num_pages > 1, "Not enough memory for attention-side KV cache"
        logger.info(
            "AfdAttentionState allocating %d pages (%s; memory_ratio=%.2f, cap=%d)",
            num_pages,
            mem_GB(num_pages * cache_per_page),
            config.memory_ratio,
            memory_cap,
        )
        return num_pages

    def install_dummy_req(self, table_idx: int, num_tokens: int) -> Req:
        dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=table_idx,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
        )
        self.page_table[dummy_req.table_idx].fill_(num_tokens)
        return dummy_req

    def shutdown(self) -> None:
        torch.distributed.destroy_process_group()
        destroy_distributed()


# ---------------------------------------------------------------------------
# AG forward helpers and decode-graph runner
# ---------------------------------------------------------------------------


def _fp8_dispatch_quant(hidden: torch.Tensor):
    from minisgl.kernel.fp8_quant import per_token_cast_to_fp8

    return per_token_cast_to_fp8(
        hidden.contiguous(), use_ue8m0=True, gran_k=128, use_packed_ue8m0=True, backend="cuda"
    )


# AG side has no experts, so its combine "expert output" contribution must be the
# zero/identity element (combine ignores its actual content — coercing the fp8 recv
# via .to(bf16) was a ~70us/layer copy that did nothing useful). Use a per-shape
# cached zeros bf16 buffer instead: never written, so in a captured graph it's just
# a read (zero per-step cost), and reused across eager calls too. Module-level so
# the buffers stay alive for the lifetime of any graph that captured them.
_AG_COMBINE_ZEROS: dict[tuple, torch.Tensor] = {}


def ag_combine_input(recv: torch.Tensor) -> torch.Tensor:
    """Zero bf16 combine input matching the AG dispatch recv shape (combine needs
    bf16; the AG contributes nothing). Replaces disp.hidden_states.to(bf16)."""
    dev = recv.device.index if recv.device.index is not None else -1
    key = (int(recv.shape[0]), int(recv.shape[1]), dev)
    buf = _AG_COMBINE_ZEROS.get(key)
    if buf is None:
        buf = torch.zeros(recv.shape, dtype=torch.bfloat16, device=recv.device)
        _AG_COMBINE_ZEROS[key] = buf
    return buf


def materialize_deepep_topk(adapter: Any, topk: Any, hidden: torch.Tensor) -> Any:
    if bool(getattr(topk, "skip_moe", False)):
        return topk
    router_logits = getattr(topk, "router_logits", None)
    if router_logits is None:
        return topk
    ids, weights = adapter.route_to_deepep_topk(
        router_logits,
        hidden,
        renormalize=bool(getattr(topk, "renormalize", False)),
    )
    return topk._replace(
        topk_ids=ids,
        topk_weights=weights,
        router_logits=None,
        deepep_topk=True,
    )


def afd_moe_output(hidden: torch.Tensor, topk: Any) -> torch.Tensor:
    if bool(getattr(topk, "skip_moe", False)):
        local_output = getattr(topk, "local_output", None)
        if local_output is None:
            raise RuntimeError("skip_moe topk is missing local_output")
        return local_output
    scale = float(getattr(topk, "routed_scaling_factor", 1.0))
    if scale != 1.0:
        hidden = hidden * scale
    shared_output = getattr(topk, "shared_output", None)
    if shared_output is not None:
        hidden = hidden + shared_output
    return hidden


def afd_dispatch_payload(hidden: torch.Tensor, topk: Any):
    if not bool(getattr(topk, "dispatch_fp8", True)):
        return hidden.contiguous(), None
    return _fp8_dispatch_quant(hidden)


@dataclass
class _AgGraphBuf:
    graph: torch.cuda.CUDAGraph
    capture_batch: Batch
    sub_batches: list | None = None  # mb=2: per-MB capture sub-batches


class AfdAgDecodeGraphRunner:
    """Per-bucket decode graph for the AG forward (embed..finalize)."""

    def __init__(
        self,
        *,
        runtime_state: Any,
        model: Any,
        adapters: Any,
        num_mb: int = 1,
        comm_stream: Any = None,
        lane_comm_streams: tuple[Any, ...] = (),
        runtime: Any,
        num_layers: int,
        hidden_size: int,
        bs_list: tuple[int, ...],
        log_path: str,
        label: str,
    ) -> None:
        self.state = runtime_state
        self.model = model
        self.adapters = list(adapters)
        self.adapter = self.adapters[0]
        self.num_mb = max(1, int(num_mb))
        if self.num_mb > 1 and len(self.adapters) < self.num_mb:
            raise RuntimeError(
                f"AG decode graph requires >= num_mb DeepEP adapters, "
                f"got adapters={len(self.adapters)} num_mb={self.num_mb}"
            )
        self.comm_stream = comm_stream
        self.lane_comm_streams = (
            tuple(lane_comm_streams or ()) if self.num_mb > 1 else ()
        )
        self.runtime = runtime
        self.device = runtime_state.device
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        # decode_graph_bs entries are PER-MB sizes (the coordinator pads a decode
        # batch to padded_per_mb * num_mb). For num_mb==1 per-MB == total. The
        # graphs dict is keyed by the TOTAL bs (== runtime len(padded_reqs)).
        self.bs_list = tuple(sorted(int(b) for b in bs_list if int(b) > 0))
        self.max_bs = (max(self.bs_list) if self.bs_list else 0) * self.num_mb
        self.log_path = log_path
        self.label = label
        self._pool = None
        self.graphs: dict[int, _AgGraphBuf] = {}
        # fixed input/output buffers (sliced per bucket)
        self._input_ids = torch.zeros((self.max_bs,), dtype=torch.int32, device=self.device)
        self._positions = torch.zeros((self.max_bs,), dtype=torch.int32, device=self.device)
        self._out_loc = torch.zeros((self.max_bs,), dtype=torch.int32, device=self.device)
        self._state_indices = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=self.device
        )
        self._final_out = torch.empty(
            (self.max_bs, self.hidden_size), dtype=torch.bfloat16, device=self.device
        )
        self._capture_initialized: set[int] = set()
        self._dummy_req = runtime._dummy_req
        # Per-layer/per-MB compute->comm->next-layer events, reused across
        # captured buckets (the event objects must persist for replay).
        self._pipeline_dispatch_ready = [
            [torch.cuda.Event() for _ in range(self.num_mb)]
            for _ in range(self.num_layers)
        ]
        self._pipeline_combine_done = [
            [torch.cuda.Event() for _ in range(self.num_mb)]
            for _ in range(self.num_layers)
        ]
        self._mb_capture_initialized: set[int] = set()

    def has_bucket(self, bs: int) -> bool:
        return int(bs) in self.graphs

    def _build_capture_batch(self, bs: int) -> Batch:
        reqs = [self._dummy_req] * int(bs)
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = reqs
        batch.input_ids = self._input_ids[:bs]
        batch.positions = self._positions[:bs]
        batch.out_loc = self._out_loc[:bs]
        batch.state_indices = self._state_indices[:bs]
        return batch

    def warmup(self, bs: int) -> None:
        bs = int(bs)
        if bs not in self.bs_list:
            raise RuntimeError(f"AG decode graph bucket bs={bs} not in {self.bs_list}")
        if self.num_mb > 1:
            self._warmup_mb(bs)  # bs is PER-MB here; graph keyed by bs*num_mb
            return
        if bs in self.graphs:
            return
        backend = self.state.attn_backends[0]
        if 0 not in self._capture_initialized:
            backend.init_capture_graph(
                max_seq_len=self.state.max_seq_len, bs_list=list(self.bs_list)
            )
            self._capture_initialized.add(0)

        capture_batch = self._build_capture_batch(bs)
        with nvtx_range(f"AFD_AG_GraphPrepareCapture_{bs}"):
            backend.prepare_for_capture(capture_batch)

        ctx = get_global_ctx()
        model, adapter = self.model, self.adapter
        out_buf = self._final_out[:bs]
        is_megamoe_backend = getattr(adapter, "backend", "deepep") == "megamoe_m2n"

        def _fn() -> None:
            with ctx.forward_batch(capture_batch):
                state = model.embed_input_ids(capture_batch.input_ids)
                hidden, residual = state.hidden_states, state.residual
                for layer in range(self.num_layers):
                    st = model.forward_attention(layer, hidden, residual)
                    hidden, residual = st.hidden_states, st.residual
                    if is_megamoe_backend and hasattr(model, "route_for_megamoe"):
                        topk = model.route_for_megamoe(layer, hidden, adapter, lane=0)
                    else:
                        topk = model.route(layer, hidden)
                    if bool(getattr(topk, "skip_moe", False)):
                        hidden = afd_moe_output(hidden, topk)
                        continue
                    if is_megamoe_backend:
                        hidden = adapter.ag_moe(
                            hidden, topk.topk_ids, topk.topk_weights,
                            external_quant=bool(
                                getattr(topk, "external_quant", False)
                            ),
                        )
                        hidden = afd_moe_output(hidden, topk)
                        continue
                    topk = materialize_deepep_topk(adapter, topk, hidden)
                    disp_x, disp_scale = afd_dispatch_payload(hidden, topk)
                    disp = adapter.dispatch(
                        disp_x,
                        topk.topk_ids,
                        topk.topk_weights,
                        hidden_states_scale=disp_scale,
                        num_max_dispatch_tokens_per_rank=bs,
                        router_logits=topk.router_logits,
                        renormalize=topk.renormalize,
                        topk_ids_are_deepep=topk.deepep_topk,
                    )
                    hidden = adapter.combine(ag_combine_input(disp.hidden_states), disp)
                    hidden = afd_moe_output(hidden, topk)
                final = model.finalize_hidden(hidden, residual)
                out_buf.copy_(final)

        # Run the capture body ONCE eagerly first (mirrors the eager decode path): forces all lazy
        # init (cuBLAS handles, attention workspace, JIT, DeepEP buffer setup) to
        # happen OUTSIDE the graph, otherwise those allocs/syncs are illegal during
        # capture ("operation failed due to a previous error during capture").
        with torch.cuda.stream(self.state.stream):
            _fn()
        self.state.stream.synchronize()
        graph, self._pool = capture_cuda_graph(
            device=self.device,
            engine_stream=self.state.stream,
            comm_stream=self.state.stream,
            overlap_comm=False,
            pool=self._pool,
            fn=_fn,
        )
        self.graphs[bs] = _AgGraphBuf(graph=graph, capture_batch=capture_batch)
        log_line(self.log_path, f"[{self.label}] afd_ag_decode_graph warmup:done bs={bs}", flush=True)

    def _warmup_mb(self, bs: int) -> None:
        """Capture the multi-MB AG forward for a decode bucket: per layer, each MB's
        attn/route/quant runs on the compute stream while DeepEP dispatch/combine
        runs on that MB's communication lane, the whole multi-stream region folded
        into one graph via overlap_comm=True. Each MB owns its own attention backend
        (backends[mb]), per-MB fixed metadata buffers, adapter lane, and comm stream.

        `bs` is the PER-MB bucket (a decode_graph_bs entry). Each MB has `bs` tokens;
        the total padded batch is `bs * num_mb`, which is how the graph is keyed
        (matches runtime len(padded_reqs))."""
        num_mb = self.num_mb
        mb_bs = int(bs)             # per-MB token count
        total = mb_bs * num_mb      # graph key (== runtime graph_bs)
        if total in self.graphs:
            return
        ctx = get_global_ctx()
        model = self.model
        out_buf = self._final_out[:total]
        comm_stream = self.comm_stream if self.comm_stream is not None else self.state.stream
        lane_streams = (
            self.lane_comm_streams
            if self.lane_comm_streams
            else tuple(comm_stream for _ in range(num_mb))
        )

        # whole-batch embed view + per-MB attention sub-batches over the fixed buffers
        whole = self._build_capture_batch(total)
        subs = []
        for mb in range(num_mb):
            s, e = mb * mb_bs, (mb + 1) * mb_bs
            backend = self.state.attn_backends[mb]
            if mb not in self._mb_capture_initialized:
                backend.init_capture_graph(
                    max_seq_len=self.state.max_seq_len,
                    bs_list=list(self.bs_list),  # per-MB sizes
                )
                self._mb_capture_initialized.add(mb)
            sub = Batch(reqs=[self._dummy_req] * mb_bs, phase="decode")
            sub.padded_reqs = [self._dummy_req] * mb_bs
            sub.input_ids = self._input_ids[s:e]
            sub.positions = self._positions[s:e]
            sub.out_loc = self._out_loc[s:e]
            sub.state_indices = self._state_indices[s:e]
            backend.prepare_for_capture(sub)
            subs.append(sub)
        whole.attn_metadata = subs[0].attn_metadata  # embed ignores it

        backends = self.state.attn_backends
        adapters = self.adapters
        engine_stream = self.state.stream

        def _fn() -> None:
            with ctx.forward_batch(whole):
                embed_state = model.embed_input_ids(whole.input_ids)
            hidden_full = embed_state.hidden_states
            hidden = [hidden_full[mb * mb_bs:(mb + 1) * mb_bs] for mb in range(num_mb)]
            state = _AfdAgMbState(
                num_mb=num_mb,
                bucket=mb_bs,
                lane_streams=lane_streams,
                subs=subs,
                backends=backends,
                hidden=hidden,
                residual=[None] * num_mb,
                ready_events=self._pipeline_dispatch_ready,
                done_events=self._pipeline_combine_done,
                dispatches=[
                    [None for _ in range(num_mb)] for _ in range(self.num_layers)
                ],
                route_topks=[
                    [None for _ in range(num_mb)] for _ in range(self.num_layers)
                ],
                retired=[],
                skip_marker=object(),
                use_megamoe=getattr(adapters[0], "backend", "deepep") == "megamoe_m2n",
            )
            with nvtx_range("AFD_AG_Graph_PipelineMB"):
                _run_ag_pipeline_body(
                    state=state,
                    num_layers=self.num_layers,
                    model=model,
                    adapters=adapters,
                    ctx=ctx,
                    engine_stream=engine_stream,
                    phase="decode",
                    retain_graph_objects=True,
                    retire_async_objects=False,
                )
            for mb in range(num_mb):
                with ctx.forward_batch(subs[mb]):
                    final_mb = model.finalize_hidden(
                        state.hidden[mb], state.residual[mb]
                    )
                s, e = mb * mb_bs, (mb + 1) * mb_bs
                out_buf[s:e].copy_(final_mb)

        # eager warmup run (lazy init out of the graph), sync BOTH streams
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
            overlap_comm=True,
            pool=self._pool,
            fn=_fn,
        )
        self.graphs[total] = _AgGraphBuf(graph=graph, capture_batch=whole, sub_batches=subs)
        log_line(
            self.log_path,
            f"[{self.label}] afd_ag_decode_graph mb warmup:done total={total} "
            f"mb_bs={mb_bs} pipeline_mb=1 schedule=paper_pipeline "
            f"lanes={len(lane_streams)}",
            flush=True,
        )

    def _replay_mb(self, batch: Batch, bs: int) -> torch.Tensor:
        # bs here is the TOTAL (== runtime graph_bs == len(padded_reqs)); the graph
        # was keyed by total in _warmup_mb.
        num_mb = self.num_mb
        buf = self.graphs[bs]
        real_subs = batch.afd_mb_subbatches
        with torch.cuda.stream(self.state.stream):
            self._input_ids[:bs].copy_(batch.input_ids[:bs])
            self._positions[:bs].copy_(batch.positions[:bs])
            self._out_loc[:bs].copy_(batch.out_loc[:bs])
            self._state_indices[:bs].copy_(batch.state_indices[:bs])
            for mb in range(num_mb):
                cap_sub = buf.sub_batches[mb]
                cap_sub.reqs = real_subs[mb].reqs
                cap_sub.padded_reqs = real_subs[mb].padded_reqs
                cap_sub.attn_metadata = real_subs[mb].attn_metadata
                self.state.attn_backends[mb].prepare_for_replay(cap_sub)
            buf.graph.replay()
        return self._final_out[:bs]

    def replay(self, batch: Batch, bs: int) -> torch.Tensor:
        """Copy the real decode inputs into the fixed buffers, update attn metadata,
        replay the captured graph, and return the finalized hidden [bs, hidden]."""
        bs = int(bs)
        if self.num_mb > 1:
            return self._replay_mb(batch, bs)
        buf = self.graphs[bs]
        backend = self.state.attn_backends[0]
        with torch.cuda.stream(self.state.stream):
            # copy the real step's inputs into the FIXED graph buffers (the captured
            # graph reads these addresses); keep the captured attn_metadata identity
            # and let prepare_for_replay refill it from the real reqs/positions.
            self._input_ids[:bs].copy_(batch.input_ids[:bs])
            self._positions[:bs].copy_(batch.positions[:bs])
            self._out_loc[:bs].copy_(batch.out_loc[:bs])
            self._state_indices[:bs].copy_(batch.state_indices[:bs])
            cap = buf.capture_batch
            cap.reqs = batch.reqs
            cap.padded_reqs = batch.padded_reqs
            # CRITICAL: prepare_for_replay copies batch.attn_metadata (the SOURCE)
            # into the captured fixed KV buffers that the graph reads. Point it at
            # the REAL current-step metadata (built by materialize_ag_plan's
            # prepare_metadata) — leaving the captured metadata here would copy the
            # capture buffers into themselves (no-op) and the graph would replay
            # with the stale warmup/dummy KV layout every step.
            cap.attn_metadata = batch.attn_metadata
            backend.prepare_for_replay(cap)
            buf.graph.replay()
        return self._final_out[:bs]



@dataclass
class _AfdAgInFlight:
    """AG step handle: sampled next_tokens are copied to host asynchronously on
    the D2H stream and synchronized when the fixed two-step pipeline retires it."""

    step_id: int
    sent_ns: int
    next_tokens_cpu: torch.Tensor  # pinned/host tensor, copy in flight on d2h stream
    copy_done: torch.cuda.Event


@dataclass
class _AfdAgMbState:
    num_mb: int
    bucket: int
    lane_streams: tuple[torch.cuda.Stream, ...]
    subs: list
    backends: list
    hidden: list[torch.Tensor]
    residual: list[Any]
    ready_events: list[list[torch.cuda.Event]]
    done_events: list[list[torch.cuda.Event]]
    dispatches: list[list[Any]]
    route_topks: list[list[Any]]
    retired: list[tuple[torch.cuda.Event, Any, Any]]
    skip_marker: object
    use_megamoe: bool


def _record_stream(tensor, stream: torch.cuda.Stream) -> None:
    if tensor is not None:
        tensor.record_stream(stream)


def _record_topk_streams(topk, stream: torch.cuda.Stream, *, include_router: bool) -> None:
    _record_stream(getattr(topk, "topk_ids", None), stream)
    _record_stream(getattr(topk, "topk_weights", None), stream)
    if include_router:
        _record_stream(getattr(topk, "router_logits", None), stream)
        _record_stream(getattr(topk, "shared_output", None), stream)


def _run_ag_pipeline_body(
    *,
    state: _AfdAgMbState,
    num_layers: int,
    model,
    adapters,
    ctx,
    engine_stream: torch.cuda.Stream,
    phase: str,
    retain_graph_objects: bool,
    retire_async_objects: bool,
) -> None:
    """Shared AG multi-MB pipeline body used by eager fallback and graph capture."""
    num_layers = int(num_layers)
    graph_live_tensors: list[torch.Tensor] = []
    graph_live_objects: list[Any] = []

    def retain_tensors(*items) -> None:
        if not (retain_graph_objects and torch.cuda.is_current_stream_capturing()):
            return
        graph_live_tensors.extend(tensor for tensor in items if tensor is not None)

    def retain_objects(*items) -> None:
        if not (retain_graph_objects and torch.cuda.is_current_stream_capturing()):
            return
        graph_live_objects.extend(obj for obj in items if obj is not None)

    def retire(force: bool = False) -> None:
        if not state.retired:
            return
        keep: list[tuple[torch.cuda.Event, Any, Any]] = []
        for event, disp_obj, topk_obj in state.retired:
            if force:
                event.synchronize()
            elif not event.query():
                keep.append((event, disp_obj, topk_obj))
        state.retired[:] = keep

    def launch_compute(layer: int, mb: int):
        ready_event = state.ready_events[layer][mb]
        prev_backend = ctx.attn_backend
        ctx.attn_backend = state.backends[mb]
        try:
            with ctx.forward_batch(state.subs[mb]):
                with torch.cuda.stream(engine_stream):
                    if layer > 0:
                        engine_stream.wait_event(state.done_events[layer - 1][mb])
                    st = model.forward_attention(
                        layer, state.hidden[mb], state.residual[mb]
                    )
                    state.hidden[mb], state.residual[mb] = (
                        st.hidden_states,
                        st.residual,
                    )
                    if state.use_megamoe and hasattr(model, "route_for_megamoe"):
                        adapter_mb = adapters[mb % len(adapters)]
                        topk = model.route_for_megamoe(
                            layer,
                            state.hidden[mb],
                            adapter_mb,
                            lane=mb,
                        )
                    else:
                        topk = model.route(layer, state.hidden[mb])
                    if state.use_megamoe or bool(getattr(topk, "skip_moe", False)):
                        disp_x, disp_scale = None, None
                    else:
                        adapter_mb = adapters[mb % len(adapters)]
                        topk = materialize_deepep_topk(
                            adapter_mb,
                            topk,
                            state.hidden[mb],
                        )
                        disp_x, disp_scale = afd_dispatch_payload(
                            state.hidden[mb],
                            topk,
                        )
                    ready_event.record(engine_stream)
        finally:
            ctx.attn_backend = prev_backend
        retain_tensors(
            disp_x,
            disp_scale,
            getattr(topk, "topk_ids", None),
            getattr(topk, "topk_weights", None),
            getattr(topk, "router_logits", None),
            getattr(topk, "shared_output", None),
            getattr(topk, "local_output", None),
        )
        return disp_x, disp_scale, topk

    def launch_dispatch(layer: int, mb: int, packet) -> None:
        disp_x, disp_scale, topk = packet
        comm_stream = state.lane_streams[mb % len(state.lane_streams)]
        if bool(getattr(topk, "skip_moe", False)):
            state.hidden[mb] = afd_moe_output(state.hidden[mb], topk)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(state.ready_events[layer][mb])
                state.done_events[layer][mb].record(comm_stream)
            state.dispatches[layer][mb] = state.skip_marker
            return

        adapter_mb = adapters[mb % len(adapters)]
        if state.use_megamoe:
            _record_stream(state.hidden[mb], comm_stream)
            _record_topk_streams(topk, comm_stream, include_router=False)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(state.ready_events[layer][mb])
                state.hidden[mb] = adapter_mb.ag_moe(
                    state.hidden[mb],
                    topk.topk_ids,
                    topk.topk_weights,
                    lane=mb,
                    external_quant=bool(getattr(topk, "external_quant", False)),
                )
                state.hidden[mb] = afd_moe_output(state.hidden[mb], topk)
            state.dispatches[layer][mb] = ()
            return

        if disp_x is None:
            raise RuntimeError(f"Missing AG dispatch payload for layer={layer} mb={mb}")
        _record_stream(disp_x, comm_stream)
        _record_stream(disp_scale, comm_stream)
        _record_topk_streams(topk, comm_stream, include_router=True)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(state.ready_events[layer][mb])
            disp = adapter_mb.dispatch(
                disp_x,
                topk.topk_ids,
                topk.topk_weights,
                hidden_states_scale=disp_scale,
                num_max_dispatch_tokens_per_rank=state.bucket,
                do_cpu_sync=BaseAfdWorker._deepep_do_cpu_sync(phase),
                router_logits=topk.router_logits,
                renormalize=topk.renormalize,
                topk_ids_are_deepep=topk.deepep_topk,
            )
        state.dispatches[layer][mb] = disp
        state.route_topks[layer][mb] = topk

    def launch_combine(layer: int, mb: int) -> None:
        disp = state.dispatches[layer][mb]
        if disp is state.skip_marker:
            state.dispatches[layer][mb] = None
            state.route_topks[layer][mb] = None
            return
        if disp is None:
            raise RuntimeError(f"Missing AG dispatch for layer={layer} mb={mb}")

        comm_stream = state.lane_streams[mb % len(state.lane_streams)]
        if state.use_megamoe:
            with torch.cuda.stream(comm_stream):
                state.done_events[layer][mb].record(comm_stream)
            state.dispatches[layer][mb] = None
            return

        adapter_mb = adapters[mb % len(adapters)]
        topk = state.route_topks[layer][mb]
        with torch.cuda.stream(comm_stream):
            state.hidden[mb] = adapter_mb.combine(
                ag_combine_input(disp.hidden_states),
                disp,
            )
            state.hidden[mb] = afd_moe_output(state.hidden[mb], topk)
            state.done_events[layer][mb].record(comm_stream)
        if retire_async_objects:
            state.retired.append((state.done_events[layer][mb], disp, topk))
            if len(state.retired) >= 4:
                retire(force=True)
        else:
            retain_objects(disp, topk)
        state.dispatches[layer][mb] = None
        state.route_topks[layer][mb] = None

    # AG collective order:
    #   compute+D(L0, mb0), compute+D(L0, mb1), ...
    #   C(L, mb) then immediately compute+D(L+1, mb).
    # This mirrors the EG pipeline body without materializing a schedule list.
    for mb in range(state.num_mb):
        packet = launch_compute(0, mb)
        launch_dispatch(0, mb, packet)
        if retire_async_objects:
            retire()
    for layer in range(num_layers):
        for mb in range(state.num_mb):
            launch_combine(layer, mb)
            if layer + 1 < num_layers:
                packet = launch_compute(layer + 1, mb)
                launch_dispatch(layer + 1, mb, packet)
            if retire_async_objects:
                retire()
    if retire_async_objects:
        retire(force=True)
    for mb in range(state.num_mb):
        engine_stream.wait_event(state.done_events[num_layers - 1][mb])


class AfdAttentionWorker(BaseAfdWorker):
    """Per-GPU AFD attention worker actor."""

    def __init__(self, **kwargs):
        kwargs.pop("role", None)
        super().__init__(role="attention", **kwargs)

    # ------------------------------------------------------------------
    # BaseAfdWorker hooks
    # ------------------------------------------------------------------

    def _create_runtime(self, config: AfdRuntimeConfig):
        return AfdAttentionRuntime(config, AfdAttentionState(config))

    def _run_hot_rpc_loop(self) -> None:
        """Drive the hot loop until an AfdStopCmd is received."""
        self._prepare_hot_rpc_runtime()
        if self.disable_overlap:
            self.centralized_normal_loop()
        else:
            self.centralized_overlap_loop()

    def centralized_normal_loop(self) -> None:
        """Synchronous AG loop: run a step, wait for its D2H copy, then reply."""
        while True:
            with nvtx_range("AFD_AttnCentralizedNormalLoop_Iter"):
                with nvtx_range(nvtx_label("AFD_AttnCentralizedHotLoop_Recv", blocking=True)):
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
                if isinstance(cmd, AfdRunAGStepCmd):
                    self._process_last_afd_ag(
                        self._run_afd_ag_step(cmd.plan, sent_ns=cmd.sent_ns)
                    )
                    continue
                raise RuntimeError(
                    f"Unsupported centralized attention command: {type(cmd).__name__}"
                )

    def centralized_overlap_loop(self) -> None:
        # Launch a finite future-command window before retiring the oldest reply
        # handle. With the default pending window 3, cudaGraphLaunch for N+2
        # enqueues before the CPU waits on N's D2H copy event.
        pending_afd_ag: deque[_AfdAgInFlight] = deque()

        def retire_afd_ag(force: bool = False) -> None:
            while pending_afd_ag and (
                force or len(pending_afd_ag) > self.async_retire_depth
            ):
                self._process_last_afd_ag(pending_afd_ag.popleft())

        while True:
            with nvtx_range("AFD_AttnCentralizedHotLoop_Iter"):
                with nvtx_range(
                    nvtx_label("AFD_AttnCentralizedHotLoop_Recv", blocking=True)
                ):
                    cmd = self._recv_hot_command(blocking=True)
                if cmd is None:
                    continue
                if isinstance(cmd, AfdFlushStepCmd):
                    retire_afd_ag(force=True)
                    continue
                if isinstance(cmd, AfdStopCmd):
                    retire_afd_ag(force=True)
                    return
                if isinstance(cmd, AfdProfilerCmd):
                    self.handle_profiler_cmd(cmd)
                    continue
                if isinstance(cmd, AfdRunAGStepCmd):
                    pending_afd_ag.append(
                        self._run_afd_ag_step(cmd.plan, sent_ns=cmd.sent_ns)
                    )
                    retire_afd_ag()
                    continue
                raise RuntimeError(
                    f"Unsupported centralized attention command: {type(cmd).__name__}"
                )

    def _run_ag_forward_eager(self, batch, plan: AfdAGStepPlan, model, adapters, ctx):
        """Eager AG fallback used for prefill, graph-off debug, or bucket miss.

        The same pipeline body handles num_mb=1 and num_mb>1, so eager execution
        has one ordering contract: embed once, then per-MB attention/route and
        dispatch/combine through the AG pipeline.
        """
        state = self._build_ag_mb_state(batch, plan, model, adapters, ctx)
        with nvtx_range("AFD_AG_PipelineMB"):
            _run_ag_pipeline_body(
                state=state,
                num_layers=self.num_layers,
                model=model,
                adapters=adapters,
                ctx=ctx,
                engine_stream=self.runtime_state.stream,
                phase=str(plan.phase),
                retain_graph_objects=False,
                retire_async_objects=True,
            )
        for stream in self._lane_comm_streams():
            self.runtime_state.stream.wait_stream(stream)
        return self._finalize_ag_mbs(state, model, ctx)

    def _build_ag_mb_state(
        self,
        batch,
        plan: AfdAGStepPlan,
        model,
        adapters,
        ctx,
    ) -> _AfdAgMbState:
        num_mb = int(plan.num_mb)
        runtime_state = self.runtime_state
        tok_offs = plan.microbatch_token_offsets
        with ctx.forward_batch(batch):
            with torch.cuda.stream(runtime_state.stream):
                embed_state = model.embed_input_ids(batch.input_ids)
        hidden_full = embed_state.hidden_states
        hidden = [
            hidden_full[int(tok_offs[mb]) : int(tok_offs[mb + 1])]
            for mb in range(num_mb)
        ]
        return _AfdAgMbState(
            num_mb=num_mb,
            bucket=int(plan.dispatch_bucket),
            lane_streams=self._lane_comm_streams(),
            subs=batch.afd_mb_subbatches,
            backends=runtime_state.attn_backends,
            hidden=hidden,
            residual=[None] * num_mb,
            ready_events=[
                [torch.cuda.Event() for _ in range(num_mb)]
                for _ in range(self.num_layers)
            ],
            done_events=[
                [torch.cuda.Event() for _ in range(num_mb)]
                for _ in range(self.num_layers)
            ],
            dispatches=[
                [None for _ in range(num_mb)] for _ in range(self.num_layers)
            ],
            route_topks=[
                [None for _ in range(num_mb)] for _ in range(self.num_layers)
            ],
            retired=[],
            skip_marker=object(),
            use_megamoe=getattr(adapters[0], "backend", "deepep") == "megamoe_m2n",
        )

    def _finalize_ag_mbs(self, state: _AfdAgMbState, model, ctx) -> torch.Tensor:
        with torch.cuda.stream(self.runtime_state.stream):
            finals = []
            for mb in range(state.num_mb):
                with ctx.forward_batch(state.subs[mb]):
                    finals.append(
                        model.finalize_hidden(state.hidden[mb], state.residual[mb])
                    )
            return finals[0] if state.num_mb == 1 else torch.cat(finals, dim=0)

    def _materialize_ag_batch(self, plan: AfdAGStepPlan):
        sched = self.runtime
        try:
            return sched.materialize_ag_plan(plan)
        except Exception:
            tables = tuple(int(x) for x in plan.table_indices)
            extends = tuple(int(x) for x in plan.extend_lens)
            cached = sched._table_cached_len
            cached_preview = tuple(
                (int(t), int(cached.get(int(t), -1))) for t in tables[:8]
            )
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_ag_materialize:error "
                f"step_id={int(plan.step_id)} phase={plan.phase} "
                f"real={int(plan.real_size)} tables_head={tables[:8]} "
                f"tables_minmax={((min(tables), max(tables)) if tables else None)} "
                f"extend_head={extends[:8]} cached_head={cached_preview} "
                f"mb_offsets={tuple(int(x) for x in plan.microbatch_offsets)} "
                f"mb_token_offsets={tuple(int(x) for x in plan.microbatch_token_offsets)}",
                flush=True,
            )
            raise

    def _ag_graph_replay_info(self, batch, plan: AfdAGStepPlan) -> tuple[bool, int]:
        graph_bs = len(batch.padded_reqs)
        ag_graph = self._afd_ag_graph
        use_graph = (
            self.enable_decode_graph
            and plan.use_decode_graph
            and ag_graph is not None
            and plan.phase == "decode"
            and ag_graph.has_bucket(graph_bs)
        )
        return bool(use_graph), int(graph_bs)

    def _run_ag_forward(self, batch, plan: AfdAGStepPlan, ctx) -> torch.Tensor:
        model = self._afd_ag_model
        use_graph, graph_bs = self._ag_graph_replay_info(batch, plan)
        if use_graph:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_ag_decode_graph:replay "
                f"step_id={int(plan.step_id)} real={int(plan.real_size)} "
                f"bs={int(graph_bs)} num_mb={int(plan.num_mb)}",
                flush=True,
            )
            return self._afd_ag_graph.replay(batch, graph_bs)
        return self._run_ag_forward_eager(batch, plan, model, self._afd_adapters, ctx)

    def _sample_and_writeback_ag(
        self,
        batch,
        final: torch.Tensor,
        ctx,
        *,
        step_id: int,
        profile: bool,
    ) -> torch.Tensor:
        sched = self.runtime
        runtime_state = self.runtime_state
        num_sampling = int(batch.afd_num_sampling)
        if num_sampling <= 0:
            return torch.empty((0,), dtype=torch.int32, device=runtime_state.device)

        total_start_ns = time.perf_counter_ns()
        select_start_ns = total_start_ns
        sel = final.index_select(0, batch.afd_last_indices)
        select_done_ns = time.perf_counter_ns()
        sampling_batch_start_ns = select_done_ns
        sampling_batch = sched.build_sampling_batch(num_sampling)
        sampling_batch_done_ns = time.perf_counter_ns()
        with ctx.forward_batch(sampling_batch):
            lm_head_start_ns = time.perf_counter_ns()
            logits = self._afd_ag_model.forward_lm_head(sel)
            lm_head_done_ns = time.perf_counter_ns()
            args_start_ns = lm_head_done_ns
            sampling_args = batch.afd_sampling_plan.to_sampling_args(
                runtime_state.device, int(sched.config.model_config.vocab_size)
            )
            args_done_ns = time.perf_counter_ns()
            sample_start_ns = args_done_ns
            next_tokens = self._afd_sampler.sample(logits, sampling_args).to(torch.int32)
            sample_done_ns = time.perf_counter_ns()

        # Writeback runs on the runtime stream but reads next_tokens produced on
        # the compute stream; order it so the next step's embed never sees stale ids.
        wait_start_ns = sample_done_ns
        sched.stream.wait_stream(runtime_state.stream)
        wait_done_ns = time.perf_counter_ns()
        writeback_start_ns = wait_done_ns
        sched.writeback_tokens_ag(batch, next_tokens)
        writeback_done_ns = time.perf_counter_ns()
        if profile:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] control_ag_sample "
                f"step_id={int(step_id)} num_sampling={int(num_sampling)} "
                f"index_select_ms={(select_done_ns - select_start_ns) / 1_000_000.0:.3f} "
                f"sampling_batch_ms={(sampling_batch_done_ns - sampling_batch_start_ns) / 1_000_000.0:.3f} "
                f"lm_head_enqueue_ms={(lm_head_done_ns - lm_head_start_ns) / 1_000_000.0:.3f} "
                f"sample_args_ms={(args_done_ns - args_start_ns) / 1_000_000.0:.3f} "
                f"sampler_enqueue_ms={(sample_done_ns - sample_start_ns) / 1_000_000.0:.3f} "
                f"write_wait_ms={(wait_done_ns - wait_start_ns) / 1_000_000.0:.3f} "
                f"writeback_enqueue_ms={(writeback_done_ns - writeback_start_ns) / 1_000_000.0:.3f} "
                f"total_ms={(writeback_done_ns - total_start_ns) / 1_000_000.0:.3f}",
            )
        return next_tokens

    def _copy_next_tokens_to_host(
        self,
        step_id: int,
        sent_ns: int,
        next_tokens: torch.Tensor,
    ) -> _AfdAgInFlight:
        sched = self.runtime
        runtime_state = self.runtime_state
        d2h = self._d2h_stream
        d2h.wait_stream(runtime_state.stream)
        next_tokens.record_stream(sched.stream)
        next_tokens.record_stream(d2h)
        with torch.cuda.stream(d2h):
            next_tokens_cpu_t = next_tokens.to("cpu", non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(d2h)
        return _AfdAgInFlight(
            step_id=int(step_id),
            sent_ns=int(sent_ns),
            next_tokens_cpu=next_tokens_cpu_t,
            copy_done=copy_done,
        )

    def _lane_comm_streams(self) -> tuple[torch.cuda.Stream, ...]:
        return tuple(self._afd_comm_streams)

    def _wait_for_ag_materialization(self) -> None:
        sched = self.runtime
        runtime_state = self.runtime_state
        runtime_state.stream.wait_stream(sched.stream)
        self._comm_stream.wait_stream(sched.stream)
        for stream in self._lane_comm_streams():
            if stream is not self._comm_stream:
                stream.wait_stream(sched.stream)

    def _run_afd_ag_step(
        self,
        plan: AfdAGStepPlan,
        *,
        sent_ns: int = 0,
    ) -> _AfdAgInFlight:
        """Run one AG step and return the delayed D2H token-copy handle."""
        runtime_state = self.runtime_state
        profile = self._control_profile_enabled(int(plan.step_id))
        total_start_ns = time.perf_counter_ns()
        materialize_start_ns = total_start_ns
        batch = self._materialize_ag_batch(plan)
        materialize_done_ns = time.perf_counter_ns()
        use_graph, graph_bs = self._ag_graph_replay_info(batch, plan)
        ctx = get_global_ctx()
        # materialize_ag_plan writes input_ids/positions/metadata on sched.stream;
        # both compute and comm streams must observe those writes before launch.
        wait_start_ns = materialize_done_ns
        self._wait_for_ag_materialization()
        wait_done_ns = time.perf_counter_ns()
        with torch.cuda.stream(runtime_state.stream):
            forward_start_ns = time.perf_counter_ns()
            final = self._run_ag_forward(batch, plan, ctx)
            forward_done_ns = time.perf_counter_ns()
            next_tokens = self._sample_and_writeback_ag(
                batch,
                final,
                ctx,
                step_id=int(plan.step_id),
                profile=profile,
            )
            sample_done_ns = time.perf_counter_ns()
        # ONE-STEP-AHEAD: instead of synchronizing the compute/runtime streams and
        # blocking on a gpu->cpu copy EVERY step (the big AG/EG replay gap), queue an
        # async D2H of next_tokens on the dedicated d2h stream + record a copy-done
        # event, and return the in-flight handle. The host read + reply happen one
        # step LATER in _process_last_afd_ag (after the NEXT step's GPU work is already
        # launched), so the compute stream runs steps back-to-back with no host stall.
        copy_start_ns = time.perf_counter_ns()
        handle = self._copy_next_tokens_to_host(
            int(plan.step_id),
            int(sent_ns),
            next_tokens,
        )
        copy_done_ns = time.perf_counter_ns()
        if profile:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] control_ag_step "
                f"step_id={int(plan.step_id)} phase={plan.phase} "
                f"real={int(plan.real_size)} padded={len(batch.padded_reqs)} "
                f"bucket={int(plan.dispatch_bucket)} num_mb={int(plan.num_mb)} "
                f"use_graph={int(bool(use_graph))} graph_bs={int(graph_bs)} "
                f"materialize_ms={(materialize_done_ns - materialize_start_ns) / 1_000_000.0:.3f} "
                f"stream_wait_ms={(wait_done_ns - wait_start_ns) / 1_000_000.0:.3f} "
                f"forward_enqueue_ms={(forward_done_ns - forward_start_ns) / 1_000_000.0:.3f} "
                f"sample_enqueue_ms={(sample_done_ns - forward_done_ns) / 1_000_000.0:.3f} "
                f"d2h_enqueue_ms={(copy_done_ns - copy_start_ns) / 1_000_000.0:.3f} "
                f"total_enqueue_ms={(copy_done_ns - total_start_ns) / 1_000_000.0:.3f}",
            )
        return handle

    def _process_last_afd_ag(self, last: "_AfdAgInFlight | None") -> None:
        """Finalize the PREVIOUS AG step: wait its (one-step-delayed) D2H copy event,
        read next_tokens on the host, and reply. Called after the current step's GPU
        work is launched, so the wait overlaps the current step's kernels."""
        if last is None:
            return
        with nvtx_range(nvtx_label("AFD_AG_ProcessLast", step=last.step_id)):
            profile = self._control_profile_enabled(int(last.step_id))
            start_ns = time.perf_counter_ns()
            last.copy_done.synchronize()
            sync_done_ns = time.perf_counter_ns()
            next_tokens_cpu = [int(t) for t in last.next_tokens_cpu.tolist()]
            list_done_ns = time.perf_counter_ns()
            self._send_reply(
                AfdAGStepReply(
                    worker_rank=self.worker_rank,
                    step_id=last.step_id,
                    sent_ns=int(last.sent_ns),
                    next_tokens=next_tokens_cpu,
                )
            )
            send_done_ns = time.perf_counter_ns()
            if profile:
                log_line(
                    self.log_path,
                    f"[{self.role} rank={self.tp_rank}] control_ag_retire "
                    f"step_id={int(last.step_id)} "
                    f"d2h_sync_ms={(sync_done_ns - start_ns) / 1_000_000.0:.3f} "
                    f"tolist_ms={(list_done_ns - sync_done_ns) / 1_000_000.0:.3f} "
                    f"reply_send_ms={(send_done_ns - list_done_ns) / 1_000_000.0:.3f} "
                    f"total_ms={(send_done_ns - start_ns) / 1_000_000.0:.3f}",
                )

    def _log_plan_recv(
        self,
        cmd,
        *,
        blocking: bool,
        recv_wait_ms: float,
    ) -> None:
        if not isinstance(cmd, AfdRunAGStepCmd):
            return
        profile = self._control_profile_enabled(int(cmd.plan.step_id))
        if not profile:
            return
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] plan_recv kind=attention "
            f"step_id={cmd.plan.step_id} worker_rank={self.worker_rank} "
            f"blocking={int(blocking)} recv_wait_ms={float(recv_wait_ms):.3f}",
        )
