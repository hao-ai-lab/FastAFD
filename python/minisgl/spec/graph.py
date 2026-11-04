"""CUDA graphs for the AG side of a speculative verify step.

A verify step pads every microbatch to ``bs`` request slots of ``block`` tokens
(afd_scheduler._pad_verify_batch), so its shapes are fixed by (bs, block, num_mb)
exactly like a decode step's by bs. The graph covers embed .. finalize: per layer
the verify attention (trtllm decode kernel in multi-token mode), the route, and
the MoE dispatch/combine with a bucket of bs * block tokens per microbatch, plus
the capture of the target hidden states the draft conditions on.

Replay copies the step's token inputs and attention metadata into the captured
buffers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from minisgl.core import Batch, get_global_ctx
from minisgl.utils import nvtx_range

from ..afd_support import log_line
from ..cuda_graph_utils import capture_cuda_graph


@dataclass
class _VerifyGraph:
    graph: torch.cuda.CUDAGraph
    whole: Batch
    subs: list[Batch]
    finish: torch.cuda.CUDAGraph | None = None  # accept + commit + reply
    draft: torch.cuda.CUDAGraph | None = None  # draft context + propose
    pending: dict | None = None  # head outputs the draft graph reads


class AfdAgVerifyGraphRunner:
    def __init__(
        self,
        *,
        runtime_state: Any,
        model: Any,
        adapters: Any,
        num_mb: int,
        comm_stream: Any,
        lane_comm_streams: tuple[Any, ...],
        runtime: Any,
        spec: Any,
        num_layers: int,
        hidden_size: int,
        bs_list: tuple[int, ...],
        log_path: str,
        label: str,
    ) -> None:
        self.state = runtime_state
        self.model = model
        self.adapters = list(adapters)
        self.num_mb = max(1, int(num_mb))
        self.comm_stream = comm_stream
        self.lane_comm_streams = tuple(lane_comm_streams or ()) if self.num_mb > 1 else ()
        self.runtime = runtime
        self.spec = spec
        self.block = int(spec.spec.block)
        self.device = runtime_state.device
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        # MINISGL_AFD_LANE_COMPUTE_STREAMS=1: one compute stream per microbatch (see _AfdAgMbState.compute_streams)
        self.lane_compute_streams: tuple[torch.cuda.Stream, ...] = ()
        if self.num_mb > 1 and os.environ.get("MINISGL_AFD_LANE_COMPUTE_STREAMS") == "1":
            self.lane_compute_streams = tuple(torch.cuda.Stream(device=self.device) for _ in range(self.num_mb))
        self.bs_list = tuple(sorted(int(b) for b in bs_list if int(b) > 0))
        self.log_path = log_path
        self.label = label
        self._pool = None
        self.graphs: dict[int, _VerifyGraph] = {}
        max_reqs = (max(self.bs_list) if self.bs_list else 0) * self.num_mb
        max_tokens = max_reqs * self.block
        dev = self.device
        self._input_ids = torch.zeros((max_tokens,), dtype=torch.int32, device=dev)
        self._positions = torch.zeros((max_tokens,), dtype=torch.int32, device=dev)
        self._out_loc = torch.zeros((max_tokens,), dtype=torch.int32, device=dev)
        self._state_indices = torch.zeros((max_tokens,), dtype=torch.int64, device=dev)
        self._final_out = torch.empty((max_tokens, self.hidden_size), dtype=torch.bfloat16, device=dev)
        self._features = spec.new_feature_buffer(max_tokens, torch.bfloat16)
        self._uniforms = torch.zeros((max_reqs,), dtype=torch.float32, device=dev)
        # Packed reply rows (block + 1 ints per slot), then (device-input mode) each slot's committed
        # length after the step, so the host learns the exact lengths with the same copy.
        self._reply = torch.zeros((max_reqs * (self.block + 2),), dtype=torch.int32, device=dev)
        self.device_inputs = bool(getattr(spec, "device_inputs", False))
        self._dev_cols = None
        self.reply_ready: torch.cuda.Event | None = None
        self._ready = [[torch.cuda.Event() for _ in range(self.num_mb)] for _ in range(self.num_layers)]
        self._done = [[torch.cuda.Event() for _ in range(self.num_mb)] for _ in range(self.num_layers)]
        self._backends_initialized: set[int] = set()

    def has_bucket(self, total_reqs: int) -> bool:
        return int(total_reqs) in self.graphs

    def _capture_sub(self, mb: int, bs: int) -> Batch:
        from minisgl.attention.trtllm import TRTLLMMetadata

        block = self.block
        dummy = self.runtime._verify_dummy_req
        r0, r1 = mb * bs, (mb + 1) * bs
        t0, t1 = r0 * block, r1 * block
        backend = self.state.attn_backends[mb]
        if mb not in self._backends_initialized:
            backend.init_capture_graph(max_seq_len=self.state.max_seq_len, bs_list=list(self.bs_list))
            self._backends_initialized.add(mb)
        cap = backend.capture
        sub = Batch(reqs=[dummy] * bs, phase="prefill")
        sub.padded_reqs = [dummy] * bs
        sub.input_ids = self._input_ids[t0:t1]
        sub.positions = self._positions[t0:t1]
        sub.out_loc = self._out_loc[t0:t1]
        sub.state_indices = self._state_indices[t0:t1]
        sub.afd_spec_verify = True
        sub.attn_metadata = TRTLLMMetadata(
            cu_seqlens_k=cap.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=cap.cu_seqlens_q[: bs + 1],
            cache_seqlens=cap.seq_lens[:bs],
            max_seqlen_k=cap.page_table.size(1) * backend.page_size,
            max_seqlen_q=block,
            page_table=cap.page_table[:bs, :],
            q_len_per_req=block,
        )
        return sub

    def warmup(self, bs: int) -> None:
        """Capture the verify forward for ``bs`` request slots per microbatch."""
        from ..afd_attention_worker import _AfdAgMbState, _run_ag_pipeline_body

        bs = int(bs)
        num_mb, block = self.num_mb, self.block
        total = bs * num_mb
        if total in self.graphs:
            return
        tokens = total * block
        mb_tokens = bs * block
        dummy = self.runtime._verify_dummy_req
        whole = Batch(reqs=[dummy] * total, phase="prefill")
        whole.padded_reqs = [dummy] * total
        whole.input_ids = self._input_ids[:tokens]
        whole.afd_spec_verify = True
        subs = [self._capture_sub(mb, bs) for mb in range(num_mb)]
        whole.attn_metadata = subs[0].attn_metadata
        with torch.cuda.stream(self.state.stream):
            # Capture inputs: every slot is a fresh dummy sequence on the dummy row,
            # its KV on the dummy page (the multi-token kernel needs kv_len >= block).
            dummy_page = int(self.state.num_pages)
            self._state_indices[:tokens].fill_(int(dummy.table_idx))
            self._positions[:tokens].copy_(torch.arange(block, device=self.device).repeat(total))
            self._out_loc[:tokens].fill_(dummy_page * int(self.runtime.config.page_size))
            for mb in range(num_mb):
                cap = self.state.attn_backends[mb].capture
                cap.seq_lens[:bs].fill_(block)
                cap.page_table[:bs].fill_(dummy_page)
        ctx = get_global_ctx()
        model, adapters = self.model, self.adapters
        engine_stream = self.state.stream
        comm_stream = self.comm_stream if self.comm_stream is not None else engine_stream
        lane_streams = (
            self.lane_comm_streams if self.lane_comm_streams else tuple(comm_stream for _ in range(num_mb))
        )
        out_buf = self._final_out[:tokens]
        features = self._features[:tokens]

        def _fn() -> None:
            with ctx.forward_batch(whole):
                embed_state = model.embed_input_ids(whole.input_ids)
            hidden_full = embed_state.hidden_states
            state = _AfdAgMbState(
                num_mb=num_mb,
                bucket=mb_tokens,
                lane_streams=lane_streams,
                subs=subs,
                backends=self.state.attn_backends,
                hidden=[hidden_full[mb * mb_tokens : (mb + 1) * mb_tokens] for mb in range(num_mb)],
                residual=[None] * num_mb,
                ready_events=self._ready,
                done_events=self._done,
                dispatches=[[None] * num_mb for _ in range(self.num_layers)],
                route_topks=[[None] * num_mb for _ in range(self.num_layers)],
                retired=[],
                skip_marker=object(),
                use_megamoe=getattr(adapters[0], "backend", "deepep") == "megamoe_m2n",
                capture_cols=self.spec.capture_cols,
                features=[features[mb * mb_tokens : (mb + 1) * mb_tokens] for mb in range(num_mb)],
                compute_streams=self.lane_compute_streams or None,
            )
            with nvtx_range("AFD_AG_VerifyGraph_PipelineMB"):
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
                    final_mb = model.finalize_hidden(state.hidden[mb], state.residual[mb])
                out_buf[mb * mb_tokens : (mb + 1) * mb_tokens].copy_(final_mb)

        with torch.cuda.stream(engine_stream):
            _fn()
        engine_stream.synchronize()
        for stream in tuple(lane_streams) + self.lane_compute_streams:
            stream.synchronize()
        graph, self._pool = capture_cuda_graph(
            device=self.device,
            engine_stream=engine_stream,
            comm_stream=comm_stream,
            extra_streams=(tuple(lane_streams) + self.lane_compute_streams) if num_mb > 1 else (),
            overlap_comm=num_mb > 1,
            pool=self._pool,
            fn=_fn,
        )
        buf = _VerifyGraph(graph=graph, whole=whole, subs=subs)
        buf.finish, buf.draft, buf.pending = self._capture_finish(total, subs, comm_stream)
        self.graphs[total] = buf
        log_line(
            self.log_path,
            f"[{self.label}] afd_ag_verify_graph warmup:done total={total} mb_bs={bs} "
            f"block={block} tokens={tokens}",
            flush=True,
        )

    def _capture_finish(self, total: int, subs: list[Batch], comm_stream):
        """Capture the verify finish over the bucket's slots as two graphs.

        The head (accept, commit, packed reply) and the draft (draft context,
        propose) replay back to back; the reply's D2H waits only for the head, so
        the host prepares the next step while the draft runs.
        """
        block = self.block
        tokens = total * block
        sim = self.spec.spec.sim_accept > 0
        reply = self._reply[: total * (block + 1)]
        tail = self._reply[total * (block + 1) : total * (block + 2)]
        pending: dict = {}

        def _head() -> None:
            out, outputs = self.spec.finish_head(
                self._final_out[:tokens], self._input_ids[:tokens], self._positions[:tokens],
                self._out_loc[:tokens], self._state_indices[:tokens], self._features[:tokens],
                subs, self._uniforms[:total] if sim else None,
            )
            reply.copy_(out)
            if self.device_inputs:
                tail.copy_(outputs["start"].to(torch.int32))
            pending.clear()
            pending.update(outputs)

        def _draft() -> None:
            self.spec.finish_draft(pending)

        engine_stream = self.state.stream
        with torch.cuda.stream(engine_stream):
            _head()
            _draft()
        engine_stream.synchronize()
        self.spec._marks.clear()
        graphs = []
        # The head's outputs stay referenced in ``pending``, so the draft capture
        # (same pool) does not reuse their memory.
        for fn in (_head, _draft):
            graph, self._pool = capture_cuda_graph(
                device=self.device,
                engine_stream=engine_stream,
                comm_stream=engine_stream,
                overlap_comm=False,
                pool=self._pool,
                fn=fn,
            )
            graphs.append(graph)
        return graphs[0], graphs[1], pending

    def replay_finish(self, total_reqs: int) -> torch.Tensor:
        """Replay the finish of the step whose forward just replayed; packed reply of every slot.

        ``reply_ready`` is recorded once the reply is final (before the draft).
        """
        total = int(total_reqs)
        buf = self.graphs[total]
        with torch.cuda.stream(self.state.stream):
            if self.spec.spec.sim_accept > 0:
                self._uniforms[:total].uniform_(generator=self.spec._generator)
            buf.finish.replay()
            self.reply_ready = torch.cuda.Event()
            self.reply_ready.record()
            buf.draft.replay()
        width = self.block + 2 if self.device_inputs else self.block + 1
        return self._reply[: total * width]

    def replay(self, batch: Batch, total_reqs: int) -> torch.Tensor:
        """Run a materialized verify step through its graph; returns the final hidden."""
        buf = self.graphs[int(total_reqs)]
        block = self.block
        tokens = int(total_reqs) * block
        with torch.cuda.stream(self.state.stream):
            if self.device_inputs:
                self._replay_inputs_from_device(batch, buf, int(total_reqs))
            else:
                self._input_ids[:tokens].copy_(batch.input_ids[:tokens])
                self._positions[:tokens].copy_(batch.positions[:tokens])
                self._out_loc[:tokens].copy_(batch.out_loc[:tokens])
                self._state_indices[:tokens].copy_(batch.state_indices[:tokens])
                for mb, real_sub in enumerate(batch.afd_mb_subbatches):
                    cap_sub = buf.subs[mb]
                    cap_sub.attn_metadata, captured = real_sub.attn_metadata, cap_sub.attn_metadata
                    self.state.attn_backends[mb].prepare_for_replay(cap_sub)
                    cap_sub.attn_metadata = captured
            buf.graph.replay()
        batch.afd_spec_features = self._features[:tokens]
        batch.afd_spec_graph_total = int(total_reqs)
        return self._final_out[:tokens]


    def _replay_inputs_from_device(self, batch: Batch, buf: _VerifyGraph, total: int) -> None:
        """Device-input mode: every length-dependent input from the device committed lengths.

        The host materialized this step from an upper bound on each row's start (steps still in
        flight); its positions / KV slots / attention metadata are not used. Only the slot -> table
        row map comes from the host (it does not depend on lengths).
        """
        from .device_inputs import verify_inputs_from_committed

        block = self.block
        tokens = total * block
        rows = batch.state_indices[:tokens].view(total, block)[:, 0].to(torch.int64)
        runtime = self.runtime
        backend0 = self.state.attn_backends[0]
        cols = int(backend0.capture.page_table.size(1))
        positions, out_loc, input_ids, start, seq_lens, block_table = verify_inputs_from_committed(
            runtime._spec_committed_dev, rows, block, runtime.page_table, runtime.token_pool,
            int(runtime.config.page_size), cols,
        )
        self._positions[:tokens].copy_(positions)
        self._out_loc[:tokens].copy_(out_loc)
        self._input_ids[:tokens].copy_(input_ids)
        self._state_indices[:tokens].copy_(batch.state_indices[:tokens])
        mb_bs = total // self.num_mb
        for mb in range(self.num_mb):
            cap = self.state.attn_backends[mb].capture
            r0, r1 = mb * mb_bs, (mb + 1) * mb_bs
            cap.seq_lens[:mb_bs].copy_(seq_lens[r0:r1])
            cap.cu_seqlens_k[0].zero_()
            torch.cumsum(seq_lens[r0:r1], dim=0, dtype=cap.cu_seqlens_k.dtype, out=cap.cu_seqlens_k[1 : mb_bs + 1])
            cap.page_table[:mb_bs, :cols].copy_(block_table[r0:r1])


__all__ = ["AfdAgVerifyGraphRunner"]
