"""AG plan materialization for the AFD attention worker.

This file intentionally does not run the model or own the hot RPC loop. It owns
the attention-side mutable request state that EG ranks do not have: token ids,
KV page allocation, cached Req objects, pinned H2D staging buffers, attention
metadata, and sampling/writeback metadata. The worker consumes the materialized
Batch and handles CUDA graph/eager execution.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.scheduler.config import SchedulerConfig
from minisgl.utils import div_ceil, init_logger, nvtx_label, nvtx_range

from .afd_protocol import AfdAGStepPlan

logger = init_logger(__name__)


class AfdAttentionRuntime:
    """
    AFD attention-side plan materializer and request-state owner.

    The coordinator owns request scheduling. Attention workers only apply
    coordinator-provided plans to local GPU state and build attention metadata
    for eager/graph execution.
    """

    def __init__(self, config: SchedulerConfig, state: Any):
        self.config = config
        self.state = state
        self.stream = torch.cuda.Stream(device=self.state.device)
        self._free_pages = list(
            range(
                0,
                int(self.state.num_pages) * int(self.config.page_size),
                int(self.config.page_size),
            )
        )
        self._table_cached_len: dict[int, int] = {}
        self._table_pages: dict[int, list[int]] = {}
        self._table_reqs: dict[int, Req] = {}
        self._req_input_ids = torch.empty(1, dtype=torch.int32, device="cpu")

        self._input_mapping_capacity = max(1, int(self.config.max_extend_tokens))
        self._input_mapping_stages = self._make_input_mapping_stages(
            self._input_mapping_capacity
        )
        self._input_mapping_stage_events = [
            torch.cuda.Event() for _ in self._input_mapping_stages
        ]
        self._input_mapping_stage_recorded = [
            False for _ in self._input_mapping_stages
        ]
        self._next_input_mapping_stage = 0

        # Pinned staging ring for the per-step sample/writeback metadata so its H2D
        # copies are ASYNC on the runtime stream (torch.tensor(list, device=cuda)
        # is a SYNCHRONOUS h2d -> a per-decode-step host stall = part of the AG/EG
        # replay gap). The 3-deep ring guards reuse under one-step-ahead overlap.
        _smb = int(self.config.max_running_req) + 1
        self._samp_stages = tuple(
            tuple(torch.empty(_smb, dtype=torch.int64, pin_memory=True) for _ in range(4))
            for _ in range(3)
        )
        self._samp_stage_events = [torch.cuda.Event() for _ in self._samp_stages]
        self._samp_stage_recorded = [False] * len(self._samp_stages)
        self._next_samp_stage = 0

        dummy_table_idx = self.config.max_running_req
        num_tokens = self.state.num_pages * self.config.page_size
        self._dummy_req: Req | None = self.state.install_dummy_req(
            dummy_table_idx,
            num_tokens,
        )

        # AG owns token ids, embedding input, and generated-token writeback for
        # this attention-DP lane only.  Table indices are local to each AG DP,
        # so this must be sized by the local max_running_req, not by global
        # batch or attn_dp_size.
        self.token_pool = torch.zeros(
            (int(config.max_running_req) + 1, int(config.max_seq_len)),
            dtype=torch.int32,
            device=self.state.device,
        )

        logger.info("AfdAttentionRuntime initialized (rank=%d)", config.tp_info.rank)

    # ----- AG-side materialize: KV metadata + token ids + sampling -----

    def _write_token_blocks_ag(self, token_blocks) -> None:
        """Write prompt/generated token ids into the AG-owned token_pool."""
        for block in token_blocks:
            tokens = block.tokens
            if int(tokens.numel()) == 0:
                continue
            if tokens.device != self.token_pool.device:
                tokens = tokens.to(self.token_pool.device, non_blocking=True)
            start = int(block.start_pos)
            self.token_pool[int(block.table_idx), start : start + int(tokens.numel())] = tokens

    def _build_ag_sample_meta(
        self,
        plan: AfdAGStepPlan,
    ) -> tuple[torch.Tensor, int, dict[str, torch.Tensor]]:
        """Sample-ordered last-token indices + writeback indices.

        Mirrors the AG token-pool contract: the sampled-token vector
        is indexed by `sample_index` (dense 0..num_sampling-1). The attention
        tensor layout follows the flattened execution spans in the plan, which
        include graph padding and balanced-MB padding, so last-token positions
        must be derived from that layout rather than from the compact real
        request list.
        """
        last_pos, wb_table, wb_pos, wb_batch = self._collect_ag_sample_meta(plan)
        last_indices, writeback = self._stage_ag_sample_meta(
            last_pos,
            wb_table,
            wb_pos,
            wb_batch,
        )
        return last_indices, len(last_pos), writeback

    def _collect_ag_sample_meta(
        self,
        plan: AfdAGStepPlan,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        sample_idxs = [
            int(sample_index)
            for sample_index in plan.exec_sample_indices
            if int(sample_index) >= 0
        ]
        num_sampling = (max(sample_idxs) + 1) if sample_idxs else 0
        last_pos = [0] * num_sampling
        wb_table: list[int] = []
        wb_pos: list[int] = []
        wb_batch: list[int] = []
        cursor = 0
        for table_idx, start_pos, length, sample_index, writeback in zip(
            plan.exec_table_indices,
            plan.exec_start_positions,
            plan.exec_extend_lens,
            plan.exec_sample_indices,
            plan.exec_writebacks,
            strict=True,
        ):
            length = int(length)
            si = int(sample_index)
            if si >= 0:
                last_pos[si] = cursor + length - 1
                if bool(writeback):
                    wb_table.append(int(table_idx))
                    wb_pos.append(int(start_pos) + length)
                    wb_batch.append(si)
            cursor += length
        return last_pos, wb_table, wb_pos, wb_batch

    def _stage_ag_sample_meta(
        self,
        last_pos: list[int],
        wb_table: list[int],
        wb_pos: list[int],
        wb_batch: list[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Async H2D via the pinned staging ring (replaces synchronous
        # torch.tensor(list, device=cuda) — a per-step host stall on the sched
        # stream). Pick a slot, wait its prior copy (3-deep ring guards reuse under
        # one-step-ahead overlap), fill the pinned buffers on the host, async copy.
        dev = self.state.device
        slot = self._next_samp_stage
        self._next_samp_stage = (self._next_samp_stage + 1) % len(self._samp_stages)
        if self._samp_stage_recorded[slot]:
            self._samp_stage_events[slot].synchronize()
        last_pin, tbl_pin, pos_pin, bat_pin = self._samp_stages[slot]

        def _stage(pin, lst):
            n = len(lst)
            if n == 0:
                return torch.empty((0,), dtype=torch.int64, device=dev)
            pin.numpy()[:n] = lst
            return pin[:n].to(dev, non_blocking=True)

        out = (
            _stage(last_pin, last_pos),
            {
                "table_indices": _stage(tbl_pin, wb_table),
                "write_positions": _stage(pos_pin, wb_pos),
                "batch_indices": _stage(bat_pin, wb_batch),
            },
        )
        self._samp_stage_events[slot].record(self.stream)
        self._samp_stage_recorded[slot] = True
        return out

    def build_sampling_batch(self, size: int) -> Batch:
        """Decode-phase dummy batch for pre-selected sample-order hidden states."""
        if self._dummy_req is None:
            raise RuntimeError("AFD attention dummy request is not initialized")
        reqs = [self._dummy_req] * int(size)
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = reqs
        batch.input_ids = torch.empty((0,), dtype=torch.int32)
        batch.positions = torch.empty((0,), dtype=torch.int32)
        batch.out_loc = torch.empty((0,), dtype=torch.int32)
        return batch

    def materialize_ag_plan(self, plan: AfdAGStepPlan) -> Batch:
        """AG-side materialize: KV metadata (reused) + token_pool input_ids + sampling.

        Applies the coordinator plan to local token_pool/page-table state, then
        builds the Batch fields consumed by attention, lm_head, and writeback.
        """
        with nvtx_range(nvtx_label("AFD_AG_Materialize", step=plan.step_id)):
            with torch.cuda.stream(self.stream):
                self._free_tables(plan.free_table_indices)
                self._write_token_blocks_ag(plan.token_blocks)
                real_reqs, padded_reqs = self._materialize_reqs(plan)
                batch = Batch(reqs=real_reqs, phase=plan.phase)
                batch.padded_reqs = padded_reqs
                batch.afd_microbatch_offsets = plan.microbatch_offsets
                batch.afd_microbatch_token_offsets = plan.microbatch_token_offsets
                stage_index, table_indices, pos_i64 = self._attach_input_mapping(batch, plan)
                batch.input_ids = self.token_pool[table_indices, pos_i64]
                self._attach_attention_metadata(batch, plan)
                self._record_input_mapping_stage(stage_index)
                self._attach_sampling_metadata(batch, plan)
        return batch

    def _materialize_reqs(self, plan: AfdAGStepPlan) -> tuple[list[Req], list[Req]]:
        real_reqs = [
            self._make_stateful_req(table_idx, extend_len)
            for table_idx, extend_len in zip(
                plan.table_indices,
                plan.extend_lens,
                strict=True,
            )
        ]
        return real_reqs, self._build_padded_reqs(plan, real_reqs)

    def _attach_input_mapping(
        self,
        batch: Batch,
        plan: AfdAGStepPlan,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        stage_index, positions_host, table_indices_host = (
            self._build_attention_input_mapping(batch.padded_reqs)
        )
        batch.positions_host = positions_host
        batch.positions = positions_host.to(self.state.device, non_blocking=True)
        table_indices = table_indices_host.to(self.state.device, non_blocking=True)
        batch.state_indices = table_indices
        if plan.phase == "decode":
            batch.afd_req_table_indices_gpu = table_indices
        pos_i64 = batch.positions.to(torch.int64)
        batch.out_loc = self.state.page_table[table_indices, pos_i64]
        return stage_index, table_indices, pos_i64

    def _attach_attention_metadata(self, batch: Batch, plan: AfdAGStepPlan) -> None:
        num_mb = max(1, len(plan.microbatch_offsets) - 1)
        self._prepare_attn_metadata(batch, plan.microbatch_offsets, num_mb)
        batch.afd_mb_subbatches = self._build_microbatch_subbatches(batch, plan, num_mb)

    def _build_microbatch_subbatches(
        self,
        batch: Batch,
        plan: AfdAGStepPlan,
        num_mb: int,
    ) -> list[Batch]:
        # Each sub-batch carries its token slice + metadata so the dual-MB forward
        # can enter ctx.forward_batch per MB.
        tok_offs = plan.microbatch_token_offsets
        req_offs = plan.microbatch_offsets
        subs = []
        for mb in range(num_mb):
            ts, te = int(tok_offs[mb]), int(tok_offs[mb + 1])
            rs, re = int(req_offs[mb]), int(req_offs[mb + 1])
            sb = Batch(reqs=batch.padded_reqs[rs:re], phase=plan.phase)
            sb.padded_reqs = batch.padded_reqs[rs:re]
            sb.positions = batch.positions[ts:te]
            sb.input_ids = batch.input_ids[ts:te]
            sb.out_loc = batch.out_loc[ts:te]
            sb.state_indices = batch.state_indices[ts:te]
            sb.attn_metadata = batch.attn_metadata_mbs[mb]
            subs.append(sb)
        return subs

    def _attach_sampling_metadata(self, batch: Batch, plan: AfdAGStepPlan) -> None:
        batch.afd_sampling_plan = plan.sampling
        last_indices, num_sampling, writeback = self._build_ag_sample_meta(plan)
        batch.afd_last_indices = last_indices
        batch.afd_num_sampling = num_sampling
        batch.afd_writeback = writeback

    def writeback_tokens_ag(self, batch: Batch, next_tokens: torch.Tensor) -> None:
        """Write sampled tokens back into the AG token_pool for the next step."""
        wb = batch.afd_writeback
        if wb is None or int(wb["batch_indices"].numel()) == 0:
            return
        with torch.cuda.stream(self.stream):
            self.token_pool[wb["table_indices"], wb["write_positions"]] = next_tokens[
                wb["batch_indices"]
            ].to(torch.int32)

    def shutdown(self) -> None:
        self.state.shutdown()

    def _free_tables(self, table_indices: tuple[int, ...]) -> None:
        for table_idx in table_indices:
            table_idx = int(table_idx)
            self._free_pages.extend(self._table_pages.pop(table_idx, ()))
            self._table_cached_len.pop(table_idx, None)
            self._table_reqs.pop(table_idx, None)

    def _build_padded_reqs(
        self,
        plan: AfdAGStepPlan,
        real_reqs: list[Req],
    ) -> list[Req]:
        padded_size = int(plan.microbatch_offsets[-1])
        if padded_size <= len(real_reqs):
            return list(real_reqs)
        if self._dummy_req is None:
            raise RuntimeError("AFD attention dummy request is not initialized")

        padded = self._try_build_balanced_decode_padding(plan, real_reqs, padded_size)
        if padded is not None:
            return padded
        padded_reqs = list(real_reqs)
        padded_reqs.extend([self._dummy_req] * (padded_size - len(padded_reqs)))
        return padded_reqs

    def _try_build_balanced_decode_padding(
        self,
        plan: AfdAGStepPlan,
        real_reqs: list[Req],
        padded_size: int,
    ) -> list[Req] | None:
        counts = tuple(int(x) for x in plan.microbatch_real_token_counts)
        offsets = tuple(int(x) for x in plan.microbatch_offsets)
        extend_lens = tuple(int(x) for x in plan.extend_lens)
        if not (
            str(plan.phase) == "decode"
            and len(offsets) > 2
            and len(counts) == len(offsets) - 1
            and sum(counts) == len(real_reqs)
            and len(extend_lens) == len(real_reqs)
            and all(x == 1 for x in extend_lens)
        ):
            return None

        padded_reqs = []
        cursor = 0
        for mb, real_count in enumerate(counts):
            span = int(offsets[mb + 1]) - int(offsets[mb])
            if real_count < 0 or real_count > span or cursor + real_count > len(real_reqs):
                return None
            padded_reqs.extend(real_reqs[cursor : cursor + real_count])
            cursor += real_count
            padded_reqs.extend([self._dummy_req] * (span - real_count))
        if cursor == len(real_reqs) and len(padded_reqs) == padded_size:
            return padded_reqs
        return None

    def _make_stateful_req(self, table_idx: int, extend_len: int) -> Req:
        table_idx = int(table_idx)
        extend_len = int(extend_len)
        cached_len = self._table_cached_len.get(table_idx, 0)
        device_len = cached_len + extend_len
        req = self._table_reqs.get(table_idx)
        if req is None:
            req = Req(
                input_ids=self._req_input_ids,
                table_idx=table_idx,
                cached_len=0,
                output_len=1,
                uid=-1,
                sampling_params=SamplingParams(),
                cache_handle=NaiveCacheHandle(),
            )
            self._table_reqs[table_idx] = req
        req.cached_len = cached_len
        req.device_len = device_len
        req.max_device_len = device_len + 1
        self._allocate_req_pages(req)
        self._table_cached_len[table_idx] = device_len
        return req

    def _allocate_req_pages(self, req: Req) -> None:
        page_size = int(self.config.page_size)
        first_page = div_ceil(int(req.cached_len), page_size)
        last_page = div_ceil(int(req.device_len), page_size)
        needed_pages = last_page - first_page
        if needed_pages <= 0:
            return
        if needed_pages > len(self._free_pages):
            raise RuntimeError(
                "AFD attention local page allocator exhausted: "
                f"needed_pages={needed_pages} free_pages={len(self._free_pages)}"
            )
        pages = self._free_pages[:needed_pages]
        del self._free_pages[:needed_pages]
        self._table_pages.setdefault(int(req.table_idx), []).extend(pages)
        values = self._page_to_token_tensor(pages)
        start = first_page * page_size
        end = last_page * page_size
        row_capacity = int(self.state.page_table.size(1))
        if end > row_capacity:
            raise RuntimeError(
                "AFD attention page table row overflow while materializing plan: "
                f"table_idx={int(req.table_idx)} cached_len={int(req.cached_len)} "
                f"device_len={int(req.device_len)} extend_len={int(req.extend_len)} "
                f"page_size={page_size} first_page={first_page} last_page={last_page} "
                f"slice=({start},{end}) row_capacity={row_capacity} "
                f"values={int(values.numel())}"
            )
        self.state.page_table[int(req.table_idx), start:end] = values

    def _page_to_token_tensor(self, pages: list[int]) -> torch.Tensor:
        page_tensor = torch.tensor(pages, dtype=torch.int32, pin_memory=True).to(
            self.state.device,
            non_blocking=True,
        )
        if int(self.config.page_size) == 1:
            return page_tensor
        offsets = torch.arange(
            int(self.config.page_size),
            dtype=torch.int32,
            device=self.state.device,
        )
        return (page_tensor.unsqueeze(1) + offsets).flatten()

    def _make_input_mapping_stages(
        self,
        capacity: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (
                torch.empty(capacity, dtype=torch.int32, pin_memory=True),
                torch.empty(capacity, dtype=torch.int64, pin_memory=True),
            )
            for _ in range(3)
        )

    def _ensure_input_mapping_capacity(self, needed_size: int) -> None:
        if needed_size <= self._input_mapping_capacity:
            return
        for recorded, event in zip(
            self._input_mapping_stage_recorded,
            self._input_mapping_stage_events,
            strict=True,
        ):
            if recorded:
                event.synchronize()
        self._input_mapping_capacity = max(
            needed_size,
            self._input_mapping_capacity * 2,
        )
        self._input_mapping_stages = self._make_input_mapping_stages(
            self._input_mapping_capacity
        )
        self._input_mapping_stage_recorded = [
            False for _ in self._input_mapping_stages
        ]
        self._next_input_mapping_stage = 0

    def _claim_input_mapping_stage(
        self,
        needed_size: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        self._ensure_input_mapping_capacity(needed_size)
        stage_index = self._next_input_mapping_stage
        self._next_input_mapping_stage = (
            self._next_input_mapping_stage + 1
        ) % len(self._input_mapping_stages)
        if self._input_mapping_stage_recorded[stage_index]:
            self._input_mapping_stage_events[stage_index].synchronize()
        positions, table_indices = self._input_mapping_stages[stage_index]
        return (
            stage_index,
            positions[:needed_size],
            table_indices[:needed_size],
        )

    def _record_input_mapping_stage(self, stage_index: int) -> None:
        self._input_mapping_stage_events[stage_index].record(self.stream)
        self._input_mapping_stage_recorded[stage_index] = True

    def _build_attention_input_mapping(
        self,
        padded_reqs: list[Req],
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        needed_size = sum(
            int(req.device_len) - int(req.cached_len)
            for req in padded_reqs
        )
        stage_index, positions, table_indices = self._claim_input_mapping_stage(
            needed_size
        )
        positions_np = positions.numpy()
        table_indices_np = table_indices.numpy()
        offset = 0
        for req in padded_reqs:
            length = int(req.device_len) - int(req.cached_len)
            if length <= 0:
                continue
            end = offset + length
            cached_len = int(req.cached_len)
            if length == 1:
                positions_np[offset] = cached_len
            else:
                positions_np[offset:end] = np.arange(
                    cached_len,
                    int(req.device_len),
                    dtype=np.int32,
                )
            table_indices_np[offset:end] = int(req.table_idx)
            offset = end
        if offset != needed_size:
            raise RuntimeError(
                "AFD attention mapping size mismatch: "
                f"filled={offset} expected={needed_size}"
            )
        return stage_index, positions, table_indices

    def _prepare_attn_metadata(
        self,
        batch: Batch,
        req_offsets: tuple[int, ...],
        num_mb: int,
    ) -> None:
        """Build one attention metadata object per microbatch slot."""

        backends = self.state.attn_backends
        orig_padded_reqs = batch.padded_reqs
        orig_req_table_indices_gpu = getattr(batch, "afd_req_table_indices_gpu", None)
        mb_metas: list = []
        try:
            for mb in range(num_mb):
                start, end = req_offsets[mb], req_offsets[mb + 1]
                batch.padded_reqs = orig_padded_reqs[start:end]
                if orig_req_table_indices_gpu is not None:
                    batch.afd_req_table_indices_gpu = orig_req_table_indices_gpu[start:end]
                backends[mb].prepare_metadata(batch)
                mb_metas.append(batch.attn_metadata)
        finally:
            batch.padded_reqs = orig_padded_reqs
            if orig_req_table_indices_gpu is not None:
                batch.afd_req_table_indices_gpu = orig_req_table_indices_gpu
        batch.attn_metadata = mb_metas[0]
        batch.attn_metadata_mbs = mb_metas


__all__ = ["AfdAttentionRuntime"]
