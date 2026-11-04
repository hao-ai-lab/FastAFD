"""AG plan materialization for the AFD attention worker.

This file intentionally does not run the model or own the hot RPC loop. It owns
the attention-side mutable request state that EG ranks do not have: token ids,
KV page allocation, cached Req objects, pinned H2D staging buffers, attention
metadata, and sampling/writeback metadata. The worker consumes the materialized
Batch and handles CUDA graph/eager execution.
"""
from __future__ import annotations

import os

from typing import Any

import numpy as np
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kernel.vllm_state_index import decode_state_indices
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.scheduler.config import SchedulerConfig
from minisgl.utils import div_ceil, init_logger, nvtx_label, nvtx_range

from .afd_protocol import AfdAGStepPlan
from .spec.config import AfdSpecConfig, afd_spec_config
from .afd_support import log_line

logger = init_logger(__name__)

# Debug-only integrity check on the AG-local page allocator. Off by default;
# it is O(pages) per allocation and only exists to localize a corruption bug.
_PAGE_AUDIT = os.environ.get("MINISGL_AFD_PAGE_AUDIT", "0") == "1"

_INT32_MAX = 2147483647


def build_granted_page_writes(
    writes: list[tuple[int, int, tuple[int, ...]]], page_size: int, row_stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """(flat int64 index into the page table, int32 token slot) of a step's granted
    pages: `writes` holds (table row, first token column, page ids) per request, and
    page p at token column c fills columns c..c+page_size-1 with p..p+page_size-1,
    which is what _assign_req_pages writes one request at a time."""
    rows: list[int] = []
    cols: list[int] = []
    pages: list[int] = []
    for table_idx, start, req_pages in writes:
        n = len(req_pages)
        if n == 1:
            rows.append(table_idx)
            cols.append(start)
            pages.append(req_pages[0])
        else:
            rows.extend([table_idx] * n)
            cols.extend(range(start, start + n * page_size, page_size))
            pages.extend(req_pages)
    page_np = np.asarray(pages, dtype=np.int64)
    base = np.asarray(rows, dtype=np.int64) * int(row_stride) + np.asarray(cols, dtype=np.int64)
    if page_size == 1:
        idx, vals = base, page_np
    else:
        offsets = np.arange(page_size, dtype=np.int64)
        idx = (base[:, None] + offsets).reshape(-1)
        vals = (page_np[:, None] + offsets).reshape(-1)
    if vals.size and (int(vals.min()) < 0 or int(vals.max()) > _INT32_MAX):
        raise ValueError("Page token offset outside nonnegative int32 range")
    return idx, vals.astype(np.int32)


def apply_granted_page_writes(page_table: torch.Tensor, idx: np.ndarray, vals: np.ndarray) -> None:
    """One copy of a step's granted pages into `page_table` (contiguous int32): on CUDA
    one pinned host buffer [int64 index | int32 slot] -> one async H2D -> one
    index_copy_ on the current stream (vLLM: one commit_block_table copy per step,
    v1/worker/block_table.py), instead of a pinned tensor + H2D + arange + add +
    slice write per request."""
    n = int(vals.size)
    if n == 0:
        return
    flat = page_table.view(-1)
    if not page_table.is_cuda:
        flat.index_copy_(0, torch.from_numpy(idx), torch.from_numpy(vals))
        return
    host = torch.empty(3 * n, dtype=torch.int32, pin_memory=True)
    pack_granted_page_writes(host.numpy(), idx, vals)
    index_copy_packed(flat, host.to(page_table.device, non_blocking=True), n)


def pack_granted_page_writes(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """`out` (int32, 3 * n): the n int64 indices in its first 2 * n words, then the n slots."""
    n = int(vals.size)
    out[: 2 * n].view(np.int64)[:] = idx
    out[2 * n : 3 * n] = vals


def index_copy_packed(flat: torch.Tensor, packed: torch.Tensor, n: int) -> None:
    """flat[idx] = vals from a pack_granted_page_writes buffer, on its device: one kernel."""
    flat.index_copy_(0, packed[: 2 * n].view(torch.int64), packed[2 * n : 3 * n])


class AfdAttentionRuntime:
    """
    AFD attention-side plan materializer and request-state owner.

    The coordinator owns request scheduling. Attention workers only apply
    coordinator-provided plans to local GPU state and build attention metadata
    for eager/graph execution.
    """

    # Speculative decoding config; __init__ reads MINISGL_AFD_SPEC.
    spec = AfdSpecConfig()
    # MINISGL_AFD_PAGE_FAST=1 (read by __init__): a step's granted pages are written to
    # the page table in one batched copy (apply_granted_page_writes), not per request.
    _page_fast = False

    def __init__(self, config: SchedulerConfig, state: Any):
        self._page_fast = os.environ.get("MINISGL_AFD_PAGE_FAST", "0") == "1"
        self.config = config
        self.page_table = state.page_table
        self.spec = afd_spec_config()
        self._mapping_uniform_len = 0
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
        # MINISGL_AFD_SPEC_DEVICE_INPUTS (spec.device_inputs): set by AfdSpecAG. The host's view of each
        # row's committed length while verify steps are in flight, and the device copy the verify
        # graph gathers its inputs from.
        self._spec_lengths = None
        self._spec_committed_dev = None
        self._spec_init: list[tuple[int, int]] = []
        self._table_pages: dict[int, list[int]] = {}
        self._granted_tables: set[int] = set()  # tables paged by the coordinator
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
        # Verify-step padding: one block of tokens on the dummy row (its page table
        # row maps every position to the dummy page), mirroring the coordinator.
        self._verify_dummy_req = Req(
            input_ids=torch.zeros(max(1, self.spec.block), dtype=torch.int32),
            table_idx=dummy_table_idx,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
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
        self._last_writeback_host = (wb_table, wb_pos, wb_batch)
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

    def materialize_ag_plan(self, plan: AfdAGStepPlan, *, kv_only: bool = False) -> Batch:
        """AG-side materialize: KV metadata (reused) + token_pool input_ids + sampling.

        Applies the coordinator plan to local token_pool/page-table state, then
        builds the Batch fields consumed by attention, lm_head, and writeback.
        `kv_only` (a CAD attention rank, which runs only the KV side of the step)
        skips the input ids and the sampling metadata.
        """
        with nvtx_range(nvtx_label("AFD_AG_Materialize", step=plan.step_id)):
            with torch.cuda.stream(self.stream):
                self._free_tables(plan.free_table_indices)
                self._write_token_blocks_ag(plan.token_blocks)
                real_reqs, padded_reqs = self._materialize_reqs(plan)
                batch = Batch(reqs=real_reqs, phase=plan.phase)
                batch.padded_reqs = padded_reqs
                batch.afd_spec_verify = bool(getattr(plan, "spec_verify", False))
                batch.afd_microbatch_offsets = plan.microbatch_offsets
                batch.afd_microbatch_token_offsets = plan.microbatch_token_offsets
                stage_index, table_indices, pos_i64 = self._attach_input_mapping(batch, plan)
                # a CAD attention rank does not embed: same-length placeholder, never read
                batch.input_ids = batch.positions if kv_only else self.token_pool[table_indices, pos_i64]
                self._attach_attention_metadata(batch, plan)
                self._record_input_mapping_stage(stage_index)
                if not kv_only:
                    self._attach_sampling_metadata(batch, plan)
        return batch

    def _materialize_reqs(self, plan: AfdAGStepPlan) -> tuple[list[Req], list[Req]]:
        if plan.grant_counts and self._page_fast:
            # explicit paging, every request's checks and bookkeeping as below, the
            # table writes of the whole step at once
            real_reqs, offset, writes = [], 0, []
            grant_pages = plan.grant_pages
            for table_idx, extend_len, cached_len, count in zip(
                plan.table_indices, plan.extend_lens, plan.cached_lens, plan.grant_counts, strict=True
            ):
                pages = grant_pages[offset : offset + count] if count else ()
                offset += count
                real_reqs.append(
                    self._make_stateful_req(
                        table_idx, extend_len, cached_len=cached_len, pages=pages, page_writes=writes
                    )
                )
            self._flush_granted_page_writes(writes)
            return real_reqs, self._build_padded_reqs(plan, real_reqs)
        if plan.grant_counts:  # explicit paging: the coordinator's page ids
            real_reqs, offset = [], 0
            for table_idx, extend_len, cached_len, count in zip(
                plan.table_indices, plan.extend_lens, plan.cached_lens, plan.grant_counts, strict=True
            ):
                pages = plan.grant_pages[offset : offset + count]
                offset += count
                real_reqs.append(
                    self._make_stateful_req(table_idx, extend_len, cached_len=cached_len, pages=pages)
                )
            return real_reqs, self._build_padded_reqs(plan, real_reqs)
        spec_step = bool(getattr(plan, "spec_verify", False)) and getattr(self, "_spec_lengths", None) is not None
        extra = {"spec_step": True} if spec_step else {}
        real_reqs = [
            self._make_stateful_req(table_idx, extend_len, **extra)
            for table_idx, extend_len in zip(
                plan.table_indices,
                plan.extend_lens,
                strict=True,
            )
        ]
        if getattr(self, "_spec_init", None):
            # Rows with no verify step in flight: seed the device lengths from the host (after a
            # prefill, or the first step of a reused row), on the materialize stream the verify
            # replay waits for.
            tables, starts = zip(*self._spec_init)
            self._spec_init = []
            dev = self._spec_committed_dev.device
            idx = torch.tensor(tables, dtype=torch.int64, pin_memory=True).to(dev, non_blocking=True)
            val = torch.tensor(starts, dtype=torch.int32, pin_memory=True).to(dev, non_blocking=True)
            self._spec_committed_dev.index_copy_(0, idx, val)
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
        # int32 under MINISGL_QWEN35_VLLM_KERNELS for decode (vLLM's slot-id dtype; kernel/vllm_state_index.py); the request-table
        # rows below (afd_req_table_indices_gpu) and every prefill / verify batch keep int64
        batch.state_indices = decode_state_indices(table_indices) if plan.phase == "decode" else table_indices
        if plan.phase == "decode":
            batch.afd_req_table_indices_gpu = table_indices
        elif getattr(plan, "spec_verify", False) and self._mapping_uniform_len:
            # One table slot per request: the attention backend then gathers its
            # block table in one launch instead of stacking per-request slices.
            batch.afd_req_table_indices_gpu = table_indices[:: self._mapping_uniform_len]
        pos_i64 = batch.positions.to(torch.int64)
        batch.out_loc = self.page_table[table_indices, pos_i64]
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
        real_index: list[int] = []
        if getattr(plan, "spec_verify", False):
            # Padded position -> real request row (len(batch.reqs) for padding).
            dummies = (self._verify_dummy_req, self._dummy_req)
            next_real = 0
            for req in batch.padded_reqs:
                if any(req is d for d in dummies):
                    real_index.append(len(batch.reqs))
                else:
                    real_index.append(next_real)
                    next_real += 1
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
            table_rows = getattr(batch, "afd_req_table_indices_gpu", None)
            if table_rows is not None:  # decode: one row per request
                sb.afd_req_table_indices_gpu = table_rows[rs:re]
            if getattr(plan, "spec_verify", False):
                sb.afd_spec_verify = True
            subs.append(sb)
        if getattr(plan, "spec_verify", False):
            # Slots of the real requests, in request order (the reply rows kept).
            batch.afd_spec_real_slots = [
                pos for pos, r in enumerate(real_index) if r < len(batch.reqs)
            ]
        return subs

    def _attach_sampling_metadata(self, batch: Batch, plan: AfdAGStepPlan) -> None:
        batch.afd_sampling_plan = plan.sampling
        last_indices, num_sampling, writeback = self._build_ag_sample_meta(plan)
        batch.afd_last_indices = last_indices
        batch.afd_num_sampling = num_sampling
        batch.afd_writeback = writeback
        batch.afd_writeback_host = self._last_writeback_host

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
            pages = self._table_pages.pop(table_idx, ())
            # explicitly granted pages belong to the coordinator (and may be shared or
            # cached there): they never enter this worker's own free list
            if table_idx not in self._granted_tables:
                self._free_pages.extend(pages)
            self._granted_tables.discard(table_idx)
            self._table_cached_len.pop(table_idx, None)
            self._table_reqs.pop(table_idx, None)
            if getattr(self, "_spec_lengths", None) is not None:
                self._spec_lengths.free(table_idx)

    def _build_padded_reqs(
        self,
        plan: AfdAGStepPlan,
        real_reqs: list[Req],
    ) -> list[Req]:
        padded_size = int(plan.microbatch_offsets[-1])
        if getattr(plan, "spec_verify", False):
            return self._build_verify_padded_reqs(plan, real_reqs)
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

    def _build_verify_padded_reqs(self, plan: AfdAGStepPlan, real_reqs: list[Req]) -> list[Req]:
        """Each microbatch slice: its real requests, then block-sized dummies."""
        block = self.spec.block
        offsets = tuple(int(x) for x in plan.microbatch_offsets)
        counts = tuple(int(x) // block for x in plan.microbatch_real_token_counts)
        padded: list[Req] = []
        cursor = 0
        for mb, real_count in enumerate(counts):
            span = offsets[mb + 1] - offsets[mb]
            padded.extend(real_reqs[cursor : cursor + real_count])
            cursor += real_count
            padded.extend([self._verify_dummy_req] * (span - real_count))
        if cursor != len(real_reqs) or len(padded) != offsets[-1]:
            raise RuntimeError(
                f"verify padding mismatch: real={len(real_reqs)} placed={cursor} "
                f"padded={len(padded)} offsets={offsets} counts={counts}"
            )
        return padded

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

    def _make_stateful_req(
        self, table_idx: int, extend_len: int, *, cached_len: int | None = None, pages=None, page_writes=None,
        spec_step: bool = False,
    ) -> Req:
        """`pages` (explicit paging): the request's next pages, as granted by the
        coordinator; `cached_len`: its cached length there (a fresh table with a prefix
        hit starts cached, and the grant then begins with the prefix's pages).
        `page_writes` (MINISGL_AFD_PAGE_FAST): collect the table write of `pages` there
        for _flush_granted_page_writes instead of writing it now.
        `spec_step` (device-input verify): the start is an upper bound over the steps still in
        flight (spec.device_inputs.SpecLengths); it only sizes the pages here, the device computes
        the real positions."""
        table_idx = int(table_idx)
        extend_len = int(extend_len)
        held_len = self._table_cached_len.get(table_idx)
        spec_gen = None
        lengths = getattr(self, "_spec_lengths", None)
        if spec_step and lengths is not None and cached_len is None:
            fallback = held_len or 0
            cached_len, exact, spec_gen = lengths.begin_step(table_idx, fallback)
            if exact:  # nothing in flight for this row: the host's length is the device's
                self._spec_init.append((table_idx, cached_len))
        if cached_len is None:
            cached_len = held_len or 0
        else:
            cached_len = int(cached_len)
            if held_len is not None and held_len != cached_len:
                raise RuntimeError(
                    f"AFD paging out of sync: table {table_idx} cached {held_len} here, "
                    f"{cached_len} on the coordinator"
                )
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
        req.afd_spec_gen = spec_gen
        if pages is not None:
            if page_writes is None:
                self._assign_req_pages(req, pages)
            else:
                self._assign_req_pages(req, pages, page_writes=page_writes)
        else:
            # Most steps stay inside the pages a request already holds; the allocator
            # would return without work then, so skip the call.
            held = self._table_pages.get(table_idx)
            if held is None or device_len + self.spec.lookahead > len(held) * int(self.config.page_size):
                self._allocate_req_pages(req)
        self._table_cached_len[table_idx] = device_len
        if _PAGE_AUDIT:
            path = getattr(self, "audit_log_path", None)
            if path:
                log_line(
                    path,
                    f"[afd-audit] stateful_req table_idx={table_idx} "
                    f"cached_len={cached_len} extend_len={extend_len} "
                    f"device_len={device_len} "
                    f"pages_held={len(self._table_pages.get(table_idx, ()))} "
                    f"free_pages={len(self._free_pages)}",
                    flush=True,
                )
        return req

    def _assign_req_pages(self, req: Req, pages, page_writes=None) -> None:
        """Explicit paging: put the coordinator's pages right after the ones this
        table holds; they must cover the request up to its device length.
        `page_writes`: append (row, first token column, pages) there, not written yet."""
        page_size = int(self.config.page_size)
        if int(req.table_idx) in self._table_pages and int(req.table_idx) not in self._granted_tables:
            raise RuntimeError(f"AFD table {int(req.table_idx)} mixes local and granted pages")
        self._granted_tables.add(int(req.table_idx))
        held = self._table_pages.setdefault(int(req.table_idx), [])
        last_page = div_ceil(int(req.device_len), page_size)
        if len(held) + len(pages) != max(last_page, len(held)):
            raise RuntimeError(
                f"AFD page grant mismatch: table {int(req.table_idx)} holds {len(held)} pages, "
                f"granted {len(pages)}, needs {last_page} for device_len {int(req.device_len)}"
            )
        if not pages:
            return
        start = len(held) * page_size
        if page_writes is not None:
            if len(pages) == 1:  # a decode step crossing a page boundary
                held.append(int(pages[0]))
            else:
                held.extend(int(p) for p in pages)
            end = start + len(pages) * page_size
            if end > int(self.page_table.shape[1]):
                raise RuntimeError(f"AFD page table row overflow: table {int(req.table_idx)} slice ({start},{end})")
            page_writes.append((int(req.table_idx), start, pages))
            return
        held.extend(int(p) for p in pages)
        values = self._page_to_token_tensor(list(pages))
        end = start + int(values.numel())
        if end > int(self.page_table.size(1)):
            raise RuntimeError(f"AFD page table row overflow: table {int(req.table_idx)} slice ({start},{end})")
        self.page_table[int(req.table_idx), start:end] = values

    def _flush_granted_page_writes(self, writes) -> None:
        """Write a step's collected grants (_assign_req_pages page_writes) to the page
        table: one batched copy, on the stream the per-request writes would use."""
        if not writes:
            return
        page_size = int(self.config.page_size)
        table = self.page_table
        if not table.is_contiguous() or table.dtype != torch.int32:
            for table_idx, start, pages in writes:  # the per-request write
                values = self._page_to_token_tensor(list(pages))
                table[int(table_idx), start : start + int(values.numel())] = values
            return
        idx, vals = build_granted_page_writes(writes, page_size, int(table.stride(0)))
        apply_granted_page_writes(table, idx, vals)

    def _allocate_req_pages(self, req: Req) -> None:
        page_size = int(self.config.page_size)
        # Mirrors _CpuNaiveCacheManager.allocate_paged: speculative decoding rolls
        # cached_len back below the pages a request holds and stages the draft
        # block past the step, so allocate from the held pages up to the lookahead.
        held = len(self._table_pages.get(int(req.table_idx), ()))
        first_page = max(div_ceil(int(req.cached_len), page_size), held)
        last_page = div_ceil(int(req.device_len) + self.spec.lookahead, page_size)
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
        row_capacity = int(self.page_table.size(1))
        if end > row_capacity:
            raise RuntimeError(
                "AFD attention page table row overflow while materializing plan: "
                f"table_idx={int(req.table_idx)} cached_len={int(req.cached_len)} "
                f"device_len={int(req.device_len)} extend_len={int(req.extend_len)} "
                f"page_size={page_size} first_page={first_page} last_page={last_page} "
                f"slice=({start},{end}) row_capacity={row_capacity} "
                f"values={int(values.numel())}"
            )
        self.page_table[int(req.table_idx), start:end] = values
        if _PAGE_AUDIT:
            self._audit_pages(req)

    def _audit_pages(self, req: Req) -> None:
        """Fail loudly if the local page allocator has double-handed a page.

        Off unless MINISGL_AFD_PAGE_AUDIT=1. Chunked prefill with >= 2 requests per
        attention replica corrupts output while a single chunked request is
        bitwise correct, which is the signature of two table slots sharing a page.
        This turns that hypothesis into an assertion instead of an inference.
        """
        seen: dict[int, int] = {}
        for tid, pages in self._held_page_lists():
            for page in pages:
                prev = seen.get(page)
                if prev is not None:
                    raise RuntimeError(
                        "AFD attention page audit FAILED: page %d is held by table "
                        "slots %d and %d (just allocated for slot %d, cached_len=%d "
                        "device_len=%d)"
                        % (page, prev, tid, int(req.table_idx),
                           int(req.cached_len), int(req.device_len))
                    )
                seen[page] = tid
        dupes = len(self._free_pages) - len(set(self._free_pages))
        if dupes:
            raise RuntimeError(
                f"AFD attention page audit FAILED: free list holds {dupes} duplicate "
                f"page(s) after allocating for slot {int(req.table_idx)}"
            )
        overlap = seen.keys() & set(self._free_pages)
        if overlap:
            raise RuntimeError(
                "AFD attention page audit FAILED: %d page(s) are BOTH free and held, "
                "e.g. %s (slot %d, cached_len=%d device_len=%d)"
                % (len(overlap), sorted(overlap)[:4], int(req.table_idx),
                   int(req.cached_len), int(req.device_len))
            )

    def _held_page_lists(self):
        """(table slot, pages) of every table slot."""
        yield from self._table_pages.items()

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
        stage_index, positions, table_indices = self._claim_input_mapping_stage(needed_size)
        positions_np = positions.numpy()
        table_indices_np = table_indices.numpy()
        # A verify step extends every slot by the same block: one vectorized fill.
        lens = [int(req.device_len) - int(req.cached_len) for req in padded_reqs]
        self._mapping_uniform_len = lens[0] if lens and lens[0] > 1 and lens.count(lens[0]) == len(lens) else 0
        if self._mapping_uniform_len:
            n, length = len(padded_reqs), self._mapping_uniform_len
            cached = np.fromiter((int(req.cached_len) for req in padded_reqs), dtype=np.int32, count=n)
            positions_np[: n * length] = (cached[:, None] + np.arange(length, dtype=np.int32)).reshape(-1)
            table_indices_np[: n * length] = np.repeat(
                np.fromiter((int(req.table_idx) for req in padded_reqs), dtype=np.int32, count=n), length
            )
            return stage_index, positions, table_indices
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
