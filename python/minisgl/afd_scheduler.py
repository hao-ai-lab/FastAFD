from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from minisgl.core import Batch, Req
from minisgl.kvcache import MatchResult
from minisgl.kvcache.base import BasePrefixCache
from minisgl.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache, RadixTreeNode, _get_key_fn
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.message import DetokenizeMsg, UserMsg
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import _complete_batch_results
from minisgl.scheduler.table import TableManager
from minisgl.utils import div_ceil

from .afd_protocol import AfdModelPlan
from .spec.config import afd_spec_config, unpack_spec_reply
from .afd_support import select_decode_graph_bs

# MINISGL_AFD_PAGE_FAST=1: the naive page ledger keeps page ids only (an O(1) FIFO
# free queue, per-slot page lists, the pages not yet granted) and fills the CPU page
# table with one vectorised write per allocate_paged call; page_grants hands out the
# ids just allocated instead of re-reading the table. Off (the default) = the
# original per-request torch path. Same page ids, table and grants either way.
PAGE_FAST_ENV = "MINISGL_AFD_PAGE_FAST"


def page_fast_from_env() -> bool:
    return os.environ.get(PAGE_FAST_ENV, "0") == "1"


@dataclass(frozen=True)
class AfdBatchCompletion:
    replies: list[DetokenizeMsg]
    freed_table_indices: tuple[int, ...]
    reply_uids: list[int] | None = None
    reply_next_tokens: list[int] | None = None
    reply_finished: list[bool] | None = None


class _CpuNaiveCacheManager:
    """Coordinator-side KV page ledger for centralized AFD scheduling.

    This intentionally implements only the small CacheManager surface used by
    PrefillManager/_complete_batch_results, and only for cache_type='naive'.
    GPU page-table writes are returned as compact deltas for attention workers.
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        *,
        page_fast: bool = False,
    ):
        self.num_pages = int(num_pages)
        self.page_size = int(page_size)
        self.page_table = page_table
        # MINISGL_AFD_PAGE_FAST (CentralizedAfdDpScheduler decides; only this class,
        # never a subclass: they slice the free list as a Python list).
        self._page_fast = bool(page_fast) and type(self) is _CpuNaiveCacheManager
        if self._page_fast:
            # vLLM's FreeKVCacheBlockQueue.popleft_n / append_n (v1/core/kv_cache_utils.py)
            # in its C form: the same FIFO order as the list below, O(1) per page.
            self.free_slots = deque(range(0, self.num_pages * self.page_size, self.page_size))
            # Page ids (token slot of each page's first token) each table slot holds,
            # columns [0, n) of its row; None once the row is not that (never expected).
            self._slot_pages: dict[int, list[int] | None] = {}
            # Pages allocated but not yet in a step plan's grants: table -> (first
            # page column, page ids); page_grants pops them (vLLM: the scheduler hands
            # the new block ids of the step to the workers, no table re-read).
            self._ungranted: dict[int, tuple[int, list[int]]] = {}
            table_np = self._table_np = page_table.numpy()  # shares the CPU table's memory
            # The same memory as [row, page column, offset in page] over the whole pages
            # of a row (a write past them raises, as the per-request slice write does).
            self._table_pages_np = np.lib.stride_tricks.as_strided(
                table_np,
                shape=(table_np.shape[0], table_np.shape[1] // self.page_size, self.page_size),
                strides=(table_np.strides[0], self.page_size * table_np.strides[1], table_np.strides[1]),
            )
            self._page_offsets_np = np.arange(self.page_size, dtype=np.int64)
        else:
            self.free_slots = list(range(0, self.num_pages * self.page_size, self.page_size))
        # Pages each table slot holds, [0, n). Kept per slot, not on the request:
        # chunked prefill builds a new ChunkedReq for every chunk.
        self._held_pages: dict[int, int] = {}
        self._page_offsets = (
            torch.arange(self.page_size, dtype=torch.int32)
            if self.page_size != 1
            else None
        )

    def match_req(self, req) -> MatchResult:
        if req.input_len <= 0:
            raise RuntimeError("AFD centralized scheduler got empty input request")
        return MatchResult(NaiveCacheHandle())

    @property
    def available_size(self) -> int:
        return len(self.free_slots) * self.page_size

    def lock(self, handle) -> None:
        del handle

    def unlock(self, handle) -> None:
        del handle

    def held_pages(self, table_idx: int) -> int:
        """Pages table slot ``table_idx`` holds: [0, held_pages) of its page-table row."""
        return self._held_pages.get(int(table_idx), 0)

    def allocate_paged(self, reqs: list[Req], lookahead: int = 0) -> None:
        # Pages are handed out in order, so a table slot holds pages [0, held_pages).
        # Speculative decoding rolls cached_len back below what it holds and wants
        # `lookahead` positions past the step allocated, so allocate from what is
        # held rather than from cached_len; a prefill chunk that ends on a page
        # boundary leaves its lookahead page to the next chunk. Without
        # speculation the two agree.
        if self._page_fast:
            self._allocate_paged_fast(reqs, int(lookahead))
            return
        for req in reqs:
            held = self.held_pages(int(req.table_idx))
            first_page = max(div_ceil(req.cached_len, self.page_size), held)
            last_page = div_ceil(req.device_len + int(lookahead), self.page_size)
            needed_pages = last_page - first_page
            if needed_pages <= 0:
                continue
            if needed_pages > len(self.free_slots):
                raise RuntimeError(
                    "AFD centralized naive cache exhausted: "
                    f"needed_pages={needed_pages} free_pages={len(self.free_slots)}"
                )
            pages = self.free_slots[:needed_pages]
            del self.free_slots[:needed_pages]
            values = self._page_to_token_tensor(pages)
            start = first_page * self.page_size
            end = last_page * self.page_size
            self.page_table[int(req.table_idx), start:end] = values
            self._held_pages[int(req.table_idx)] = last_page

    def _allocate_paged_fast(self, reqs: list[Req], lookahead: int) -> None:
        """allocate_paged under MINISGL_AFD_PAGE_FAST: the same pages in the same order,
        written to the CPU table in one vectorised step (vLLM: BlockTable.append_row
        into a numpy table, v1/worker/block_table.py) instead of one torch op per request."""
        page_size = self.page_size
        held_pages = self._held_pages
        slot_pages = self._slot_pages
        ungranted = self._ungranted
        free = self.free_slots
        popleft = free.popleft
        w_rows: list[int] = []  # one entry per allocated page: table row,
        w_cols: list[int] = []  # page column,
        w_pages: list[int] = []  # page id (token slot of the page's first token)
        try:
            for req in reqs:
                table_idx = int(req.table_idx)
                held = held_pages.get(table_idx, 0)
                first_page = max((int(req.cached_len) + page_size - 1) // page_size, held)
                last_page = (int(req.device_len) + lookahead + page_size - 1) // page_size
                needed_pages = last_page - first_page
                if needed_pages <= 0:
                    continue
                if needed_pages > len(free):
                    raise RuntimeError(
                        "AFD centralized naive cache exhausted: "
                        f"needed_pages={needed_pages} free_pages={len(free)}"
                    )
                if needed_pages == 1:  # a decode step crossing a page boundary
                    page = popleft()
                    pages = [page]
                    w_rows.append(table_idx)
                    w_cols.append(first_page)
                    w_pages.append(page)
                else:
                    pages = [popleft() for _ in range(needed_pages)]
                    w_rows.extend([table_idx] * needed_pages)
                    w_cols.extend(range(first_page, last_page))
                    w_pages.extend(pages)
                held_pages[table_idx] = last_page
                row = slot_pages.get(table_idx)
                if row is None and table_idx not in slot_pages:
                    row = slot_pages[table_idx] = []
                if row is not None:
                    if len(row) == first_page:
                        row.extend(pages)
                    else:  # a gap in the row: cache_req reads the table instead
                        slot_pages[table_idx] = None
                entry = ungranted.get(table_idx)
                if entry is None:
                    ungranted[table_idx] = (first_page, pages)  # `pages` is not aliased elsewhere
                elif entry[0] + len(entry[1]) == first_page:
                    entry[1].extend(pages)
                else:  # not contiguous: page_grants reads the table instead
                    ungranted[table_idx] = (-1, [])
        finally:
            if w_pages:
                self._write_table_pages(w_rows, w_cols, w_pages)

    def _write_table_pages(self, rows: list[int], cols: list[int], pages: list[int]) -> None:
        """Token-level table write of page-level entries, one numpy assignment: page p at
        page column c of row r fills tokens c * page_size + k with p + k."""
        page_size = self.page_size
        r = np.asarray(rows, dtype=np.int64)
        p = np.asarray(pages, dtype=np.int64)
        c = np.asarray(cols, dtype=np.int64)
        if int(c.max()) * page_size + page_size > self._table_np.shape[1]:
            raise RuntimeError(
                f"AFD centralized page table row overflow: page column {int(c.max())} "
                f"x page_size {page_size} > row capacity {self._table_np.shape[1]}"
            )
        if page_size == 1:
            self._table_np[r, c] = p
            return
        self._table_pages_np[r, c] = p[:, None] + self._page_offsets_np

    def cache_req(self, req: Req, *, finished: bool) -> None:
        if not finished:
            return
        held_pages = self._held_pages.pop(int(req.table_idx), 0)
        held_len = max(int(req.cached_len), held_pages * self.page_size)
        if self._page_fast:
            row = self._slot_pages.pop(int(req.table_idx), None)
            self._ungranted.pop(int(req.table_idx), None)
            if held_len <= 0:
                return
            if row is not None and len(row) == held_pages and held_len == held_pages * self.page_size:
                # page_table[row, :held_len:page_size], without reading the table
                self.free_slots.extend(row)
                return
        if held_len <= 0:
            return
        page_tokens = self.page_table[int(req.table_idx), :held_len]
        self._free(page_tokens)

    def check_integrity(self) -> None:
        if len(self.free_slots) > self.num_pages:
            raise RuntimeError("AFD centralized naive cache has too many free pages")

    @contextmanager
    def lazy_free_region(self):
        yield

    def _free(self, token_indices: torch.Tensor) -> None:
        if int(token_indices.numel()) == 0:
            return
        self.free_slots.extend(int(x) for x in token_indices[:: self.page_size].tolist())

    def _page_to_token_tensor(self, pages: list[int]) -> torch.Tensor:
        page_tensor = torch.tensor(pages, dtype=torch.int32)
        if self.page_size == 1:
            return page_tensor
        assert self._page_offsets is not None
        return (page_tensor.unsqueeze(1) + self._page_offsets).flatten()


class _CpuRadixPrefixCache(RadixPrefixCache):
    """The radix prefix cache (minisgl.kvcache.radix_cache) on the coordinator's CPU
    page ledger. RadixPrefixCache.__init__ reads the page size from the global context,
    which the coordinator does not set up, so this one takes it."""

    def __init__(self, page_size: int):
        BasePrefixCache.__init__(self)
        self.page_size = int(page_size)
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected


def _probe_prefix_len(cache, input_ids: torch.Tensor) -> int:
    """How many tokens of `input_ids` the radix cache holds, without touching it (no
    node split, no LRU timestamp): for choosing a DP group before admitting."""
    from minisgl.utils import align_down

    node, matched, n = cache.root_node, 0, len(input_ids)
    while matched < n:
        child = node.children.get(cache.key_fn(input_ids[matched:]))
        if child is None:
            break
        step = align_down(child.get_match_len(input_ids[matched:]), cache.page_size)
        matched += step
        if step != child.length:
            break
        node = child
    return matched


class _CpuRadixCacheManager(_CpuNaiveCacheManager):
    """Coordinator-side ledger with radix prefix caching (MINISGL_AFD_PREFIX_CACHE=radix).

    The same surface as the serve CacheManager (minisgl.scheduler.cache) on CPU lists:
    a request's matched prefix pages are copied into its table row by PrefillManager,
    finished (and freshly prefilled) requests are inserted, and zero-ref leaves are
    evicted LRU-first when pages run out. Workers only ever see page ids in the step
    plans' grants, so sharing and eviction stay here. Decode does not append tokens to
    `req.input_ids` (see _complete_decode_results_fast); it keeps them in
    `req.afd_out_ids`, from which a finished request's generated part is cached too.
    """

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor):
        super().__init__(num_pages, page_size, page_table)
        self.prefix_cache = _CpuRadixPrefixCache(page_size)

    def _handle(self, handle):
        if isinstance(handle, RadixCacheHandle):
            return handle
        return RadixCacheHandle(0, self.prefix_cache.root_node)

    def match_req(self, req) -> MatchResult:
        if req.input_len <= 0:
            raise RuntimeError("AFD centralized scheduler got empty input request")
        return self.prefix_cache.match_prefix(req.input_ids[: req.input_len - 1])

    def probe(self, input_ids: torch.Tensor) -> int:
        return _probe_prefix_len(self.prefix_cache, input_ids[: max(0, len(input_ids) - 1)])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle) -> None:
        self.prefix_cache.lock_handle(self._handle(handle), unlock=False)

    def unlock(self, handle) -> None:
        self.prefix_cache.lock_handle(self._handle(handle), unlock=True)

    def allocate_paged(self, reqs: list[Req]) -> None:
        needed = sum(
            max(0, div_ceil(r.device_len, self.page_size) - div_ceil(r.cached_len, self.page_size)) for r in reqs
        )
        if needed > len(self.free_slots):
            short = (needed - len(self.free_slots)) * self.page_size
            evicted = self.prefix_cache.evict(min(short, self.prefix_cache.size_info.evictable_size))
            self.free_slots.extend(int(x) for x in evicted[:: self.page_size].tolist())
        super().allocate_paged(reqs)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        table_idx = int(req.table_idx)
        ids = req.input_ids
        out = getattr(req, "afd_out_ids", None)
        if out:
            ids = torch.cat([ids, torch.tensor(out, dtype=ids.dtype)])
        n = min(int(req.cached_len), int(len(ids)))  # positions whose KV is written
        pages = self.page_table[table_idx, : int(req.cached_len)]
        old = self._handle(req.cache_handle)
        in_tree, new = self.prefix_cache.insert_prefix(ids[:n], pages[:n])
        self.unlock(old)
        self._free(pages[old.cached_len : in_tree])  # another request cached these first
        if finished:
            self._free(pages[new.cached_len :])
            self._held_pages.pop(table_idx, None)
        else:
            req.cache_handle = new
            self.lock(new)

    def check_integrity(self) -> None:
        cached = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cached > self.num_pages:
            raise RuntimeError("AFD radix ledger holds more pages than the pool")


def _compute_microbatch_offsets(
    batch: Batch,
    num_mb: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split one local DP batch into request and flat-token microbatch offsets."""
    num_mb = max(1, int(num_mb))
    padded_reqs = batch.padded_reqs
    size = len(padded_reqs)
    if batch.phase == "decode":
        if num_mb == 1 or size < num_mb:
            return (0, size), (0, size)
        base = size // num_mb
        rem = size % num_mb
        offsets = [0]
        for mb in range(num_mb):
            offsets.append(offsets[-1] + base + (1 if mb < rem else 0))
        offsets_t = tuple(offsets)
        return offsets_t, offsets_t

    extend_lens = [int(req.extend_len) for req in padded_reqs]
    total_tokens = sum(extend_lens)
    if num_mb == 1 or size < num_mb:
        return (0, size), (0, total_tokens)

    cumsum = [0]
    for length in extend_lens:
        cumsum.append(cumsum[-1] + length)
    req_offsets = [0]
    token_offsets = [0]
    for mb in range(1, num_mb):
        target = total_tokens * mb // num_mb
        lo = req_offsets[-1] + 1
        hi = size - (num_mb - mb)
        best = lo
        best_delta = abs(cumsum[lo] - target)
        for idx in range(lo + 1, hi + 1):
            delta = abs(cumsum[idx] - target)
            if delta >= best_delta:
                break
            best = idx
            best_delta = delta
        req_offsets.append(best)
        token_offsets.append(cumsum[best])
    req_offsets.append(size)
    token_offsets.append(total_tokens)
    return tuple(req_offsets), tuple(token_offsets)


class CentralizedAfdDpScheduler:
    """One coordinator-owned CPU scheduler state for one attention DP."""

    def __init__(
        self,
        *,
        dp_rank: int,
        max_running_req: int,
        max_seq_len: int,
        num_pages: int,
        page_size: int,
        prefill_budget: int,
        decode_graph_bs: tuple[int, ...],
        cuda_graph_max_bs: int,
        afd_num_mb: int,
        mlp_fanout: int,
        eos_token_id: int,
        prefix_cache: str = "naive",
        page_fast: bool | None = None,
    ):
        self.max_running_req = int(max_running_req)
        self.max_seq_len = int(max_seq_len)
        self.page_size = int(page_size)
        self.prefill_budget = int(prefill_budget)
        self.decode_graph_bs = tuple(int(x) for x in decode_graph_bs)
        self.decode_graph_bucket_set = set(self.decode_graph_bs)
        self.cuda_graph_max_bs = int(cuda_graph_max_bs)
        self.afd_num_mb = max(1, int(afd_num_mb))
        self.mlp_fanout = max(1, int(mlp_fanout))
        self.eos_token_id = int(eos_token_id)
        self.spec = afd_spec_config()
        aligned_max_seq_len = ((self.max_seq_len + 31) // 32) * 32
        self.page_table = torch.zeros(
            (self.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
        )
        self.page_table[self.max_running_req].fill_(int(num_pages) * int(page_size))
        self.table_manager = TableManager(
            self.max_running_req,
            self.page_table,
            with_token_pool=False,
        )
        if prefix_cache not in ("naive", "radix"):
            raise ValueError(f"unknown AFD prefix cache {prefix_cache!r}")
        # MINISGL_AFD_PAGE_FAST: the naive ledger only (radix keeps the original
        # path), and not under speculative decoding (its per-request
        # lookahead allocation and the workers' own allocator stay as they were).
        self.page_fast = (
            bool(page_fast_from_env() if page_fast is None else page_fast)
            and not self.spec.enabled
            and prefix_cache == "naive"
        )
        if prefix_cache == "radix":
            self.cache_manager = _CpuRadixCacheManager(int(num_pages), int(page_size), self.page_table)
        else:
            self.cache_manager = _CpuNaiveCacheManager(
                int(num_pages),
                int(page_size),
                self.page_table,
                page_fast=self.page_fast,
            )
        # A verify step stages block positions past its own block for the draft, so a
        # request can hold up to 2 * block tokens past max_device_len.
        self.decode_manager = DecodeManager(
            int(page_size), extra_reserve=2 * self.spec.block if self.spec.enabled else 0
        )
        self.prefill_manager = PrefillManager(
            self.cache_manager,  # type: ignore[arg-type]
            self.table_manager,
            self.decode_manager,
        )
        self.finished_reqs: set[Req] = set()
        # pages each table's attention workers hold (explicit paging, page_grants)
        self._granted_pages: dict[int, int] = {}
        self.prefix_hit_tokens = 0  # prompt tokens served from the prefix cache
        self._dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32),
            table_idx=self.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=NaiveCacheHandle(),
        )
        # Verify-step padding: a dummy request carries a whole block, so every
        # padded slot of a verify step has the same token count (graph shapes).
        self._verify_dummy_req = Req(
            input_ids=torch.zeros(max(1, self.spec.block), dtype=torch.int32),
            table_idx=self.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=NaiveCacheHandle(),
        )

    def verify_slot(self, num_reqs: int) -> int:
        """Per-microbatch request slots of a verify step holding ``num_reqs`` requests."""
        per_mb = div_ceil(max(1, int(num_reqs)), self.afd_num_mb)
        if self.decode_graph_bs and per_mb <= max(self.decode_graph_bs):
            return int(select_decode_graph_bs(per_mb, self.decode_graph_bs))
        return int(per_mb)

    def _pad_verify_batch(self, batch: Batch, slot: int) -> int:
        """Pad a verify step to ``slot`` requests per microbatch, real requests first in each."""
        m = self.afd_num_mb
        block = self.spec.block
        real = len(batch.reqs)
        if div_ceil(real, m) > slot:
            raise RuntimeError(f"verify step of {real} requests does not fit {m} x {slot} slots")
        padded: list[Req] = []
        req_offsets = [0]
        cursor = 0
        for mb in range(m):
            count = real // m + (1 if mb < real % m else 0)
            padded.extend(batch.reqs[cursor : cursor + count])
            cursor += count
            padded.extend([self._verify_dummy_req] * (slot - count))
            req_offsets.append(len(padded))
        batch.padded_reqs = padded
        batch.afd_microbatch_offsets = tuple(req_offsets)
        batch.afd_microbatch_token_offsets = tuple(o * block for o in req_offsets)
        return len(padded) * block

    def dummy_verify_step(self, step_id: int, *, slot: int) -> tuple[AfdModelPlan, Batch]:
        """A lane with nothing to verify joins a verify step's collectives with dummies."""
        batch = Batch(reqs=[], phase="prefill")
        batch.afd_spec_verify = True
        graph_token_count = self._pad_verify_batch(batch, int(slot))
        model_plan = AfdModelPlan.from_batch(
            step_id, batch, graph_token_count=graph_token_count
        )
        return model_plan, batch

    def max_concurrency(self, seq_len: int) -> int:
        """Requests of ``seq_len`` tokens this DP holds at once: pages bound it by length
        and table slots by count. vLLM logs the same quantity as "Maximum concurrency
        for N tokens per request" (v1/core/kv_cache_utils.py,
        get_max_concurrency_for_kv_cache_config).
        """
        pages_per_req = div_ceil(int(seq_len), self.cache_manager.page_size)
        return min(self.cache_manager.num_pages // pages_per_req, self.max_running_req)

    def add_reqs(self, msgs: list[UserMsg]) -> None:
        self.prefill_manager.add_reqs(msgs)

    def prefix_match_len(self, input_ids: torch.Tensor) -> int:
        """Tokens of a prompt this DP's prefix cache holds (0 without one)."""
        probe = getattr(self.cache_manager, "probe", None)
        return int(probe(input_ids)) if probe is not None else 0

    def abort_req(self, uid: int) -> tuple[int, ...]:
        req_to_free = self.prefill_manager.abort_req(int(uid))
        req_to_free = req_to_free or self.decode_manager.abort_req(int(uid))
        if req_to_free is None:
            return ()
        table_idx = int(req_to_free.table_idx)
        self._free_req_resources(req_to_free)
        return (table_idx,)

    def has_runnable_reqs(self) -> bool:
        return bool(self.prefill_manager.runnable or self.decode_manager.runnable)

    def describe_waiting_admission(self) -> str | None:
        """The page-ledger numbers behind this lane's head waiting request, or None.

        Admission (PrefillAdder) takes a request only if its prompt plus
        max_tokens fits the free pages after reserving every running request's
        remaining tokens, in FIFO order; these are the numbers it compares.
        """
        pending = self.prefill_manager.pending_list
        if not pending:
            return None
        head = pending[0]
        return (
            f"waiting={len(pending)} running={len(self.decode_manager.running_reqs)} "
            f"free_tables={self.table_manager.available_size} "
            f"head_request_tokens={head.input_len + head.output_len} "
            f"reserved_running_tokens={self.decode_manager.inflight_tokens} "
            f"free_tokens={self.cache_manager.available_size} "
            f"pool_pages={self.cache_manager.num_pages} "
            f"page_size={self.cache_manager.page_size}"
        )

    def waiting_head_exceeds_pool(self) -> bool:
        """True if the head waiting request does not fit even the whole page pool."""
        pending = self.prefill_manager.pending_list
        if not pending or pending[0].chunked_req is not None:
            return False
        head = pending[0]
        need = head.input_len + head.output_len + self.decode_manager.inflight_tokens
        return need > self.cache_manager.num_pages * self.cache_manager.page_size

    def schedule_step(
        self,
        step_id: int,
        *,
        phase: str,
        verify_slot: int | None = None,
    ) -> tuple[AfdModelPlan, Batch] | None:
        if phase == "prefill":
            batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        elif phase == "decode":
            batch = self.decode_manager.schedule_next_batch()
        else:
            raise RuntimeError(f"Unsupported AFD schedule phase: {phase}")
        if batch is None:
            return None
        if phase == "decode" and self.spec.enabled:
            # Verify the bonus + draft block of every running request: an extend of
            # spec.block tokens on top of the committed KV (see spec.config).
            for req in batch.reqs:
                req.device_len = int(req.cached_len) + self.spec.block
            batch.phase = "prefill"
            batch.afd_spec_verify = True
            slot = verify_slot if verify_slot is not None else self.verify_slot(len(batch.reqs))
            graph_token_count = self._pad_verify_batch(batch, int(slot))
        else:
            graph_token_count = self._pad_and_split_batch(batch)
        if self.spec.enabled:
            # Under the overlap loop cached_len lags the steps still in flight, each
            # of which may advance the request by a block: allocate past them too.
            for req in batch.reqs:
                inflight = int(getattr(req, "afd_spec_inflight", 0))
                self.cache_manager.allocate_paged(
                    [req], lookahead=self.spec.lookahead * (1 + inflight)
                )
        else:
            self.cache_manager.allocate_paged(batch.reqs)
        plan = AfdModelPlan.from_batch(
            int(step_id),
            batch,
            graph_token_count=graph_token_count,
        )
        return plan, batch

    def dummy_decode_step(
        self,
        step_id: int,
        *,
        padded_size: int,
    ) -> tuple[AfdModelPlan, Batch]:
        """Build a no-result AG participant so collective steps include every DP.

        The batch has zero real requests, but padded dummy decode tokens make the
        AG worker run the same MoE collective path and return an empty token list.
        """
        padded_size = max(self.afd_num_mb, int(padded_size))
        batch = Batch(reqs=[], phase="decode")
        batch.padded_reqs = [self._dummy_req] * padded_size
        req_offsets, tok_offsets = _compute_microbatch_offsets(
            batch,
            self.afd_num_mb,
        )
        batch.afd_microbatch_offsets = req_offsets
        batch.afd_microbatch_token_offsets = tok_offsets
        graph_token_count = int(len(batch.padded_reqs))
        plan = AfdModelPlan.from_batch(
            int(step_id),
            batch,
            graph_token_count=graph_token_count,
        )
        return plan, batch

    def complete_results(self, batch: Batch, next_tokens: list[int]) -> AfdBatchCompletion:
        if getattr(batch, "afd_spec_verify", False):
            return self._complete_spec_results(batch, next_tokens)
        if batch.is_decode:
            return self._complete_decode_results_fast(batch, next_tokens)
        replies, new_finished = _complete_batch_results(
            batch=batch,
            next_tokens_cpu=torch.tensor(next_tokens, dtype=torch.int32),
            eos_token_id=self.eos_token_id,
            finished_reqs=self.finished_reqs,
            decode_manager=self.decode_manager,
            cache_manager=self.cache_manager,  # type: ignore[arg-type]
            free_req_resources=self._free_req_resources,
        )
        self.finished_reqs = new_finished
        return AfdBatchCompletion(
            replies=replies,
            freed_table_indices=tuple(int(req.table_idx) for req in new_finished),
        )

    def _complete_decode_results_fast(
        self,
        batch: Batch,
        next_tokens: list[int],
    ) -> AfdBatchCompletion:
        """Complete an AFD decode step without per-token CPU tensor append.

        AFD advances `cached_len/device_len` before launching the AG command.
        The decode completion path only needs to emit the sampled token and
        remove finished requests. Keeping a full CPU `req.input_ids` by
        torch.cat-ing one token per request is O(context_len) work per token and
        dominates multi-node coordinator time.
        """
        reply_uids: list[int] = []
        reply_next_tokens: list[int] = []
        reply_finished: list[bool] = []
        new_finished: set[Req] = set()
        eos_token_id = self.eos_token_id
        eos_set = (
            {int(v) for v in eos_token_id}
            if isinstance(eos_token_id, (list, tuple, set))
            else None
        )
        finished_reqs = self.finished_reqs
        decode_manager = self.decode_manager
        free_req_resources = self._free_req_resources
        record_out = isinstance(self.cache_manager, _CpuRadixCacheManager)
        for i, req in enumerate(batch.reqs):
            next_token_id = int(next_tokens[i]) if i < len(next_tokens) else 0
            completed_len = int(getattr(req, "afd_completed_device_len", len(req.input_ids))) + 1
            req.afd_completed_device_len = completed_len
            finished = completed_len >= int(req.max_device_len)
            if not req.sampling_params.ignore_eos:
                if eos_set is not None:
                    finished |= next_token_id in eos_set
                elif eos_token_id is not None:
                    finished |= next_token_id == int(eos_token_id)
            reply_uids.append(int(req.uid))
            reply_next_tokens.append(next_token_id)
            reply_finished.append(bool(finished))
            if record_out:  # the radix cache stores the generated part too
                req.__dict__.setdefault("afd_out_ids", []).append(next_token_id)
            if finished and req not in finished_reqs:
                decode_manager.remove_req(req)
                free_req_resources(req)
                new_finished.add(req)
        self.finished_reqs = new_finished
        return AfdBatchCompletion(
            replies=[],
            freed_table_indices=tuple(int(req.table_idx) for req in new_finished),
            reply_uids=reply_uids,
            reply_next_tokens=reply_next_tokens,
            reply_finished=reply_finished,
        )

    def _complete_spec_results(
        self,
        batch: Batch,
        packed_tokens: list[int],
    ) -> AfdBatchCompletion:
        """Complete a speculative verify step: each request commits n tokens.

        The AG reply carries, per request, the committed tokens (accepted drafts and
        the bonus). They are emitted one by one until EOS or max_tokens; the
        committed KV length advances by n either way, since a finished request's
        pages are freed right here.
        """
        per_req = unpack_spec_reply(packed_tokens, self.spec.block)
        if len(per_req) != len(batch.reqs):
            raise RuntimeError(
                f"spec reply has {len(per_req)} rows for {len(batch.reqs)} requests"
            )
        reply_uids: list[int] = []
        reply_next_tokens: list[int] = []
        reply_finished: list[bool] = []
        new_finished: set[Req] = set()
        eos_token_id = self.eos_token_id
        if isinstance(eos_token_id, (list, tuple, set)):
            eos_set = {int(v) for v in eos_token_id}
        else:
            eos_set = set() if eos_token_id is None else {int(eos_token_id)}
        for req, tokens in zip(batch.reqs, per_req):
            completed_len = int(getattr(req, "afd_completed_device_len", len(req.input_ids)))
            finished = False
            for token in tokens:
                completed_len += 1
                finished = completed_len >= int(req.max_device_len)
                if not req.sampling_params.ignore_eos:
                    finished |= int(token) in eos_set
                reply_uids.append(int(req.uid))
                reply_next_tokens.append(int(token))
                reply_finished.append(bool(finished))
                if finished:
                    break
            req.afd_completed_device_len = completed_len
            req.cached_len = int(req.cached_len) + len(tokens)
            req.device_len = req.cached_len + 1
            req.afd_spec_inflight = max(0, int(getattr(req, "afd_spec_inflight", 1)) - 1)
            if finished and req not in self.finished_reqs:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                new_finished.add(req)
        self.finished_reqs = new_finished
        return AfdBatchCompletion(
            replies=[],
            freed_table_indices=tuple(int(req.table_idx) for req in new_finished),
            reply_uids=reply_uids,
            reply_next_tokens=reply_next_tokens,
            reply_finished=reply_finished,
        )

    def repad_decode_step(
        self,
        step_id: int,
        batch: Batch,
        *,
        padded_size: int,
    ) -> AfdModelPlan:
        """Re-pad a scheduled decode batch up to the collective step's bucket.

        Each DP lane pads to its own decode graph size, but one collective step
        dispatches with a single bucket (the max over lanes). A lane replaying a
        smaller graph would pass DeepEP a different num_max_dispatch_tokens_per_rank
        than its peers, so every decode lane is padded to the shared bucket. Only
        dummy requests are added; KV allocation for real requests is unchanged.
        """
        if str(batch.phase) != "decode":
            raise RuntimeError(f"repad_decode_step expects a decode batch, got {batch.phase!r}")
        graph_token_count = self._pad_and_split_batch(batch, padded_size=int(padded_size))
        return AfdModelPlan.from_batch(
            int(step_id),
            batch,
            graph_token_count=graph_token_count,
        )

    def _pad_and_split_batch(self, batch: Batch, *, padded_size: int | None = None) -> int:
        m = self.afd_num_mb
        batch.afd_microbatch_offsets = None
        batch.afd_microbatch_token_offsets = None
        if padded_size is None:
            padded_size = self._select_padded_size(
                phase=str(batch.phase),
                real_size=int(batch.size),
            )
        padded_size = int(padded_size)
        if padded_size < batch.size:
            raise RuntimeError(
                f"AFD padded_size smaller than batch size: "
                f"padded={padded_size} size={batch.size}"
            )
        balanced_reqs = self._balanced_decode_padded_reqs(batch, padded_size)
        if balanced_reqs is not None:
            batch.padded_reqs = balanced_reqs
        else:
            batch.padded_reqs = (
                batch.reqs + [self._dummy_req] * (padded_size - batch.size)
                if padded_size > batch.size
                else list(batch.reqs)
            )
        req_offsets, tok_offsets = _compute_microbatch_offsets(batch, m)
        batch.afd_microbatch_offsets = req_offsets
        batch.afd_microbatch_token_offsets = tok_offsets
        return int(sum(req.extend_len for req in batch.padded_reqs))

    def _balanced_decode_padded_reqs(
        self,
        batch: Batch,
        padded_size: int,
    ) -> list[Req] | None:
        """Distribute decode real requests across MB graph slots.

        AFD uses multi-MB graph replay. If decode padding is appended only at
        the end, a case like real=64, padded=128, mb=2 becomes mb0=64 real and
        mb1=64 dummy, which destroys overlap analysis and runs one real MB plus
        one empty MB. Keep each MB slice as a real prefix followed by dummy tail.
        """

        if str(batch.phase) != "decode":
            return None
        m = max(1, int(self.afd_num_mb))
        real_size = int(batch.size)
        padded_size = int(padded_size)
        if m <= 1 or padded_size <= real_size:
            return None
        if any(int(req.extend_len) != 1 for req in batch.reqs):
            return None
        base = padded_size // m
        rem = padded_size % m
        slot_sizes = [base + (1 if mb < rem else 0) for mb in range(m)]
        real_base = real_size // m
        real_rem = real_size % m
        real_counts = [real_base + (1 if mb < real_rem else 0) for mb in range(m)]
        if any(real_counts[mb] > slot_sizes[mb] for mb in range(m)):
            return None
        cursor = 0
        padded_reqs: list[Req] = []
        for mb in range(m):
            real_count = int(real_counts[mb])
            slot_size = int(slot_sizes[mb])
            padded_reqs.extend(batch.reqs[cursor : cursor + real_count])
            cursor += real_count
            padded_reqs.extend([self._dummy_req] * (slot_size - real_count))
        return padded_reqs

    def _select_padded_size(self, *, phase: str, real_size: int) -> int:
        if real_size <= 0:
            return 0
        m = self.afd_num_mb
        if phase != "decode" and m > 1:
            return max(int(real_size), int(m))
        if self.mlp_fanout > 1:
            if self.decode_graph_bs and phase == "decode":
                per_mb = (int(real_size) + m - 1) // m
                # Keep attention slots sized by their own per-MB work; MLP graph
                # padding is applied later when materializing model work items.
                try:
                    padded_per_mb = select_decode_graph_bs(per_mb, self.decode_graph_bs)
                    if padded_per_mb in self.decode_graph_bucket_set:
                        return int(padded_per_mb) * m
                except RuntimeError:
                    pass
                padded_per_mb = next(
                    (
                        bucket
                        for bucket in self.decode_graph_bs
                        if bucket >= per_mb
                        and bucket % self.mlp_fanout == 0
                        and (bucket // self.mlp_fanout) in self.decode_graph_bucket_set
                    ),
                    per_mb,
                )
                return int(padded_per_mb) * m
            return int(real_size)
        per_mb = (int(real_size) + m - 1) // m
        if self.decode_graph_bs and phase == "decode" and per_mb <= self.cuda_graph_max_bs:
            return int(select_decode_graph_bs(per_mb, self.decode_graph_bs)) * m
        return int(real_size)

    def page_grants(self, reqs) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """(cached_lens, grant_counts, grant_pages) of `reqs` after allocate_paged: the
        pages of each request's table row the workers do not hold yet (its matched
        prefix included, on a fresh table), so they never allocate on their own.

        Empty under speculative decoding: a verify step
        rolls each length back to what it commits and stages pages a block ahead, which
        the workers' allocator mirrors (spec.config.check_spec_supported keeps prefix
        caching, the one user of grants that the mirror cannot follow, off)."""
        if self.spec.enabled:
            return (), (), ()
        if self.page_fast:
            return self._page_grants_fast(reqs)
        page_size = int(self.page_size)
        cached_lens, counts, pages = [], [], []
        for req in reqs:
            table_idx = int(req.table_idx)
            held = self._granted_pages.get(table_idx, 0)
            if held == 0 and int(req.cached_len) > 0:
                self.prefix_hit_tokens += int(req.cached_len)  # admitted on a prefix hit
            last = div_ceil(int(req.device_len), page_size)
            new = (
                self.page_table[table_idx, held * page_size : last * page_size : page_size].tolist()
                if last > held
                else []
            )
            cached_lens.append(int(req.cached_len))
            counts.append(len(new))
            pages.extend(int(x) for x in new)
            if last > held:
                self._granted_pages[table_idx] = last
        return tuple(cached_lens), tuple(counts), tuple(pages)

    def _page_grants_fast(self, reqs) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """page_grants under MINISGL_AFD_PAGE_FAST: the page ids allocate_paged recorded
        for the columns [granted, last) of a row are handed out as they are; only a row
        whose recorded pages do not cover exactly those columns re-reads the table."""
        page_size = int(self.page_size)
        granted = self._granted_pages
        ungranted = self.cache_manager._ungranted
        page_table = self.page_table
        cached_lens: list[int] = []
        counts: list[int] = []
        pages: list[int] = []
        for req in reqs:
            table_idx = int(req.table_idx)
            cached_len = int(req.cached_len)
            held = granted.get(table_idx, 0)
            if held == 0 and cached_len > 0:
                self.prefix_hit_tokens += cached_len  # admitted on a prefix hit
            last = (int(req.device_len) + page_size - 1) // page_size
            cached_lens.append(cached_len)
            if last <= held:
                counts.append(0)
                continue
            entry = ungranted.pop(table_idx, None)
            if entry is not None and entry[0] == held and len(entry[1]) == last - held:
                new = entry[1]
            else:
                new = page_table[table_idx, held * page_size : last * page_size : page_size].tolist()
            counts.append(len(new))
            pages.extend(new)
            granted[table_idx] = last
        return tuple(cached_lens), tuple(counts), tuple(pages)

    def forget_granted(self, table_idx: int) -> None:
        self._granted_pages.pop(int(table_idx), None)

    def _free_req_resources(self, req: Req) -> None:
        self.forget_granted(int(req.table_idx))
        self.table_manager.free(int(req.table_idx))
        self.cache_manager.cache_req(req, finished=True)


__all__ = ["CentralizedAfdDpScheduler"]
