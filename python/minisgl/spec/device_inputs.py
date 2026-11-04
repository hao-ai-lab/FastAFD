"""Device-resident committed lengths for speculative verify steps (MINISGL_AFD_SPEC_DEVICE_INPUTS=1).

Without it, the AG worker retires every verify step before it materializes the next one: the next
block starts after the tokens the previous step committed, and the host learns that count only from
the step's reply. The host work between two verify steps (reply wait, materialize, attention
metadata) then sits on the GPU's critical path, which is the cross-step overlap the decode loop has
and speculation lost.

With it, the verify finish keeps each row's committed length on the device, and the verify graph's
inputs that depend on it (positions, KV write slots, input ids, the attention kernel's sequence
lengths and block tables) are gathered on the device just before replay. The host materializes the
next step from an upper bound (every in-flight block fully accepted), which only sizes the page
allocation, and learns the exact lengths later from the reply tail.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

_ENV = "MINISGL_AFD_SPEC_DEVICE_INPUTS"


def spec_device_inputs_enabled() -> bool:
    return os.environ.get(_ENV, "0").strip() == "1"


@dataclass
class SpecLengths:
    """Host view of each table row's committed length while verify steps are in flight.

    ``exact[t]`` is the committed length the host last learned (from a reply, or the prefill);
    ``inflight[t]`` counts verify steps materialized for row t whose reply has not been applied;
    ``gen[t]`` changes whenever row t is freed, so a late reply of a finished request cannot write
    into the next request that reuses the row.
    """

    block: int
    exact: dict[int, int] = field(default_factory=dict)
    inflight: dict[int, int] = field(default_factory=dict)
    gen: dict[int, int] = field(default_factory=dict)

    def begin_step(self, table: int, fallback_len: int) -> tuple[int, bool, int]:
        """Materialize one verify step for ``table``.

        Returns (start upper bound, whether the host knows the exact start, row generation).
        ``fallback_len`` is the host's length when the row has no verify history (just prefilled).
        """
        table = int(table)
        if table not in self.exact:
            self.exact[table] = int(fallback_len)
        n = self.inflight.get(table, 0)
        start = self.exact[table] + self.block * n
        self.inflight[table] = n + 1
        return start, n == 0, self.gen.get(table, 0)

    def finish_step(self, table: int, gen: int, committed: int) -> bool:
        """Apply a verify reply: ``committed`` is the row's absolute length after that step."""
        table = int(table)
        if self.gen.get(table, 0) != int(gen) or table not in self.inflight:
            return False   # the row was freed (and maybe reused) since the step was materialized
        self.exact[table] = int(committed)
        n = self.inflight[table] - 1
        if n > 0:
            self.inflight[table] = n
        else:
            self.inflight.pop(table)
        return True

    def free(self, table: int) -> None:
        table = int(table)
        self.exact.pop(table, None)
        self.inflight.pop(table, None)
        self.gen[table] = self.gen.get(table, 0) + 1


def verify_inputs_from_committed(
    committed: torch.Tensor,
    slot_tables: torch.Tensor,
    block: int,
    token_table: torch.Tensor,
    token_pool: torch.Tensor,
    page_size: int,
    table_cols: int,
):
    """The length-dependent inputs of a verify step, from the device committed lengths.

    committed: [rows] int32, the committed length of every table row (the dummy row holds 0).
    slot_tables: [S] int64, the table row of each request slot (padding slots: the dummy row).
    token_table: [rows, max_len] int32 token-level page table (KV slot of each position).
    token_pool: [rows, max_len] int32 token ids.
    Returns (positions [S*block] int32, out_loc [S*block] int32, input_ids [S*block] int32,
    seq_first [S] int32, seq_lens [S] int32, block_table [S, table_cols] int32 in pages).
    """
    start = committed.index_select(0, slot_tables)                     # [S] int32
    off = torch.arange(block, device=committed.device, dtype=torch.int32)
    positions = (start.unsqueeze(1) + off.unsqueeze(0)).reshape(-1)    # [S*block]
    rows = slot_tables.repeat_interleave(block)
    pos64 = positions.to(torch.int64)
    out_loc = token_table[rows, pos64]
    input_ids = token_pool[rows, pos64]
    seq_lens = start + block
    cols = torch.arange(table_cols, device=committed.device, dtype=torch.int64) * page_size
    block_table = token_table[slot_tables.unsqueeze(1), cols.unsqueeze(0)]
    if page_size > 1:
        block_table = torch.div(block_table, page_size, rounding_mode="floor")
    return positions, out_loc.to(torch.int32), input_ids.to(torch.int32), start, seq_lens, block_table.to(torch.int32)


def commit_lengths(committed: torch.Tensor, slot_tables: torch.Tensor, start: torch.Tensor, dummy_row: int) -> None:
    """Write each real slot's new committed length; padding slots (the dummy row) keep 0."""
    val = torch.where(slot_tables == int(dummy_row), torch.zeros_like(start), start).to(committed.dtype)
    committed.scatter_(0, slot_tables, val)


__all__ = ["SpecLengths", "commit_lengths", "spec_device_inputs_enabled", "verify_inputs_from_committed"]
