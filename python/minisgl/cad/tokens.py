"""CAD token path on the model GPU (TP1): the step's
embedding and its final norm / LM head / greedy sampling run on the model GPU, for the rows of its r attention GPUs.

Rows: one microbatch of a step on model GPU j holds r x B rows (B = the step's bucket): peer p's rows p B .. (p + 1) B
(layout.model_rows); a step's rows are mb-major: row = mb r B + p B + i (row_of). Each real request of the step has one
row; padding rows embed token 0, sit at position 0 and write their sampled token into the dump slot.

Token handoff: every request owns a last-token slot on its model GPU, slot = peer x table_rows + table index (its AG
request-table row on the attention GPU, stable while it lives); the dump slot is the last one. Per row the coordinator
sends the host token (the input token at position KV - 1 where the host has it: the first decode step after an AFD prefill,
whose token the attention GPU sampled, or a reply already applied) or -1, which takes the token the previous CAD step
sampled for the request (last_tok[slot]), so pipelined steps need no host round trip.

Everything per step is fixed-size device tensors (graph-capturable); fill() copies the host lists in with no host sync.
Sampling is greedy (argmax, ties to the smallest index as torch.argmax); a non-greedy request must be refused upstream:
random sampling inside a captured graph would replay the same random numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CadRow:
    """One request of the step on this model GPU."""

    peer: int          # its attention GPU's position among the model GPU's peers
    mb: int
    i: int             # its index in that peer's microbatch (< bucket)
    table_index: int   # its AG request-table row on the attention GPU
    position: int      # the new token's position (its KV tokens before this step's token)
    host_token: int    # >= 0: the input token from the host; -1: the last token a CAD step sampled for it


def row_of(r: int, bucket: int, mb: int, peer: int, i: int) -> int:
    if not (0 <= peer < r and 0 <= i < bucket and mb >= 0):
        raise ValueError(f"row (mb {mb}, peer {peer}, i {i}) outside r {r} x bucket {bucket}")
    return (mb * r + peer) * bucket + i


def host_token(input_ids: Sequence[int], kv_tokens: int) -> int:
    """The coordinator's host token of a request with `kv_tokens` KV tokens before this step: its input token at
    position kv_tokens - 1 when the host already has it, else -1."""
    return int(input_ids[int(kv_tokens) - 1]) if len(input_ids) >= int(kv_tokens) else -1


@dataclass
class CadTokenInputs:
    """Fixed device buffers of the token path for up to num_mb x r x t_max rows and r x table_rows requests."""

    r: int
    num_mb: int
    t_max: int
    table_rows: int
    ids_host: torch.Tensor   # int32 [T]
    slot: torch.Tensor       # int64 [T]
    pos: torch.Tensor        # int32 [T]
    out: torch.Tensor        # int32 [T]: the step's sampled token per row
    last_tok: torch.Tensor   # int32 [r table_rows + 1]: per request slot the last sampled token, then the dump slot
    bucket: int = 0

    @classmethod
    def allocate(cls, *, r: int, num_mb: int, t_max: int, table_rows: int, device) -> "CadTokenInputs":
        T = num_mb * r * t_max
        dump = r * table_rows
        return cls(r=r, num_mb=num_mb, t_max=t_max, table_rows=table_rows,
                   ids_host=torch.zeros(T, dtype=torch.int32, device=device),
                   slot=torch.full((T,), dump, dtype=torch.int64, device=device),
                   pos=torch.zeros(T, dtype=torch.int32, device=device),
                   out=torch.zeros(T, dtype=torch.int32, device=device),
                   last_tok=torch.zeros(dump + 1, dtype=torch.int32, device=device))

    @property
    def dump_slot(self) -> int:
        return self.r * self.table_rows

    def bind(self, bucket: int) -> None:
        if not 0 <= int(bucket) <= self.t_max:
            raise ValueError(f"bucket {bucket} outside 0..{self.t_max}")
        self.bucket = int(bucket)

    @property
    def rows(self) -> int:
        """Rows of the bound bucket (num_mb r B)."""
        return self.num_mb * self.r * self.bucket

    def mb_rows(self, mb: int) -> slice:
        n = self.r * self.bucket
        return slice(mb * n, (mb + 1) * n)

    def host_lists(self, rows: Sequence[CadRow]) -> tuple[list[int], list[int], list[int], list[int]]:
        """(ids_host, slot, pos, row of each request) for the bound bucket; padding rows: token 0, dump slot, position 0."""
        n = self.rows
        ids, slot, pos = [0] * n, [self.dump_slot] * n, [0] * n
        where, seen = [], set()
        for q in rows:
            if q.mb >= self.num_mb or not 0 <= q.table_index < self.table_rows:
                raise ValueError(f"row {q}: mb / table index outside {self.num_mb} / {self.table_rows}")
            k = row_of(self.r, self.bucket, q.mb, q.peer, q.i)
            if k in seen:
                raise ValueError(f"two requests on row {k}")
            seen.add(k)
            ids[k], slot[k], pos[k] = int(q.host_token), q.peer * self.table_rows + q.table_index, int(q.position)
            where.append(k)
        return ids, slot, pos, where

    def fill(self, rows: Sequence[CadRow]) -> list[int]:
        """Copy the step's rows in (pinned host -> device, no host sync); returns each request's row (for the reply)."""
        ids, slot, pos, where = self.host_lists(rows)
        n = self.rows
        for dst, vals, dt in ((self.ids_host, ids, torch.int32), (self.slot, slot, torch.int64), (self.pos, pos, torch.int32)):
            src = torch.tensor(vals, dtype=dt, pin_memory=dst.is_cuda)
            dst[:n].copy_(src, non_blocking=dst.is_cuda)
        return where

    def fill_dummy(self) -> None:
        """A step of padding rows only (warm-up / capture): token 0, dump slot, position 0."""
        n = self.rows
        self.ids_host[:n].zero_()
        self.slot[:n].fill_(self.dump_slot)
        self.pos[:n].zero_()

    # ---- the in-graph ops (stream-ordered, fixed shapes)
    def input_ids(self) -> torch.Tensor:
        """This step's token per row: the host's where it sent one, else the request's last sampled token."""
        n = self.rows
        host = self.ids_host[:n]
        return torch.where(host >= 0, host, self.last_tok.index_select(0, self.slot[:n]))

    def record(self, tokens: torch.Tensor) -> None:
        """Store the step's sampled tokens: out per row, and into each request's last-token slot (padding -> dump)."""
        n = self.rows
        self.out[:n].copy_(tokens.to(torch.int32))
        self.last_tok.index_copy_(0, self.slot[:n], self.out[:n])


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling: the first maximum of every row (torch.argmax), int32."""
    return torch.argmax(logits, dim=-1).to(torch.int32)


def reply_tokens(out_host: Sequence[int], rows: Sequence[CadRow], where: Sequence[int]) -> dict[int, list[int]]:
    """The reply: per peer its requests' sampled tokens, in the order the rows were given."""
    got: dict[int, list[int]] = {}
    for q, k in zip(rows, where, strict=True):
        got.setdefault(int(q.peer), []).append(int(out_host[k]))
    return got


__all__ = ["CadRow", "CadTokenInputs", "greedy", "host_token", "reply_tokens", "row_of"]
