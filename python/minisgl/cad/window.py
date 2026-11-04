"""CAD window buffers: one symmetric-memory buffer per GPU laid out by minisgl.cad.layout.CadLayout, and the typed views
the hop kernels (minisgl.kernel.cad_hop) and the tasks (packages B / C) use.

  CadWindow.allocate(layout, role, rank, peers, group, device, timeout_ns)   torch symm_mem buffer + rendezvous (GPU)
  CadWindow.local(layout, role, buffers, rank, peers, timeout_ns)            caller-owned buffers (CPU tests, fake ranks)

`peers`: the ranks (in `group`) this GPU talks to. A model GPU's peers are its r attention GPUs in peer order j; an
attention GPU's peers are [its model GPU]. Views of a peer's buffer are writes over NVLink (symmetric memory).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .layout import ROLE_ATTN, ROLE_MODEL, WORD, CadLayout


class CadWindow:
    def __init__(self, layout: CadLayout, role: str, rank: int, peers: Sequence[int], base: torch.Tensor,
                 peer_bases: Mapping[int, torch.Tensor], timeout_ns: int, handle=None, peer_j: int | None = None):
        if role not in (ROLE_ATTN, ROLE_MODEL):
            raise ValueError(f"role {role!r}: {ROLE_ATTN} or {ROLE_MODEL}")
        peers = [int(p) for p in peers]
        if role == ROLE_MODEL and len(peers) != layout.r:
            raise ValueError(f"a model GPU has r = {layout.r} attention peers, got {peers}")
        if role == ROLE_ATTN and len(peers) != 1:
            raise ValueError(f"an attention GPU has exactly one model GPU peer, got {peers}")
        for b in [base, *peer_bases.values()]:
            if b.dtype != torch.uint8 or b.dim() != 1 or b.numel() != layout.window_bytes:
                raise ValueError(f"window buffers must be uint8 [{layout.window_bytes}], got {b.dtype} {tuple(b.shape)}")
        missing = [p for p in peers if p not in peer_bases]
        if missing:
            raise ValueError(f"no buffer for peers {missing}")
        if not isinstance(timeout_ns, int) or timeout_ns < 0:
            raise ValueError(f"timeout_ns {timeout_ns!r}")
        self.layout, self.role, self.rank, self.peers = layout, role, int(rank), peers
        self.base, self.peer_bases, self.timeout_ns, self.handle = base, dict(peer_bases), int(timeout_ns), handle
        words = self.words()
        self.epoch, self.err = words[layout.epoch_word:layout.epoch_word + 1], words[layout.err_word:layout.err_word + 1]
        self._peer_j = None
        if role == ROLE_ATTN:
            if peer_j is None:
                raise ValueError("an attention GPU needs peer_j, its position among its model GPU's peers (layout.peer_index)")
            self.set_peer_j(peer_j)

    # ---- construction
    @classmethod
    def allocate(cls, layout: CadLayout, role: str, rank: int, peers: Sequence[int], group, device,
                 timeout_ns: int, peer_j: int | None = None) -> "CadWindow":
        """Collective over `group` (every rank calls it with the same layout): one uint8 symmetric buffer per GPU."""
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        base = symm_mem.empty(layout.window_bytes, dtype=torch.uint8, device=device)
        base.zero_()
        handle = symm_mem.rendezvous(base, group=group)
        dist.barrier(group=group)
        peer_bases = {int(p): handle.get_buffer(int(p), (layout.window_bytes,), torch.uint8, 0) for p in peers}
        return cls(layout, role, rank, peers, base, peer_bases, timeout_ns, handle=handle, peer_j=peer_j)

    @classmethod
    def local(cls, layout: CadLayout, role: str, buffers: Mapping[int, torch.Tensor], rank: int, peers: Sequence[int],
              timeout_ns: int, peer_j: int | None = None) -> "CadWindow":
        """Fake ranks in one process: `buffers` maps every rank to its window (uint8 [window_bytes])."""
        return cls(layout, role, rank, peers, buffers[int(rank)], {int(p): buffers[int(p)] for p in peers}, timeout_ns,
                   peer_j=peer_j)

    # ---- raw views
    def buf(self, peer: int | None = None) -> torch.Tensor:
        if peer is None:
            return self.base
        if int(peer) not in self.peer_bases:
            raise ValueError(f"rank {peer} is not a peer of rank {self.rank} ({self.peers})")
        return self.peer_bases[int(peer)]

    def words(self, peer: int | None = None) -> torch.Tensor:
        """int32 view of the flag region (num_flags flag words, then epoch, err)."""
        lay = self.layout
        return self.buf(peer)[lay.data_bytes:lay.data_bytes + (lay.num_flags + 2) * WORD].view(torch.int32)

    # ---- the role-free interface (by word / peer; the role-specific views below are built on the same bytes)
    def flag(self, word: int, peer: int | None = None) -> torch.Tensor:
        """Flag word `word` (layout.qkv_flag / layout.o_flag, relative to the flag region) of this GPU or of `peer`."""
        lay = self.layout
        if not 0 <= int(word) < lay.num_flags:
            raise ValueError(f"flag word {word} outside 0..{lay.num_flags - 1}")
        return self.words(peer)[int(word):int(word) + 1]

    def qkv_slot_at(self, peer: int, mb: int) -> torch.Tensor:
        """Attention GPU `peer`'s microbatch-mb slot as flat int16 [t_max (q + 2 kv)] (the model GPU's put target)."""
        self._need(ROLE_MODEL)
        lay = self.layout
        off = lay.qkv_slot_offset(mb)
        return self.buf(peer)[off:off + lay.t_max * lay.qkv_row_elems * lay.elt].view(torch.int16)

    def o_rows_at(self, peer: int | None, mb: int, j: int, rows: int) -> torch.Tensor:
        """Peer j's `rows` o rows (from row j t_max) in the o slot of `peer` (the model GPU; None = this GPU) as bf16
        [rows, o_dim]: the attention GPU's decode `out` (a peer view) or the model GPU's own rows."""
        lay = self.layout
        if not 0 <= int(rows) <= lay.t_max:
            raise ValueError(f"rows {rows} outside 0..{lay.t_max}")
        if (self.role == ROLE_MODEL) != (peer is None):
            raise ValueError("o rows live on the model GPU: an attention GPU passes its model GPU's rank, a model GPU None")
        off = lay.o_rows_offset(mb, j)
        return self.buf(peer)[off:off + int(rows) * lay.o_dim * lay.elt].view(torch.bfloat16).view(int(rows), lay.o_dim)

    def _need(self, role: str) -> None:
        if self.role != role:
            raise ValueError(f"a {self.role} GPU has no {role}-side region")

    def _model_peer(self) -> int:
        return self.peers[0]

    # ---- attention GPU (its own slot, its model GPU's o rows and flags)
    def qkv_slot(self, mb: int) -> torch.Tensor:
        """This attention GPU's microbatch-mb slot as flat int16 [t_max (q + 2 kv)] (wait_store's `slot`)."""
        self._need(ROLE_ATTN)
        lay = self.layout
        off = lay.qkv_slot_offset(mb)
        return self.base[off:off + lay.t_max * lay.qkv_row_elems * lay.elt].view(torch.int16)

    def q_rows(self, mb: int, rows: int) -> torch.Tensor:
        """The q segment of `rows` rows as bf16 [rows, q_dim] (the decode's query)."""
        (q0, ql), _, _ = self.layout.qkv_segments(rows)
        return self.qkv_slot(mb)[q0:q0 + ql].view(torch.bfloat16).view(rows, self.layout.q_dim)

    def qkv_flag(self, layer: int, mb: int) -> torch.Tensor:
        self._need(ROLE_ATTN)
        w = self.layout.qkv_flag(layer, mb)
        return self.words()[w:w + 1]

    def o_rows_at_model(self, mb: int, rows: int) -> torch.Tensor:
        """This attention GPU's o rows in its model GPU's o slot (peer view) as bf16 [rows, o_dim]."""
        self._need(ROLE_ATTN)
        lay, j = self.layout, self.peer_j
        off = lay.o_rows_offset(mb, j)
        if not 0 <= rows <= lay.t_max:
            raise ValueError(f"rows {rows} outside 0..{lay.t_max}")
        return self.buf(self._model_peer())[off:off + rows * lay.o_dim * lay.elt].view(torch.bfloat16).view(rows, lay.o_dim)

    def o_flag_at_model(self, layer: int, mb: int) -> torch.Tensor:
        self._need(ROLE_ATTN)
        w = self.layout.o_flag(layer, mb, self.peer_j)
        return self.words(self._model_peer())[w:w + 1]

    @property
    def peer_j(self) -> int:
        """An attention GPU's position among its model GPU's peers (set by the owner: minisgl.cad.layout.peer_index)."""
        if getattr(self, "_peer_j", None) is None:
            raise ValueError("peer_j not set (CadWindow.set_peer_j)")
        return self._peer_j

    def set_peer_j(self, j: int) -> "CadWindow":
        self._need(ROLE_ATTN)
        if not 0 <= int(j) < self.layout.r:
            raise ValueError(f"peer index {j} outside 0..{self.layout.r - 1}")
        self._peer_j = int(j)
        return self

    # ---- model GPU (its o slot and flags, its peers' q / k / v slots and flags)
    def o_slot(self, mb: int) -> torch.Tensor:
        """This model GPU's microbatch-mb o slot as bf16 [r t_max, o_dim] (peer j's rows from row j t_max)."""
        self._need(ROLE_MODEL)
        lay = self.layout
        off = lay.o_slot_offset(mb)
        return self.base[off:off + lay.r * lay.t_max * lay.o_dim * lay.elt].view(torch.bfloat16).view(
            lay.r * lay.t_max, lay.o_dim)

    def o_flags(self, layer: int, mb: int) -> torch.Tensor:
        """The r o flags of (layer, mb), contiguous int32 [r] (cad_hop.wait_range's `words`)."""
        self._need(ROLE_MODEL)
        w = self.layout.o_flag(layer, mb, 0)
        return self.words()[w:w + self.layout.r]

    def qkv_segments_at(self, j: int, mb: int, rows: int) -> list[torch.Tensor]:
        """Flat int16 destinations of the q, k, v segments of `rows` rows in peer j's slot (put_rows' dsts)."""
        self._need(ROLE_MODEL)
        lay = self.layout
        off = lay.qkv_slot_offset(mb)
        slot = self.buf(self.peers[j])[off:off + lay.t_max * lay.qkv_row_elems * lay.elt].view(torch.int16)
        return [slot[s:s + n] for s, n in lay.qkv_segments(rows)]

    def qkv_flag_at(self, j: int, layer: int, mb: int) -> torch.Tensor:
        self._need(ROLE_MODEL)
        w = self.layout.qkv_flag(layer, mb)
        return self.words(self.peers[j])[w:w + 1]

    # ---- both
    def reset(self) -> None:
        """Zero this GPU's flag region (every flag, epoch, err): only between runs, with every rank idle and in step."""
        self.words().zero_()


__all__ = ["CadWindow"]
