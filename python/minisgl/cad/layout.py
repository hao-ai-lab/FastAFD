"""CAD window layout: where every q / k / v slot, o slot and flag word sits in the symmetric window (pure Python, no torch).

One symmetric buffer per GPU of `window_bytes` (the same size on every rank, as symmetric-memory rendezvous requires):

  [0, data_bytes)        attention GPU: num_mb q|k|v slots of t_max rows, each SEGMENT-MAJOR for c rows:
                           q rows at [0, c q), k rows at [c q, c (q + kv)), v rows at [c (q + kv), c (q + 2 kv)) elements
                         model GPU: num_mb o slots of r x t_max rows; peer j's rows start at row j t_max
  [data_bytes, ...)      int32 words: attention GPU qkv_flag(l, mb); model GPU o_flag(l, mb, j); then epoch, err

Flag protocol (kernel/cad_hop.py): a release sets a flag to epoch + 1 after its data; a wait passes at >= epoch + 1;
every rank bumps its own epoch once per decode step, so one word per (layer, mb[, peer]) serves every step. A timed-out
wait sets err; later waits on that rank do not spin.

Peers (AfdTopology's contiguous block mapping, afd_protocol.py mlp_dp_for_attn_dp / attn_dp_for_mlp_dp): A attention GPUs,
M model GPUs, r = A / M; model GPU j serves attention GPUs j r .. j r + r - 1, attention GPU a is peer a - r (a // r) of
model GPU a // r. On the model GPU the rows of peer j are rows j B .. (j + 1) B of each microbatch (B = the step's bucket,
<= t_max); in the o slot they land at row j t_max.
"""

from __future__ import annotations

from dataclasses import dataclass

ALIGN = 128          # bytes; slot and region starts
WORD = 4             # int32 flag words
ROLE_ATTN = "attn"
ROLE_MODEL = "model"


def align(x: int, a: int = ALIGN) -> int:
    return (int(x) + a - 1) // a * a


def model_of(attn: int, num_attn: int, num_model: int) -> int:
    """The model GPU (0-based among model GPUs) that serves attention GPU `attn`."""
    check_topology(num_attn, num_model)
    if not 0 <= attn < num_attn:
        raise ValueError(f"attention GPU {attn} outside 0..{num_attn - 1}")
    return attn * num_model // num_attn


def peer_index(attn: int, num_attn: int, num_model: int) -> int:
    """Attention GPU `attn`'s position j among its model GPU's peers."""
    r = num_attn // num_model
    return attn - model_of(attn, num_attn, num_model) * r


def peers_of(model: int, num_attn: int, num_model: int) -> list[int]:
    """The attention GPUs of model GPU `model`, in peer order j = 0 .. r - 1."""
    check_topology(num_attn, num_model)
    if not 0 <= model < num_model:
        raise ValueError(f"model GPU {model} outside 0..{num_model - 1}")
    r = num_attn // num_model
    return list(range(model * r, (model + 1) * r))


def check_topology(num_attn: int, num_model: int) -> None:
    if num_attn < 1 or num_model < 1 or num_attn % num_model:
        raise ValueError(f"CAD needs A = r M with r >= 1, got A {num_attn}, M {num_model}")


@dataclass(frozen=True)
class CadLayout:
    num_layers: int
    num_mb: int
    r: int            # attention GPUs per model GPU
    t_max: int        # most rows of one attention GPU in one microbatch (the largest bucket)
    q_dim: int        # elements per row (BF16): q heads x head dim
    kv_dim: int       # kv heads x head dim
    o_dim: int        # q heads x head dim
    elt: int = 2      # bytes per element (the hop kernels copy 16-bit words)

    def __post_init__(self) -> None:
        for k in ("num_layers", "num_mb", "r", "t_max", "q_dim", "kv_dim", "o_dim"):
            v = getattr(self, k)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"CadLayout.{k} must be a positive int, got {v!r}")
        if self.elt != 2:
            raise ValueError("CadLayout.elt must be 2 (BF16 rows; the hop kernels copy 16-bit words)")
        if self.kv_dim & (self.kv_dim - 1):
            raise ValueError(f"kv_dim {self.kv_dim} must be a power of two (wait_store copies one tl.arange row)")

    @classmethod
    def for_model(cls, *, num_layers: int, num_mb: int, r: int, t_max: int, q_heads: int, kv_heads: int,
                  head_dim: int) -> "CadLayout":
        return cls(num_layers=num_layers, num_mb=num_mb, r=r, t_max=t_max, q_dim=q_heads * head_dim,
                   kv_dim=kv_heads * head_dim, o_dim=q_heads * head_dim)

    # ---- sizes (bytes)
    @property
    def qkv_row_elems(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    @property
    def qkv_slot_bytes(self) -> int:
        return align(self.t_max * self.qkv_row_elems * self.elt)

    @property
    def o_slot_bytes(self) -> int:
        return align(self.r * self.t_max * self.o_dim * self.elt)

    @property
    def attn_data_bytes(self) -> int:
        return self.num_mb * self.qkv_slot_bytes

    @property
    def model_data_bytes(self) -> int:
        return self.num_mb * self.o_slot_bytes

    @property
    def data_bytes(self) -> int:
        return align(max(self.attn_data_bytes, self.model_data_bytes))

    @property
    def num_flags(self) -> int:
        """Flag words of the larger role (the model GPU's: one per (layer, mb, peer))."""
        return self.num_layers * self.num_mb * self.r

    @property
    def epoch_word(self) -> int:
        return self.num_flags

    @property
    def err_word(self) -> int:
        return self.num_flags + 1

    @property
    def flags_bytes(self) -> int:
        return align((self.num_flags + 2) * WORD)

    @property
    def window_bytes(self) -> int:
        return self.data_bytes + self.flags_bytes

    @property
    def flag_base_word(self) -> int:
        """Index of word 0 of the flag region in an int32 view of the whole window."""
        return self.data_bytes // WORD

    # ---- attention GPU
    def qkv_slot_offset(self, mb: int) -> int:
        """Byte offset of microbatch mb's q|k|v slot."""
        self._mb(mb)
        return mb * self.qkv_slot_bytes

    def qkv_segments(self, rows: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """(start, length) in ELEMENTS from the slot start of the q, k and v segments for `rows` rows."""
        self._rows(rows)
        c, q, kv = rows, self.q_dim, self.kv_dim
        return (0, c * q), (c * q, c * kv), (c * (q + kv), c * kv)

    def qkv_flag(self, layer: int, mb: int) -> int:
        """Flag word (relative to the flag region) the model GPU releases after putting (layer, mb)'s rows."""
        self._layer(layer)
        self._mb(mb)
        return layer * self.num_mb + mb

    # ---- model GPU
    def o_slot_offset(self, mb: int) -> int:
        self._mb(mb)
        return mb * self.o_slot_bytes

    def o_rows_offset(self, mb: int, j: int) -> int:
        """Byte offset of peer j's o rows in microbatch mb's o slot (row j t_max)."""
        self._peer(j)
        return self.o_slot_offset(mb) + j * self.t_max * self.o_dim * self.elt

    def o_flag(self, layer: int, mb: int, j: int) -> int:
        """Flag word (relative to the flag region) attention peer j releases after writing (layer, mb)'s o rows."""
        self._layer(layer)
        self._mb(mb)
        self._peer(j)
        return (layer * self.num_mb + mb) * self.r + j

    def model_rows(self, j: int, bucket: int) -> tuple[int, int]:
        """Rows of peer j in the model GPU's local tensors of one microbatch of `bucket` rows per peer."""
        self._peer(j)
        self._rows(bucket)
        return j * bucket, (j + 1) * bucket

    # ---- checks
    def _mb(self, mb: int) -> None:
        if not 0 <= mb < self.num_mb:
            raise ValueError(f"mb {mb} outside 0..{self.num_mb - 1}")

    def _layer(self, layer: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer {layer} outside 0..{self.num_layers - 1}")

    def _peer(self, j: int) -> None:
        if not 0 <= j < self.r:
            raise ValueError(f"peer {j} outside 0..{self.r - 1}")

    def _rows(self, rows: int) -> None:
        if not 0 <= rows <= self.t_max:
            raise ValueError(f"rows {rows} outside 0..{self.t_max}")

    def describe(self) -> dict:
        return {"num_layers": self.num_layers, "num_mb": self.num_mb, "r": self.r, "t_max": self.t_max,
                "q_dim": self.q_dim, "kv_dim": self.kv_dim, "o_dim": self.o_dim, "qkv_slot_bytes": self.qkv_slot_bytes,
                "o_slot_bytes": self.o_slot_bytes, "data_bytes": self.data_bytes, "flags_bytes": self.flags_bytes,
                "window_bytes": self.window_bytes}


__all__ = ["ALIGN", "CadLayout", "ROLE_ATTN", "ROLE_MODEL", "WORD", "align", "check_topology", "model_of", "peer_index",
           "peers_of"]
