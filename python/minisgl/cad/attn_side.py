"""CAD attention-GPU tasks: the core attention of every layer and microbatch, nothing else.

  pre(l, mb)    one kernel (kernel/cad_hop.py wait_store): wait for the model GPU's release of (l, mb)'s q|k|v rows in this
                GPU's window slot (qkv_flag(l, mb) >= epoch + 1, abort-aware), then store their K / V rows into the AG KV pool
                at the step's token slots (the coordinator's page grants, materialized as the AFD AG step does)
  attn(l, mb)   the AFD path's decode FMHA (FlashInfer trtllm_batch_decode_with_kv_cache, the TensorRTLLMBackend decode call of
                attention/trtllm.py, same arguments) reading q from the slot, writing o straight into this GPU's rows of the
                model GPU's o slot (a peer view, `out=`), then release_flag(o_flag(l, mb, j)) in the model GPU's window

A bucket with 0 rows runs no FMHA and still releases its o flag (and pre still waits for its q|k|v flag), so the model GPU's
wait over its r peers never hangs on an empty peer.

Fixed buffers per microbatch (graph-capturable; one CUDA graph per bucket replays the same addresses):
  loc[mb]    int64 [t_max] token slots of the new K / V (filled per step from the materialized sub-batch's out_loc)
  FMHA metadata: per-mb TensorRTLLMBackend capture buffers (init_capture_graph over the buckets, prepare_for_capture per
             bucket, prepare_for_replay per step: the AG decode graph's refill, afd_attention_worker.py). CAD owns
             its own per-mb backend objects (the same class and kernel as the AG stage's attn_backends; separate capture
             buffers), so the AG stage's prefill and its decode graphs stay untouched.
The window (cad/window.py) provides qkv_slot(mb), o_rows_at(peer, mb, j, rows), flag(word, peer=None), epoch,
err, timeout_ns and its layout (cad/layout.py); the hop kernels come in as `ops` (kernel/cad_hop.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """A flat BF16 view of a window region (CadWindow hands out int16 slots and bf16 [rows, o_dim] o rows)."""
    flat = t.reshape(-1)
    return flat if flat.dtype == torch.bfloat16 else flat.view(torch.bfloat16)


def _i16(t: torch.Tensor) -> torch.Tensor:
    """An int16 view of a 16-bit tensor (kernel/cad_hop.py wait_store copies 16-bit words: the BF16 KV pool)."""
    if t.element_size() != 2:
        raise ValueError(f"CAD v1 stores BF16 K / V (16-bit), got a {t.dtype} pool")
    return t if t.dtype == torch.int16 else t.view(torch.int16)


@dataclass(frozen=True)
class AttnDims:
    q_heads: int
    kv_heads: int
    head_dim: int

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim


def trtllm_decode_into(backend, layer: int, q: torch.Tensor, metadata, out: torch.Tensor) -> None:
    """TensorRTLLMBackend.forward's decode call (attention/trtllm.py, the non-prefill branch, q_len_per_req 1) with
    `out` = the peer's o rows."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    kvcache = backend.kvcache
    trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=(kvcache.k_cache(layer), kvcache.v_cache(layer)),
        workspace_buffer=backend.workspace_buffer,
        block_tables=metadata.page_table,
        seq_lens=metadata.cache_seqlens,
        max_seq_len=metadata.max_seqlen_k,
        bmm1_scale=backend.scale,
        bmm2_scale=1.0,
        kv_layout="NHD",
        out=out,
        out_dtype=q.dtype,
        q_len_per_req=1,
    )


class TrtllmCadMeta:
    """Per-microbatch FMHA metadata + K / V token slots of the CAD attention side (production: TensorRTLLMBackend
    capture buffers). bind(bucket) points the metadata at the bucket's capture views; fill(subs) refills them from the
    step's materialized sub-batches; fill_dummy(bucket) writes a step of dummy rows (every row the dummy request's page,
    seq len 1, K / V stored to the dummy slot)."""

    def __init__(self, *, backends: Sequence, num_mb: int, t_max: int, buckets: Sequence[int], max_seq_len: int,
                 dummy_req, dummy_slot: int, dummy_page: int, device):
        from minisgl.core import Batch

        if len(backends) != num_mb:
            raise ValueError(f"CAD attention side: {len(backends)} backends for {num_mb} microbatches")
        self.backends, self.num_mb, self.t_max = list(backends), int(num_mb), int(t_max)
        self.dummy_req, self.dummy_slot, self.dummy_page = dummy_req, int(dummy_slot), int(dummy_page)
        self.buckets = sorted({int(b) for b in buckets if int(b) > 0})
        for backend in self.backends:
            backend.init_capture_graph(max_seq_len=int(max_seq_len), bs_list=list(self.buckets))
        self.loc = [torch.full((self.t_max,), self.dummy_slot, dtype=torch.int64, device=device) for _ in range(num_mb)]
        self._Batch = Batch
        self._cap: dict[int, list] = {}   # bucket -> per mb the capture sub-batch
        self.bucket = 0

    def bind(self, bucket: int) -> None:
        self.bucket = int(bucket)
        if self.bucket == 0 or self.bucket in self._cap:
            return
        subs = []
        for backend in self.backends:
            sub = self._Batch(reqs=[self.dummy_req] * self.bucket, phase="decode")
            sub.padded_reqs = [self.dummy_req] * self.bucket
            backend.prepare_for_capture(sub)
            subs.append(sub)
        self._cap[self.bucket] = subs

    def metadata(self, mb: int):
        return self._cap[self.bucket][mb].attn_metadata

    def fill(self, subs: Sequence) -> None:
        B = self.bucket
        if len(subs) != self.num_mb:
            raise ValueError(f"CAD attention step: {len(subs)} sub-batches for {self.num_mb} microbatches")
        for mb, sub in enumerate(subs):
            if int(sub.padded_size) != B:
                raise ValueError(f"CAD attention step: mb {mb} has {int(sub.padded_size)} rows, the bucket is {B}")
            if B == 0:
                continue
            self.loc[mb][:B].copy_(sub.out_loc[:B])
            # the step's own metadata (materialized as the AFD AG step does) is the SOURCE prepare_for_replay copies into
            # the capture buffers; the bucket's capture sub-batch keeps pointing at those buffers (what attn() reads)
            src = self._Batch(reqs=sub.reqs, phase="decode")
            src.padded_reqs = sub.padded_reqs
            src.attn_metadata = sub.attn_metadata
            src.afd_req_table_indices_gpu = getattr(sub, "afd_req_table_indices_gpu", None)
            self.backends[mb].prepare_for_replay(src)

    def fill_dummy(self, bucket: int) -> None:
        B = int(bucket)
        for mb in range(self.num_mb):
            self.loc[mb][:B].fill_(self.dummy_slot)
            if B == 0:
                continue
            cap = self.backends[mb].capture
            cap.seq_lens[:B].fill_(1)
            cap.cu_seqlens_k[: B + 1].copy_(torch.arange(B + 1, dtype=cap.cu_seqlens_k.dtype, device=cap.cu_seqlens_k.device))
            cap.page_table[:B].fill_(self.dummy_page)


class CadAttnTasks:
    """The attention GPU's task executor for cad/runtime.py (pre / attn; bind / fill / fill_dummy)."""

    def __init__(self, *, win, ops, dims: AttnDims, kv: Callable[[int], tuple[torch.Tensor, torch.Tensor]], meta,
                 fmha: Callable, model_peer: int, j: int, subs_of: Callable | None = None):
        """win: this GPU's CadWindow (role attn); ops: kernel/cad_hop; kv(layer) -> this GPU's (k, v) pool of the layer;
        meta: TrtllmCadMeta (or a fake); fmha(layer, mb, q, metadata, out); model_peer: the window peer index of this
        GPU's model GPU; j: this GPU's position among that model GPU's peers; subs_of(cmd) -> the step's per-mb
        materialized sub-batches (the worker's AFD materialization)."""
        lay = win.layout
        if dims.q_dim != lay.q_dim or dims.kv_dim != lay.kv_dim or dims.q_dim != lay.o_dim:
            raise ValueError(f"CAD attention dims {dims} do not match the window layout {lay.describe()}")
        if not 0 <= int(j) < lay.r:
            raise ValueError(f"peer position {j} outside 0..{lay.r - 1}")
        self.win, self.ops, self.dims, self.kv, self.meta, self.fmha = win, ops, dims, kv, meta, fmha
        self.model_peer, self.j, self.subs_of = int(model_peer), int(j), subs_of
        self.bucket = 0

    # ---- step binding
    def bind(self, bucket: int) -> None:
        self.bucket = int(bucket)
        self.meta.bind(self.bucket)

    def fill(self, cmd) -> None:
        self.meta.fill(self.subs_of(cmd))

    def fill_dummy(self, bucket: int) -> None:
        self.meta.fill_dummy(bucket)

    # ---- tasks
    def pre(self, layer: int, mb: int) -> None:
        win, lay, d, B = self.win, self.win.layout, self.dims, self.bucket
        k_pool, v_pool = self.kv(layer)
        self.ops.wait_store(win.flag(lay.qkv_flag(layer, mb)), win.epoch, win.err, win.timeout_ns, win.qkv_slot(mb),
                            _i16(k_pool), _i16(v_pool), self.meta.loc[mb][:B], B, d.q_dim, d.kv_dim)

    def attn(self, layer: int, mb: int) -> None:
        win, lay, d, B = self.win, self.win.layout, self.dims, self.bucket
        if B:
            q = _bf16(win.qkv_slot(mb))[: B * d.q_dim].view(B, d.q_heads, d.head_dim)
            out = _bf16(win.o_rows_at(self.model_peer, mb, self.j, B))[: B * d.q_dim].view(B, d.q_heads, d.head_dim)
            self.fmha(layer, mb, q, self.meta.metadata(mb), out)
        self.ops.release_flag(win.flag(lay.o_flag(layer, mb, self.j), peer=self.model_peer), win.epoch)


__all__ = ["AttnDims", "CadAttnTasks", "TrtllmCadMeta", "trtllm_decode_into"]
