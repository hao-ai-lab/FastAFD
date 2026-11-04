"""CAD hop kernels: the per-layer q / k / v put from a model GPU to its attention GPUs, the o write-back,
and the flags that order them, all stream operations (CUDA-graph capturable, no host sync).

Kernels (CAD moves 19-138 rows per attention GPU per microbatch):
  cad_wait / cad_flag / cad_epoch  flag wait (acquire), release (epoch + 1) and the per-step epoch bump
  cad_copy        the put's copy over a grid of CTAs (strided over the programs)
  cad_wait_store  the fused flag wait + K / V store over a grid of CTAs (each CTA waits, then stores rows pid, pid + G, ..)
  cad_wait_range  one CTA waits on `count` contiguous flag words (a model GPU's r o flags of one (layer, mb))
A put is TWO launches on the stream: cad_copy, then cad_flag (release = epoch + 1 at system scope); stream order puts the
release after every CTA's stores. Copies move 16-bit words (BF16 rows as int16).

Flag protocol (minisgl.cad.layout): a release sets flag = epoch + 1; a wait passes at flag >= epoch + add; each rank bumps
its own epoch once per step (bump_epoch, the step's last kernel). `err`: a wait that times out sets it, and every later
wait on this rank sees it and does not spin, so a dead peer costs one timeout per rank.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer

COPY_BLOCK = 1024           # 16-bit elements per CTA per iteration
MAX_COPY_CTAS = 32
MAX_STORE_CTAS = 64
ELEMS_PER_COPY_CTA = 8 * COPY_BLOCK
_MAX_SEGMENTS = 3
_MAX_WAIT_FLAGS = 3
# Run-time integers that vary with the bucket or the call site: never specialised (a new row count must not JIT-compile a
# kernel behind a pending flag wait). The arithmetic is unchanged (copies and spins).
NO_SPECIALIZE = {
    "cad_wait": ("add", "timeout_ns"),
    "cad_wait_range": ("count", "add", "timeout_ns"),
    "cad_copy": ("ss0", "w0", "n0", "ss1", "w1", "n1", "ss2", "w2", "n2", "G"),
    "cad_wait_store": ("add", "timeout_ns", "C", "G"),
}


@triton.jit(do_not_specialize=NO_SPECIALIZE["cad_wait"])
def cad_wait(f0, f1, f2, NF: tl.constexpr, epoch, add, err, timeout_ns):
    target = tl.load(epoch) + add
    start = globaltimer()
    dead = tl.load(err) != 0
    v = tl.atomic_add(f0, 0, sem="acquire", scope="sys")
    while (v < target) & (globaltimer() - start < timeout_ns) & (not dead):
        v = tl.atomic_add(f0, 0, sem="acquire", scope="sys")
    bad = v < target
    if NF > 1:
        v = tl.atomic_add(f1, 0, sem="acquire", scope="sys")
        while (v < target) & (globaltimer() - start < timeout_ns) & (not dead):
            v = tl.atomic_add(f1, 0, sem="acquire", scope="sys")
        bad = bad | (v < target)
        v = tl.atomic_add(f2, 0, sem="acquire", scope="sys")
        while (v < target) & (globaltimer() - start < timeout_ns) & (not dead):
            v = tl.atomic_add(f2, 0, sem="acquire", scope="sys")
        bad = bad | (v < target)
    if bad:
        tl.atomic_xchg(err, 1)


@triton.jit(do_not_specialize=NO_SPECIALIZE["cad_wait_range"])
def cad_wait_range(base, count, epoch, add, err, timeout_ns):
    target = tl.load(epoch) + add
    start = globaltimer()
    dead = tl.load(err) != 0
    nbad = tl.full([], 0, tl.int32)
    for i in range(count):
        v = tl.atomic_add(base + i, 0, sem="acquire", scope="sys")
        while (v < target) & (globaltimer() - start < timeout_ns) & (not dead):
            v = tl.atomic_add(base + i, 0, sem="acquire", scope="sys")
        nbad += (v < target).to(tl.int32)
    if nbad > 0:
        tl.atomic_xchg(err, 1)


@triton.jit(do_not_specialize=NO_SPECIALIZE["cad_copy"])
def cad_copy(s0, ss0, w0, n0, d0, s1, ss1, w1, n1, d1, s2, ss2, w2, n2, d2, G,
             NSEG: tl.constexpr, BLOCK: tl.constexpr):
    # the put's body (src row stride ss, width w, n rows -> flat dst), the iterations strided over the G programs
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    t0 = n0 * w0
    for i in range(0, t0, G * BLOCK):
        idx = i + offs
        m = idx < t0
        tl.store(d0 + idx, tl.load(s0 + (idx // w0) * ss0 + idx % w0, mask=m), mask=m)
    if NSEG > 1:
        t1 = n1 * w1
        for i in range(0, t1, G * BLOCK):
            idx = i + offs
            m = idx < t1
            tl.store(d1 + idx, tl.load(s1 + (idx // w1) * ss1 + idx % w1, mask=m), mask=m)
    if NSEG > 2:
        t2 = n2 * w2
        for i in range(0, t2, G * BLOCK):
            idx = i + offs
            m = idx < t2
            tl.store(d2 + idx, tl.load(s2 + (idx // w2) * ss2 + idx % w2, mask=m), mask=m)


@triton.jit
def cad_flag(flag, epoch):
    tl.debug_barrier()
    tl.atomic_xchg(flag, tl.load(epoch) + 1, sem="release", scope="sys")


@triton.jit
def cad_epoch(epoch):
    tl.store(epoch, tl.load(epoch) + 1)


@triton.jit(do_not_specialize=NO_SPECIALIZE["cad_wait_store"])
def cad_wait_store(flag, epoch, add, err, timeout_ns, src, dk, dv, loc, C, G,
                   QDIM: tl.constexpr, KDIM: tl.constexpr):
    # every program waits, then stores rows pid, pid + G, ... (the row loop strided over the programs)
    target = tl.load(epoch) + add
    start = globaltimer()
    dead = tl.load(err) != 0
    v = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
    while (v < target) & (globaltimer() - start < timeout_ns) & (not dead):
        v = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
    bad = v < target
    if bad:
        tl.atomic_xchg(err, 1)
    tl.debug_barrier()
    pid = tl.program_id(0)
    offs = tl.arange(0, KDIM)
    for i in range(0, C, G):
        row = i + pid
        if row < C:
            p = tl.load(loc + row).to(tl.int64)
            k = tl.load(src + C * QDIM + row * KDIM + offs)
            w = tl.load(src + C * (QDIM + KDIM) + row * KDIM + offs)
            tl.store(dk + p * KDIM + offs, k)
            tl.store(dv + p * KDIM + offs, w)


# ------------------------------------------------------------------ launch wrappers

def _check_word(name: str, x: torch.Tensor) -> None:
    if x.dtype != torch.int32 or x.numel() != 1:
        raise ValueError(f"{name} must be one int32 word, got {x.dtype} x {x.numel()}")


def wait_flags(flags: Sequence[torch.Tensor], epoch: torch.Tensor, err: torch.Tensor, timeout_ns: int,
               add: int = 1) -> None:
    """One CTA waits until every flag (1 .. 3) >= epoch + add (acquire), or sets err after timeout_ns."""
    if not 1 <= len(flags) <= _MAX_WAIT_FLAGS:
        raise ValueError(f"a wait takes 1 .. {_MAX_WAIT_FLAGS} flags, got {len(flags)}")
    for i, f in enumerate(flags):
        _check_word(f"flag {i}", f)
    _check_word("epoch", epoch)
    _check_word("err", err)
    f = list(flags) + [flags[0]] * (_MAX_WAIT_FLAGS - len(flags))
    cad_wait[(1,)](f[0], f[1], f[2], 1 if len(flags) == 1 else 3, epoch, int(add), err, int(timeout_ns), num_warps=1)


def wait_range(words: torch.Tensor, epoch: torch.Tensor, err: torch.Tensor, timeout_ns: int, add: int = 1) -> None:
    """One CTA waits until every word of the contiguous int32 view `words` (1 .. n) >= epoch + add, or sets err."""
    if words.dtype != torch.int32 or words.dim() != 1 or not words.is_contiguous() or words.numel() < 1:
        raise ValueError(f"words must be a contiguous 1-D int32 view of >= 1 flag, got {words.dtype} {tuple(words.shape)}")
    _check_word("epoch", epoch)
    _check_word("err", err)
    cad_wait_range[(1,)](words, int(words.numel()), epoch, int(add), err, int(timeout_ns), num_warps=1)


def copy_ctas(total_elems: int, ctas: int | None = None) -> int:
    """CTAs of a put's copy: ~8 iterations of COPY_BLOCK per CTA, 1 .. MAX_COPY_CTAS (fixed per bucket: graph-stable)."""
    if ctas is not None:
        if not 1 <= int(ctas) <= 1024:
            raise ValueError(f"ctas {ctas} outside 1 .. 1024")
        return int(ctas)
    return max(1, min(MAX_COPY_CTAS, -(-int(total_elems) // ELEMS_PER_COPY_CTA)))


def put_rows(segments: Sequence[tuple[torch.Tensor, torch.Tensor]], flag: torch.Tensor, epoch: torch.Tensor,
             ctas: int | None = None) -> None:
    """Copy 1 .. 3 (src [rows, width] with unit inner stride, dst flat contiguous) segments over a grid of CTAs, then
    release flag = epoch + 1 (a second launch, after every CTA's stores). dst may be a peer's symmetric view. An empty
    put still releases its flag (a peer with 0 rows in the microbatch)."""
    if not 1 <= len(segments) <= _MAX_SEGMENTS:
        raise ValueError(f"a put takes 1 .. {_MAX_SEGMENTS} segments, got {len(segments)}")
    _check_word("flag", flag)
    _check_word("epoch", epoch)
    args, total = [], 0
    for i, (s, d) in enumerate(segments):
        if s.dim() != 2 or (s.shape[0] > 1 and s.stride(1) != 1) or s.dtype != d.dtype:
            raise ValueError(f"segment {i}: src must be 2-D with unit inner stride and dst's dtype "
                             f"(got {tuple(s.shape)} strides {s.stride()} {s.dtype} -> {d.dtype})")
        if s.element_size() != 2:
            raise ValueError(f"segment {i}: 16-bit elements only (pass int16 / bf16 views), got {s.dtype}")
        if d.dim() != 1 or not d.is_contiguous() or d.numel() != s.numel():
            raise ValueError(f"segment {i}: dst must be flat and contiguous with {s.numel()} elements, got "
                             f"{tuple(d.shape)}")
        args += [s, s.stride(0), s.shape[1], s.shape[0], d]
        total += s.numel()
    while len(args) < 5 * _MAX_SEGMENTS:
        args += [segments[0][0], 0, 0, 0, segments[0][1]]
    g = copy_ctas(total, ctas)
    cad_copy[(g,)](*args, g, NSEG=len(segments), BLOCK=COPY_BLOCK, num_warps=8)
    cad_flag[(1,)](flag, epoch, num_warps=1)


def release_flag(flag: torch.Tensor, epoch: torch.Tensor) -> None:
    """One CTA sets flag = epoch + 1 (release) after the stream's earlier work (e.g. the decode that wrote the o rows)."""
    _check_word("flag", flag)
    _check_word("epoch", epoch)
    cad_flag[(1,)](flag, epoch, num_warps=1)


def bump_epoch(epoch: torch.Tensor) -> None:
    """The step's last kernel: epoch += 1, so the next step's waits need the next releases."""
    _check_word("epoch", epoch)
    cad_epoch[(1,)](epoch, num_warps=1)


def store_ctas(rows: int, ctas: int | None = None) -> int:
    if ctas is not None:
        if not 1 <= int(ctas) <= 1024:
            raise ValueError(f"ctas {ctas} outside 1 .. 1024")
        return int(ctas)
    return max(1, min(MAX_STORE_CTAS, int(rows)))


def wait_store(flag: torch.Tensor, epoch: torch.Tensor, err: torch.Tensor, timeout_ns: int, qkv_slot: torch.Tensor,
               k_pool: torch.Tensor, v_pool: torch.Tensor, loc: torch.Tensor, rows: int, q_dim: int, kv_dim: int,
               add: int = 1, ctas: int | None = None) -> None:
    """The attention GPU's KV write: wait for the q / k / v flag, then the new rows' K / V from the segment-major slot
    (q [rows, q_dim] | k [rows, kv_dim] | v [rows, kv_dim], 16-bit) to pool rows loc (row p = elements p kv_dim ..), over
    a grid of CTAs, one launch. The rows are copied even after a timeout (the step fails through err)."""
    _check_word("flag", flag)
    _check_word("epoch", epoch)
    _check_word("err", err)
    if qkv_slot.element_size() != 2 or k_pool.element_size() != 2 or v_pool.element_size() != 2:
        raise ValueError("wait_store copies 16-bit elements (pass int16 views)")
    if qkv_slot.numel() < rows * (q_dim + 2 * kv_dim) or loc.numel() != rows or loc.dtype != torch.int64:
        raise ValueError(f"slot of {qkv_slot.numel()} elements / loc {loc.dtype} x {loc.numel()} do not hold {rows} "
                         "rows")
    if kv_dim & (kv_dim - 1):
        raise ValueError(f"kv_dim {kv_dim} must be a power of two (one tl.arange row)")
    g = store_ctas(rows, ctas)
    cad_wait_store[(g,)](flag, epoch, int(add), err, int(timeout_ns), qkv_slot, k_pool, v_pool, loc, int(rows), g,
                         QDIM=int(q_dim), KDIM=int(kv_dim), num_warps=4)


__all__ = ["COPY_BLOCK", "MAX_COPY_CTAS", "MAX_STORE_CTAS", "bump_epoch", "copy_ctas", "put_rows", "release_flag",
           "store_ctas", "wait_flags", "wait_range", "wait_store"]
