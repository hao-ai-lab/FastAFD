"""BF16-KV fused full attention (attn_fused on BF16 projections): CPU replay of the kernel against its chain.

The CUDA kernel cannot run in the CPU lane; its byte oracle against the real chain is
tests/kernel/test_qwen35_full_attention_decode_bf16.py. This file replays

* the chain the kernel replaces, in tensor views: split the QKV rows into query/gate,
  key and value; FlashInfer QKRMSNorm per (token, head) in place; FlashInfer neox RoPE on
  the first rotary_dim elements (the partner element sits rotary_dim / 2 away); then
  store_cache of the rotated key and the projected value at out_loc, and
* qwen35_full_attn_norm_rope_bf16_store.cu in its own index arithmetic: 36 jobs per
  token, one 32-lane warp per job, flat offsets into the row (query at head * 512, key at
  16384 + h * 256, value at 16896 + h * 256) and into the flat caches (slot * 512 +
  h * 256 + lane * 8), the RoPE partner as lane ^ (rotary_dim / 16),

and requires the in-place query and every byte of both caches to agree. Instructions
both sides execute alike under --use_fast_math share one CPU stand-in, so what is
compared is the layout, the index math, the order of operations and the rounding points.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import minisgl.kernel as kernel_pkg

Q_HEADS, KV_HEADS, HEAD_DIM, ROTARY = 32, 2, 256, 64  # Qwen3.5 full attention
Q_SIZE, KV_SIZE = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
WIDTH = 2 * Q_SIZE + 2 * KV_SIZE
EPS = 1e-6
SLOTS = 64
_SOURCE = Path(kernel_pkg.__file__).parent / "csrc" / "jit" / "qwen35_full_attn_norm_rope_bf16_store.cu"
_TINY = 2.0**-126


def _ftz(x):
    return torch.where(x.abs() < _TINY, x * 0.0, x)


def _ffma(a, b, c):
    a, b, c = _ftz(a), _ftz(b), _ftz(c)
    return _ftz((a.double() * b.double() + c.double()).float())


def _fadd(a, b):
    return _ftz(_ftz(a) + _ftz(b))


def _fmul(a, b):
    return _ftz(_ftz(a) * _ftz(b))


def _rsqrt(x):
    return (1.0 / torch.sqrt(x.double())).float()


def _rope_mix(x, cos, partner, sin):
    """`x * cos + partner * sin`, one stand-in for however the compiler contracts it."""
    return _ftz((x.double() * cos.double() + partner.double() * sin.double()).float())


def _butterfly(values):
    lanes = torch.arange(32)
    for mask in (16, 8, 4, 2, 1):
        values = _fadd(values, values[..., lanes ^ mask])
    return values[..., 0]


def _lane_norm(lanes_x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[..., 32, 8] float of one (token, head) -> BF16-rounded normed values, as float."""
    partial = torch.zeros(lanes_x.shape[:-1])
    for j in range(8):
        partial = _ffma(lanes_x[..., j], lanes_x[..., j], partial)
    total = _butterfly(partial)
    rms = _rsqrt(_fadd((_ftz(total).double() / HEAD_DIM).float(), torch.tensor(EPS)))
    scale = _fadd(torch.zeros(()), weight.float().view(32, 8))
    return _fmul(_fmul(lanes_x, rms[..., None, None]), scale).to(torch.bfloat16).float()


# ---- the chain, in tensor views --------------------------------------------------------
def chain(qkv, positions, q_weight, k_weight, cos_sin, out_loc, k_cache, v_cache):
    rows = qkv.shape[0]
    q_gate, key, value = qkv.split([2 * Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
    query = q_gate.view(rows, Q_HEADS, 2 * HEAD_DIM)[..., :HEAD_DIM]

    def norm_rope(x, weight):  # x [rows, H, 256] BF16
        normed = _lane_norm(x.float().reshape(rows, x.shape[1], 32, 8), weight).reshape(x.shape)
        half = ROTARY // 2
        cs = cos_sin[positions.long()]  # [rows, ROTARY]: cos | sin
        cos, sin = cs[:, None, :half], cs[:, None, half:]
        out = normed.clone()
        out[..., :half] = _rope_mix(normed[..., :half], cos, -normed[..., half:ROTARY], sin)
        out[..., half:ROTARY] = _rope_mix(normed[..., half:ROTARY], cos, normed[..., :half], sin)
        return out.to(torch.bfloat16)

    rotated_query = norm_rope(query, q_weight)
    rotated_key = norm_rope(key.view(rows, KV_HEADS, HEAD_DIM), k_weight)
    new_qkv = qkv.clone()
    new_qkv[:, :2 * Q_SIZE].view(rows, Q_HEADS, 2 * HEAD_DIM)[..., :HEAD_DIM] = rotated_query
    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    k_cache[out_loc.long()] = rotated_key.view(rows, KV_SIZE)
    v_cache[out_loc.long()] = value
    return new_qkv, k_cache, v_cache


# ---- the kernel, in its own index arithmetic ------------------------------------------
def kernel_constants() -> dict:
    source = _SOURCE.read_text()

    def const(name):
        return re.search(rf"constexpr uint32_t {name} = ([^;]+);", source).group(1).split("//")[0].strip()

    return {"vec": int(const("kVec")), "q_heads": int(const("kQHeads")), "kv_heads": int(const("kKVHeads")),
            "q_gate": const("kQGate"), "kv_row": const("kKVRow"), "width": const("kRowWidth"),
            "value_offset": "kQGate + kKVRow" in source.split("value head: store only")[1].split(";")[1]}


def kernel(qkv, positions, q_weight, k_weight, cos_sin, out_loc, k_cache, v_cache, *, mutation=None):
    rows = qkv.shape[0]
    flat = qkv.clone().view(-1)
    k_flat, v_flat = k_cache.clone().view(-1), v_cache.clone().view(-1)
    lanes = torch.arange(32)
    half_lanes = ROTARY // 2 // 8 if mutation != "partner_lane" else ROTARY // 8
    q_gate, kv_row = 2 * Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    for token in range(rows):
        row = token * WIDTH
        slot = int(out_loc[token]) if mutation != "slot_from_token" else token
        for head in range(Q_HEADS + 2 * KV_HEADS):
            if head >= Q_HEADS + KV_HEADS:  # value head: store only
                kv_head = head - Q_HEADS - KV_HEADS
                src = row + q_gate + (0 if mutation == "value_from_key" else kv_row) + kv_head * HEAD_DIM
                v_flat[slot * kv_row + kv_head * HEAD_DIM:][:HEAD_DIM] = flat[src:src + HEAD_DIM]
                continue
            is_query = head < Q_HEADS
            src = row + head * (2 * HEAD_DIM) if is_query else row + q_gate + (head - Q_HEADS) * HEAD_DIM
            weight = q_weight if is_query else k_weight
            lanes_x = flat[src:src + HEAD_DIM].float().view(32, 8)
            vec = _lane_norm(lanes_x, weight)
            if not (mutation == "key_not_rotated" and not is_query):
                permuted = vec[lanes ^ half_lanes]
                pos = int(positions[token])
                rotated = vec.clone()
                for lane in range(32):
                    if lane * 8 < ROTARY:
                        idx = (lane * 8) % (ROTARY // 2)
                        cos = cos_sin[pos, idx:idx + 8]
                        sin = cos_sin[pos, ROTARY // 2 + idx:ROTARY // 2 + idx + 8]
                        first_half = lane * 8 < ROTARY // 2
                        if mutation == "rope_sign":
                            first_half = not first_half
                        sign = -1.0 if first_half else 1.0
                        rotated[lane] = _rope_mix(vec[lane], cos, sign * permuted[lane], sin)
                vec = rotated
            out = vec.to(torch.bfloat16).view(-1)
            if is_query:
                flat[src:src + HEAD_DIM] = out
            else:
                k_flat[slot * kv_row + (head - Q_HEADS) * HEAD_DIM:][:HEAD_DIM] = out
    return flat.view(rows, WIDTH), k_flat.view(k_cache.shape), v_flat.view(v_cache.shape)


def _case(rows: int, seed: int, kind: str = "normal"):
    generator = torch.Generator().manual_seed(seed)
    qkv = torch.randn(rows, WIDTH, generator=generator) * 2.0
    if kind == "edge":
        qkv[0, :] = 0.0  # all-zero heads: rsqrt of eps
        if rows > 1:
            qkv[1] *= 1e-30  # subnormal inputs
        if rows > 2:
            qkv[2, ::7] *= 1e4
    qkv = qkv.to(torch.bfloat16)
    positions = torch.randint(0, 9000, (rows,), generator=generator, dtype=torch.int32)
    q_weight = (1 + torch.randn(HEAD_DIM, generator=generator) * 0.2).to(torch.bfloat16)
    k_weight = (1 + torch.randn(HEAD_DIM, generator=generator) * 0.2).to(torch.bfloat16)
    angles = torch.rand(9000, ROTARY // 2, generator=generator) * 6.3
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    out_loc = torch.randperm(SLOTS, generator=generator)[:rows].to(torch.int32)
    k_cache = (torch.randn(SLOTS, KV_SIZE, generator=generator)).to(torch.bfloat16)  # untouched slots keep bytes
    v_cache = (torch.randn(SLOTS, KV_SIZE, generator=generator)).to(torch.bfloat16)
    return qkv, positions, q_weight, k_weight, cos_sin, out_loc, k_cache, v_cache


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_the_source_layout_is_the_projection_layout():
    c = kernel_constants()
    assert (c["vec"], c["q_heads"], c["kv_heads"]) == (8, Q_HEADS, KV_HEADS)
    assert c["q_gate"] == "2 * kQHeads * kHeadDim" and c["kv_row"] == "kKVHeads * kHeadDim"
    assert c["width"] == "kQGate + 2 * kKVRow" and c["value_offset"]
    assert 2 * Q_SIZE + 2 * KV_SIZE == WIDTH == 17408


@pytest.mark.parametrize("kind", ["normal", "edge"])
@pytest.mark.parametrize("rows", [1, 3])
def test_kernel_replay_matches_the_chain_byte_for_byte(rows, kind):
    case = _case(rows, seed=20260930 + rows, kind=kind)
    ref_qkv, ref_k, ref_v = chain(*case)
    got_qkv, got_k, got_v = kernel(*case)
    assert _same(got_qkv, ref_qkv)  # rotated query in place; gate, key and value untouched
    assert _same(got_k, ref_k) and _same(got_v, ref_v)  # every slot, written or not


@pytest.mark.parametrize("mutation", ["partner_lane", "slot_from_token", "value_from_key", "key_not_rotated", "rope_sign"])
def test_the_comparison_catches_a_wrong_index_or_step(mutation):
    case = _case(2, seed=11)
    ref = chain(*case)
    got = kernel(*case, mutation=mutation)
    assert not all(_same(g, r) for g, r in zip(got, ref))
