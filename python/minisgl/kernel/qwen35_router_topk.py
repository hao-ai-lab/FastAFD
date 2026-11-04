"""One-launch softmax + top-k for the Qwen3.5-397B router (512 experts), bit-identical to the
two-kernel fallback of ``topk_softmax`` (MINISGL_QWEN35_FUSED_ROUTER=1).

``topk_softmax`` sends ``num_experts > 256`` to ``moe_topk_softmax.cu``'s workspace path: a
``.float()`` copy of the bf16 logits, ``moe_softmax_kernel<256>`` (softmax into an fp32 workspace)
and ``moe_topk_fast_kernel<256>`` (top-2 per pass, ``ceil(k / 2)`` passes). This module computes the
same ``topk_ids`` / ``topk_weights`` in one Triton launch (one program per token) and keeps every
rounding point and every tie decision of those two kernels, read from their source and from the
SASS of the ``--use_fast_math`` build (minisgl__moe_topk_softmax.so, sm_100):

softmax (moe_softmax_kernel, TPB = 256; thread ``t`` owns experts ``t, t + 256, ...``)
  * ``m`` = row max (exact).
  * ``e_i = __expf(x_i - m)`` = ``MUFU.EX2(FMUL.FTZ(FADD.FTZ(x_i, -m), 0x3FB8AA3B))``, i.e. PTX
    ``ex2.approx.ftz.f32((x_i - m) * float32(log2 e))``.
  * per thread ``s_t = ((0 + e_t) + e_{t+256}) + ...`` in expert order; cub BlockReduce
    (WARP_REDUCTIONS): ``shfl.down`` 1, 2, 4, 8, 16 inside each warp (an adjacent-pair balanced
    tree over the 32 lanes), then warp 0 folds the 8 warp sums left to right.
  * ``nf = MUFU.RCP(denominator)`` (``rcp.approx.ftz.f32``); ``p_i = FMUL.FTZ(nf, e_i)``.

top-k (moe_topk_fast_kernel, TPB = 256)
  * each pass: every thread keeps a (max, second) pair over its experts in expert order with strict
    ``>`` (initial pairs ``{key 0, -1.0}``); cub BlockReduce with ``TopKPairArgMax`` (same tree as
    above); slot ``2c`` gets the max, slot ``2c + 1`` the second; both are set to -1.0;
  * ``row_sum`` adds the selected values in slot order; renormalize = ``FMUL.FTZ(MUFU.RCP(row_sum), w)``.

``TopKPairArgMax`` is not associative under ties, so the tree shape decides which tied expert
is chosen. The kernel reproduces the tree result in closed form with exact (order-independent)
max reductions:
  * max: the rightmost maximum (ties go to the right operand) = the largest thread ``tM`` among
    the threads whose pair max equals the global max; within a thread the lower expert wins.
  * second: along ``tM``'s path to the root a sibling subtree's max replaces the running second
    whenever ``sibling.max >= running`` (both branches of the operator agree on that), starting from
    ``tM``'s own second. Hence the second is the highest sibling that holds the second value; sibling
    order, bottom-up: the in-warp lane blocks (by the highest bit of ``lane ^ lane(tM)``), the warps
    before ``tM``'s warp (one sibling, when ``tM`` is not in warp 0), then each later warp. With no
    such sibling it is ``tM``'s own second.
``qwen35_softmax_topk_reference`` is the same algorithm in PyTorch (the CPU lane checks it against a
step-by-step replay of the cub reductions); on CPU the fast-math instructions take exact
stand-ins (``FAST_PTX=False``) because their hardware approximations cannot be computed on a CPU.

Supported: ``256 < num_experts <= 1024`` with ``num_experts % 256 == 0`` and ``2 <= top_k <= 32``
(``top_k == 1`` takes ``moe_topk_kernel``, a different reduction, so it keeps the old path).
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

# float32(log2(e)) = 0x3FB8AA3B: the FMUL immediate in the SASS of moe_softmax_kernel (__expf).
LOG2E_F32 = 1.4426950216293334961
FLT_MIN = 1.1754943508222875e-38
_LANES = 256  # TPB of moe_softmax_kernel / moe_topk_fast_kernel
_MAX_TOP_K = 32


def qwen35_fused_router_supported(num_experts: int, top_k: int) -> bool:
    """Shapes for which the fused kernel replaces topk_softmax's workspace fallback."""
    return (
        256 < num_experts <= 1024
        and num_experts % _LANES == 0
        and 2 <= top_k <= min(_MAX_TOP_K, num_experts)
    )


# ---------------------------------------------------------------------------------------------
# Fast-math instructions of the CUDA build. FAST_PTX=True emits the same PTX; False (the Triton
# interpreter on CPU) uses exact stand-ins with flush-to-zero, which qwen35_softmax_topk_reference
# mirrors bit for bit.
# ---------------------------------------------------------------------------------------------
@triton.jit
def _ftz(x):
    # the sign of a flushed zero is irrelevant here: every use is >= 0 or an exp2 input
    return tl.where(tl.abs(x) < 1.1754943508222875e-38, 0.0, x)


@triton.jit
def _expf_fast(d, FAST_PTX: tl.constexpr):
    """__expf(d) under --use_fast_math: FMUL.FTZ by log2(e), then MUFU.EX2."""
    if FAST_PTX:
        return tl.inline_asm_elementwise(
            "{ .reg .f32 t; mul.ftz.f32 t, $1, 0f3FB8AA3B; ex2.approx.ftz.f32 $0, t; }",
            "=f,f", [d], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return _ftz(tl.exp2(_ftz(_ftz(d) * 1.4426950216293334961)))


@triton.jit
def _rcp_fast(x, FAST_PTX: tl.constexpr):
    """1.0f / x under --use_fast_math: MUFU.RCP."""
    if FAST_PTX:
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return _ftz(1.0 / _ftz(x))


@triton.jit
def _mul_fast(a, b, FAST_PTX: tl.constexpr):
    """FMUL.FTZ."""
    if FAST_PTX:
        return tl.inline_asm_elementwise(
            "mul.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return _ftz(_ftz(a) * _ftz(b))


@triton.jit
def _row(t, col, j):
    """Row ``j`` of a [COLS, 256] tensor (one non-zero term per sum: exact)."""
    return tl.sum(tl.where(col[:, None] == j, t, tl.zeros_like(t)), axis=0)


@triton.jit
def _qwen35_softmax_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    stride_row,
    E: tl.constexpr,
    COLS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    RENORM: tl.constexpr,
    FAST_PTX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, 256)                         # thread of the 256-thread CUDA block
    col = tl.arange(0, COLS)                         # the thread's j-th expert is j * 256 + lane
    expert = col[:, None] * 256 + lane[None, :]      # [COLS, 256]
    valid = expert < E

    # ---- moe_softmax_kernel -------------------------------------------------------------------
    x = tl.load(logits_ptr + row * stride_row + expert, mask=valid,
                other=float("-inf")).to(tl.float32)
    m = tl.max(x)
    e = tl.where(valid, _expf_fast(x - m, FAST_PTX), 0.0)
    s = tl.zeros([256], dtype=tl.float32)            # thread_data: 0, then the experts in order
    for j in tl.static_range(COLS):
        s = s + _row(e, col, j)
    v = tl.reshape(s, (8, 32))                       # [warp, lane]; shfl.down 1, 2, 4, 8, 16
    v = tl.sum(tl.reshape(v, (8, 16, 2)), axis=2)    # each sum is one add of two lanes
    v = tl.sum(tl.reshape(v, (8, 8, 2)), axis=2)
    v = tl.sum(tl.reshape(v, (8, 4, 2)), axis=2)
    v = tl.sum(tl.reshape(v, (8, 2, 2)), axis=2)
    warp_sum = tl.sum(v, axis=1)                     # [8]
    widx = tl.arange(0, 8)
    denom = tl.sum(tl.where(widx == 0, warp_sum, 0.0))
    for w in tl.static_range(1, 8):                  # warp 0 folds the warp sums left to right
        denom = denom + tl.sum(tl.where(widx == w, warp_sum, 0.0))
    nf = _rcp_fast(denom, FAST_PTX)
    p = tl.where(valid, _mul_fast(e, nf, FAST_PTX), -1.0)

    # ---- moe_topk_fast_kernel -----------------------------------------------------------------
    lane64 = lane.to(tl.int64)
    warp_of = lane // 32
    lane_in = lane % 32
    slot = tl.arange(0, TOPK_P2)
    w_out = tl.zeros([TOPK_P2], dtype=tl.float32)
    i_out = tl.zeros([TOPK_P2], dtype=tl.int32)
    row_sum = tl.sum(tl.zeros([8], dtype=tl.float32))
    for c in tl.static_range((TOPK + 1) // 2):
        # per-thread (max, second) over the thread's experts in order, strict >
        mxv = tl.full([256], -1.0, tl.float32)
        mxk = tl.zeros([256], dtype=tl.int32)
        scv = tl.full([256], -1.0, tl.float32)
        sck = tl.zeros([256], dtype=tl.int32)
        for j in tl.static_range(COLS):
            pv = _row(p, col, j)
            eid = lane + j * 256
            gt = pv > mxv
            scv_n = tl.where(gt, mxv, tl.where(pv > scv, pv, scv))
            sck = tl.where(gt, mxk, tl.where(pv > scv, eid, sck))
            scv = scv_n
            mxv = tl.where(gt, pv, mxv)
            mxk = tl.where(gt, eid, mxk)
        # Candidates are >= +0.0 (removed / absent = -1.0), so their float bits order like the
        # values. Key = value bits | tie order | expert id; the max picks value, then tie order.
        bits = mxv.to(tl.int32, bitcast=True).to(tl.int64)
        cand = mxv >= 0.0
        key1 = tl.where(cand, (bits << 32) | (lane64 << 16) | mxk.to(tl.int64), -1)
        best1 = tl.max(key1, axis=0)                 # rightmost maximum
        v1 = (best1 >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        t_m = ((best1 >> 16) & 0xFF).to(tl.int32)
        k_m = (best1 & 0xFFFF).to(tl.int32)
        # tM's own second (one lane contributes: an exact integer sum)
        sbits = scv.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        key2 = tl.sum(tl.where(lane == t_m, (sbits << 32) | sck.to(tl.int64), 0), axis=0)
        sc_v = (key2 >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        sc_k = (key2 & 0xFFFF).to(tl.int32)
        # siblings on tM's path: other warps beat tM's warp (later warp, then larger lane);
        # inside tM's warp the higher lane block (highest bit of lane ^ lane(tM)), then larger lane
        xl = lane_in ^ (t_m % 32)
        hb = (xl >= 2).to(tl.int32) + (xl >= 4).to(tl.int32) + (xl >= 8).to(tl.int32) \
            + (xl >= 16).to(tl.int32)
        order = tl.where(warp_of != t_m // 32, 0x8000 | lane, (hb << 8) | lane)
        key3 = tl.where(cand & (lane != t_m),
                        (bits << 32) | (order.to(tl.int64) << 16) | mxk.to(tl.int64), -1)
        best3 = tl.max(key3, axis=0)
        vo = tl.where(best3 >= 0, (best3 >> 32).to(tl.int32).to(tl.float32, bitcast=True), -1.0)
        k_o = tl.where(best3 >= 0, (best3 & 0xFFFF).to(tl.int32), 0)
        take_sibling = vo >= sc_v
        s_v = tl.where(take_sibling, vo, sc_v)
        s_k = tl.where(take_sibling, k_o, sc_k)

        w_out = tl.where(slot == 2 * c, v1, w_out)
        i_out = tl.where(slot == 2 * c, k_m, i_out)
        row_sum = row_sum + v1
        p = tl.where(expert == k_m, -1.0, p)
        if 2 * c + 1 < TOPK:
            w_out = tl.where(slot == 2 * c + 1, s_v, w_out)
            i_out = tl.where(slot == 2 * c + 1, s_k, i_out)
            row_sum = row_sum + s_v
            p = tl.where(expert == s_k, -1.0, p)

    if RENORM:
        w_out = _mul_fast(w_out, _rcp_fast(row_sum, FAST_PTX), FAST_PTX)
    smask = slot < TOPK
    tl.store(weights_ptr + row * TOPK + slot, w_out, mask=smask)
    tl.store(ids_ptr + row * TOPK + slot, i_out, mask=smask)


def qwen35_softmax_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """Fused ``topk_softmax`` for ``qwen35_fused_router_supported`` shapes: fills ``topk_weights``
    (fp32) and ``topk_ids`` (int32), both ``[num_tokens, top_k]``, from bf16 / fp32 logits."""
    num_tokens, num_experts = gating_output.shape
    top_k = topk_ids.shape[1]
    if not qwen35_fused_router_supported(num_experts, top_k):
        raise ValueError(f"qwen35_softmax_topk: unsupported num_experts={num_experts} top_k={top_k}")
    assert gating_output.dtype in (torch.bfloat16, torch.float32)
    assert gating_output.stride(1) == 1
    assert topk_weights.is_contiguous() and topk_ids.is_contiguous()
    assert topk_weights.dtype == torch.float32 and topk_ids.dtype == torch.int32
    if num_tokens == 0:
        return
    cols = triton.next_power_of_2(triton.cdiv(num_experts, _LANES))
    _qwen35_softmax_topk_kernel[(num_tokens,)](
        gating_output, topk_weights, topk_ids, gating_output.stride(0),
        E=num_experts, COLS=cols, TOPK=top_k, TOPK_P2=triton.next_power_of_2(top_k),
        RENORM=bool(renormalize), FAST_PTX=bool(gating_output.is_cuda),
        num_warps=1,
    )


# ---------------------------------------------------------------------------------------------
# PyTorch statement of the kernel (CPU stand-ins for the fast-math instructions).
# ---------------------------------------------------------------------------------------------
def _ftz_t(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.abs() < FLT_MIN, torch.zeros_like(x), x)


def _exp2_numpy(x: torch.Tensor) -> torch.Tensor:
    """float32 exp2 as the Triton interpreter computes ``tl.exp2`` (numpy)."""
    return torch.from_numpy(np.exp2(x.detach().cpu().numpy())).to(x.device)


def reference_expf(d: torch.Tensor, exp2=_exp2_numpy) -> torch.Tensor:
    return _ftz_t(exp2(_ftz_t(_ftz_t(d) * LOG2E_F32)))


def reference_rcp(x: torch.Tensor) -> torch.Tensor:
    return _ftz_t(1.0 / _ftz_t(x))


def reference_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _ftz_t(_ftz_t(a) * _ftz_t(b))


def _hibit5(x: torch.Tensor) -> torch.Tensor:
    return (x >= 2).long() + (x >= 4).long() + (x >= 8).long() + (x >= 16).long()


def qwen35_softmax_topk_reference(
    gating_output: torch.Tensor,
    top_k: int,
    renormalize: bool,
    *,
    exp2=_exp2_numpy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's algorithm in PyTorch: ``(weights fp32, ids int32)``, ``[num_tokens, top_k]``."""
    x = gating_output.float()
    num_tokens, num_experts = x.shape
    cols = -(-num_experts // _LANES)
    width = cols * _LANES
    lane = torch.arange(_LANES, device=x.device)
    expert = (torch.arange(cols, device=x.device)[:, None] * _LANES + lane[None, :])
    valid = expert < num_experts                                         # [cols, 256]
    xp = torch.full((num_tokens, width), float("-inf"), device=x.device)
    xp[:, :num_experts] = x
    xp = xp.view(num_tokens, cols, _LANES)

    m = x.max(dim=1).values
    e = torch.where(valid, reference_expf(xp - m[:, None, None], exp2), torch.zeros_like(xp))
    s = torch.zeros(num_tokens, _LANES, device=x.device)
    for j in range(cols):
        s = s + e[:, j]
    v = s.view(num_tokens, 8, 32)
    while v.shape[-1] > 1:
        v = v[..., 0::2] + v[..., 1::2]
    warp_sum = v[..., 0]
    denom = warp_sum[:, 0]
    for w in range(1, 8):
        denom = denom + warp_sum[:, w]
    p = torch.where(valid, reference_mul(e, reference_rcp(denom)[:, None, None]),
                    torch.full_like(e, -1.0))

    warp_of, lane_in = lane // 32, lane % 32
    rows = torch.arange(num_tokens, device=x.device)
    weights = torch.zeros(num_tokens, top_k, device=x.device)
    ids = torch.zeros(num_tokens, top_k, dtype=torch.long, device=x.device)
    row_sum = torch.zeros(num_tokens, device=x.device)
    neg = torch.full((num_tokens, _LANES), -1.0, device=x.device)
    for c in range((top_k + 1) // 2):
        mxv, scv = neg.clone(), neg.clone()
        mxk = torch.zeros(num_tokens, _LANES, dtype=torch.long, device=x.device)
        sck = mxk.clone()
        for j in range(cols):
            pv, eid = p[:, j], (lane + j * _LANES).expand(num_tokens, -1)
            gt, gs = pv > mxv, pv > scv
            scv, sck = torch.where(gt, mxv, torch.where(gs, pv, scv)), \
                torch.where(gt, mxk, torch.where(gs, eid, sck))
            mxv, mxk = torch.where(gt, pv, mxv), torch.where(gt, eid, mxk)
        cand = mxv >= 0
        v1 = torch.where(cand, mxv, neg).max(dim=1).values
        t_m = torch.where(cand & (mxv == v1[:, None]), lane, -1).max(dim=1).values
        k_m = mxk[rows, t_m]
        sc_v, sc_k = scv[rows, t_m], sck[rows, t_m]
        others = cand & (lane[None, :] != t_m[:, None])
        vo = torch.where(others, mxv, neg).max(dim=1).values
        hb = _hibit5(lane_in[None, :] ^ (t_m % 32)[:, None])
        order = torch.where(warp_of[None, :] != (t_m // 32)[:, None], 0x8000 | lane[None, :],
                            (hb << 8) | lane[None, :])
        best = torch.where(others & (mxv == vo[:, None]), order, -1).max(dim=1).values
        k_o = torch.where(best >= 0, mxk[rows, (best & 0xFF).clamp(min=0)], 0)
        vo = torch.where(best >= 0, vo, -1.0)
        take = vo >= sc_v
        s_v, s_k = torch.where(take, vo, sc_v), torch.where(take, k_o, sc_k)

        flat = p.view(num_tokens, -1)
        weights[:, 2 * c], ids[:, 2 * c] = v1, k_m
        row_sum = row_sum + v1
        flat[rows, k_m] = -1.0
        if 2 * c + 1 < top_k:
            weights[:, 2 * c + 1], ids[:, 2 * c + 1] = s_v, s_k
            row_sum = row_sum + s_v
            flat[rows, s_k] = -1.0
    if renormalize:
        weights = reference_mul(weights, reference_rcp(row_sum)[:, None])
    return weights, ids.to(torch.int32)


__all__ = [
    "LOG2E_F32",
    "qwen35_fused_router_supported",
    "qwen35_softmax_topk",
    "qwen35_softmax_topk_reference",
    "reference_expf",
    "reference_mul",
    "reference_rcp",
]
