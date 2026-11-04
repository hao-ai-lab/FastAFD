"""Qwen3.5 attention-side kernels: CPU statements against the paths they replace.

MINISGL_QWEN35_FUSED_ROUTER (kernel/qwen35_router_topk.py)
    ``qwen35_softmax_topk_reference`` (the fused Triton kernel's algorithm: closed-form tie rules,
    reshaped lane trees) must equal, bit for bit, a step-by-step replay of what
    ``topk_softmax`` runs today for 512 experts: moe_softmax_kernel<256> + moe_topk_fast_kernel<256>
    with cub's BlockReduce (WARP_REDUCTIONS: shfl.down 1..16, then warp 0 folds warps 1..7) and
    ``TopKPairArgMax``. Both sides share one CPU stand-in per fast-math instruction (__expf =
    ex2.approx.ftz, MUFU.RCP, FMUL.FTZ), so what is compared is the order of operations, the
    rounding points and the tie decisions -- not the hardware approximations, which the GPU smoke
    (tests/kernel/test_qwen35_fp8_fused_kernels.py) checks against the real kernels.
MINISGL_QWEN35_FUSED_GATED_NORM (kernel/qwen35_gated_norm.py)
    ``qwen35_gated_rmsnorm_reference`` (the FLA kernel's arithmetic) against the current CPU path
    ``qwen35_gdn._qwen35_rmsnorm_gated`` and an fp64 reference, at the Qwen3.5-397B shape [T, 64, 128] bf16,
    with z both contiguous and as the strided column slice of the merged QKVZ output.

The Triton kernels themselves run under TRITON_INTERPRET=1 in tests/kernel/test_qwen35_fp8_interpret.py
(needs Triton, which the CPU lane does not install).
"""

from __future__ import annotations

import pytest
import torch

NUM_EXPERTS = 512
TOP_K = 10
TOKENS = (1, 7, 186, 373)
FLT_MAX = 3.4028234663852886e38
CASES = ("normal", "ties", "all_equal", "large", "small", "dominant", "underflow")


def make_logits(case: str, tokens: int, dtype: torch.dtype, seed: int,
                experts: int = NUM_EXPERTS) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if case == "normal":
        x = torch.randn(tokens, experts, generator=g) * 2.0
    elif case == "ties":  # bf16-exact values on a coarse grid: heavy ties at every rank
        x = torch.randint(-3, 4, (tokens, experts), generator=g).float() * 0.5
    elif case == "all_equal":
        x = torch.full((tokens, experts), 0.75)
    elif case == "large":  # spread far past exp range: most probabilities flush to zero
        x = torch.randn(tokens, experts, generator=g) * 100.0
    elif case == "small":  # bf16 rounding of tiny logits collides many values
        x = torch.randn(tokens, experts, generator=g) * 1e-3
    elif case == "dominant":
        x = torch.randn(tokens, experts, generator=g)
        x[torch.arange(tokens), torch.randint(0, experts, (tokens,), generator=g)] += 60.0
    elif case == "underflow":  # products e * (1 / sum) in the denormal range (FTZ path)
        x = torch.full((tokens, experts), -1000.0)
        x[:, :3] = 0.0
        x[:, 3:40] = -86.0 - torch.rand(tokens, 37, generator=g) * 2.0
    else:
        raise ValueError(case)
    return x.to(dtype)


# ---- replay of moe_softmax_kernel<256> + moe_topk_fast_kernel<256> + cub BlockReduce --------
def _warp_reduce(vals, op):
    """cub WarpReduceShfl over [T, 8, 32] lanes: lane i <- op(own, lane i + off) if i + off <= 31."""
    lane = torch.arange(32)
    for off in (1, 2, 4, 8, 16):
        shifted = tuple(torch.cat([v[..., off:], v[..., -off:]], dim=-1) for v in vals)
        new = op(vals, shifted)
        ok = lane + off <= 31
        vals = tuple(torch.where(ok, n, v) for n, v in zip(new, vals))
    return tuple(v[..., 0] for v in vals)                     # lane 0 of every warp: [T, 8]


def _block_reduce(vals, op):
    """cub BlockReduceWarpReductions, TPB 256: warp reduce, then warp 0 folds warps 1..7."""
    aggs = _warp_reduce(tuple(v.view(v.shape[0], 8, 32) for v in vals), op)
    acc = tuple(a[:, 0] for a in aggs)
    for w in range(1, 8):
        acc = op(acc, tuple(a[:, w] for a in aggs))
    return acc


def _sum_op(lhs, rhs):
    return (rhs[0] + lhs[0],)                                  # @P FADD R, shuffled, own


def _max_op(lhs, rhs):
    return (torch.maximum(lhs[0], rhs[0]),)


def _pair_op(lhs, rhs):
    """moe::TopKPairArgMax on (max key, max value, second key, second value)."""
    lmk, lmv, lsk, lsv = lhs
    rmk, rmv, rsk, rsv = rhs
    take_l = lmv > rmv
    gmk, gmv = torch.where(take_l, lmk, rmk), torch.where(take_l, lmv, rmv)
    from_l = gmk == lmk
    l2 = lsv > rmv
    r2 = rsv > lmv
    gsk = torch.where(from_l, torch.where(l2, lsk, rmk), torch.where(r2, rsk, lmk))
    gsv = torch.where(from_l, torch.where(l2, lsv, rmv), torch.where(r2, rsv, lmv))
    return gmk, gmv, gsk, gsv


def replay_current_path(router, logits: torch.Tensor, top_k: int, renormalize: bool):
    """What topk_softmax runs today for num_experts > 256 (the .float() copy is exact)."""
    expf, rcp, mul = router.reference_expf, router.reference_rcp, router.reference_mul
    x = logits.float()
    tokens, experts = x.shape
    tpb = 256
    # moe_softmax_kernel
    thread_max = torch.full((tokens, tpb), -FLT_MAX)
    for col in range(0, experts, tpb):
        thread_max = torch.maximum(thread_max, x[:, col:col + tpb])
    (row_max,) = _block_reduce((thread_max,), _max_op)
    thread_sum = torch.zeros(tokens, tpb)
    for col in range(0, experts, tpb):
        thread_sum = thread_sum + expf(x[:, col:col + tpb] - row_max[:, None])
    (denom,) = _block_reduce((thread_sum,), _sum_op)
    normalizing = rcp(denom)
    work = mul(normalizing[:, None], expf(x - row_max[:, None]))  # FMUL.FTZ nf, e
    # moe_topk_fast_kernel
    weights = torch.zeros(tokens, top_k)
    ids = torch.zeros(tokens, top_k, dtype=torch.long)
    row_sum = torch.zeros(tokens)
    rows = torch.arange(tokens)
    for k_idx in range((top_k + 1) // 2):
        mk = torch.zeros(tokens, tpb, dtype=torch.long)
        mv = torch.full((tokens, tpb), -1.0)
        sk, sv = mk.clone(), mv.clone()
        for col in range(0, experts, tpb):
            kv = work[:, col:col + tpb]
            key = torch.arange(col, col + tpb).expand(tokens, -1)
            gt = kv > mv
            gs = kv > sv
            sk, sv = torch.where(gt, mk, torch.where(gs, key, sk)), \
                torch.where(gt, mv, torch.where(gs, kv, sv))
            mk, mv = torch.where(gt, key, mk), torch.where(gt, kv, mv)
        gmk, gmv, gsk, gsv = _block_reduce((mk, mv, sk, sv), _pair_op)
        for i, (k, v) in enumerate(((gmk, gmv), (gsk, gsv))):
            slot = 2 * k_idx + i
            if slot >= top_k:
                break
            work[rows, k] = -1.0
            weights[:, slot], ids[:, slot] = v, k
            row_sum = row_sum + v
    if renormalize:
        weights = mul(rcp(row_sum)[:, None], weights)
    return weights, ids.to(torch.int32)


# ---- router -----------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("tokens", TOKENS)
def test_router_reference_matches_current_kernels_bitwise(router, tokens, renormalize, dtype):
    for n, case in enumerate(CASES):
        logits = make_logits(case, tokens, dtype, seed=1000 * tokens + n)
        want_w, want_i = replay_current_path(router, logits, TOP_K, renormalize)
        got_w, got_i = router.qwen35_softmax_topk_reference(logits, TOP_K, renormalize)
        assert torch.equal(got_i, want_i), f"{case}: ids differ"
        assert torch.equal(got_w.view(torch.int32), want_w.view(torch.int32)), \
            f"{case}: weights differ (max {(got_w - want_w).abs().max().item():.3e})"


@pytest.mark.parametrize("experts,top_k", [(768, 8), (1024, 10), (512, 2), (512, 9), (512, 32)])
def test_router_reference_other_shapes(router, experts, top_k):
    for n, case in enumerate(("normal", "ties", "all_equal", "large")):
        logits = make_logits(case, 37, torch.bfloat16, seed=n, experts=experts)
        for renormalize in (True, False):
            want_w, want_i = replay_current_path(router, logits, top_k, renormalize)
            got_w, got_i = router.qwen35_softmax_topk_reference(logits, top_k, renormalize)
            assert torch.equal(got_i, want_i), (case, renormalize)
            assert torch.equal(got_w.view(torch.int32), want_w.view(torch.int32)), (case, renormalize)


def test_router_ties_take_the_cub_tree_choice(router):
    """All 512 equal: the tree, not 'lowest index', decides; pin the first slots."""
    logits = make_logits("all_equal", 2, torch.bfloat16, seed=0)
    _, ids = router.qwen35_softmax_topk_reference(logits, TOP_K, True)
    want_w, want_i = replay_current_path(router, logits, TOP_K, True)
    assert torch.equal(ids, want_i)
    # max: thread 255 (rightmost), its lower expert 255; second: the last warp-fold sibling
    # (warp 6's rightmost thread 223) -- the current kernels' answer, not torch.topk's.
    assert ids[0, :2].tolist() == [255, 223]
    assert len(set(ids[0].tolist())) == TOP_K


@pytest.mark.parametrize("renormalize", [True, False])
def test_router_reference_agrees_with_softmax_topk(router, renormalize):
    """Semantics: the selected experts are a valid top-k of softmax, weights within fp32 noise."""
    for n, case in enumerate(("normal", "ties", "small", "dominant", "large")):
        logits = make_logits(case, 186, torch.bfloat16, seed=7 + n)
        w, ids = router.qwen35_softmax_topk_reference(logits, TOP_K, renormalize)
        prob = torch.softmax(logits.double(), dim=-1)
        top = torch.topk(prob, TOP_K, dim=-1).values
        sel = prob.gather(1, ids.long())
        assert torch.all(torch.diff(torch.sort(ids, dim=-1).values, dim=-1) > 0), case
        torch.testing.assert_close(torch.sort(sel, dim=-1, descending=True).values, top,
                                   rtol=1e-4, atol=1e-7, msg=case)
        if renormalize:
            sel = sel / sel.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(w.double(), sel, rtol=1e-4, atol=1e-6, msg=case)


def test_router_support_predicate(router):
    ok = router.qwen35_fused_router_supported
    assert ok(512, 10) and ok(1024, 8) and ok(768, 2)
    assert not ok(256, 8)    # fused topk_gating v2 path stays
    assert not ok(512, 1)    # moe_topk_kernel (cub ArgMax) stays
    assert not ok(384, 8) and not ok(2048, 8) and not ok(512, 33)


# ---- gated norm --------------------------------------------------------------------------------
HEADS, HEAD_DIM, EPS = 64, 128, 1e-6
QKV_WIDTH = 2 * 16 * 128 + HEADS * HEAD_DIM      # merged QKVZ output: qkv 12288 | z 8192


def _norm_inputs(tokens: int, seed: int, strided_z: bool):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(tokens, HEADS, HEAD_DIM, generator=g) * 0.7).to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g)).to(torch.bfloat16)
    row = QKV_WIDTH + HEADS * HEAD_DIM
    z_full = (torch.randn(tokens, row, generator=g) * 2.0).to(torch.bfloat16)
    # the merged-QKVZ column slice, strides (20480, 128, 1) even for one token
    z = z_full.as_strided((tokens, HEADS, HEAD_DIM), (row, HEAD_DIM, 1), QKV_WIDTH)
    if not strided_z:
        z = z.contiguous()
    return x, z, weight


def _fp64_reference(x, z, weight):
    xd, zd = x.double(), z.double()
    y = xd / torch.sqrt((xd * xd).mean(-1, keepdim=True) + EPS) * weight.double()
    return y * (zd * torch.sigmoid(zd))


@pytest.mark.parametrize("tokens", TOKENS)
def test_gated_norm_reference_matches_current_path(gdn, gated_norm, tokens):
    x, z, weight = _norm_inputs(tokens, seed=tokens, strided_z=True)
    assert z.stride() == (QKV_WIDTH + HEADS * HEAD_DIM, HEAD_DIM, 1)
    assert gated_norm.qwen35_gated_norm_supported(x, z, weight)
    got = gated_norm.qwen35_gated_rmsnorm_reference(x, z, weight, EPS)
    current = gdn.qwen35_rmsnorm_gated(x, z, weight, EPS)     # CPU -> eager reference
    exact = _fp64_reference(x, z, weight)
    ulp = torch.finfo(torch.bfloat16).eps                       # 2^-7, relative spacing at 1
    # both are fp32 math rounded once to bf16: at most one bf16 rounding step from fp64
    for name, out in (("fused-reference", got), ("current", current)):
        err = (out.double() - exact).abs()
        assert torch.all(err <= exact.abs() * ulp / 2 * 1.01 + 1e-30), name
    diff = (got.double() - current.double()).abs()
    assert torch.all(diff <= current.double().abs() * ulp + 1e-30)
    got_contig = gated_norm.qwen35_gated_rmsnorm_reference(x, z.contiguous(), weight, EPS)
    assert torch.equal(got, got_contig)


def test_gated_norm_support_predicate(gated_norm):
    x, z, weight = _norm_inputs(3, seed=0, strided_z=True)
    ok = gated_norm.qwen35_gated_norm_supported
    assert ok(x, z, weight) and ok(x.view(3 * HEADS, HEAD_DIM), z.contiguous().view(-1, HEAD_DIM), weight)
    every_other = torch.zeros(3, HEADS, 2 * HEAD_DIM, dtype=torch.bfloat16)[..., ::2]
    assert not ok(x, every_other, weight)                        # last dim not unit-stride
    assert not ok(x, z[:, :32], weight)                          # shape mismatch
    assert gated_norm._as_rows(z) == (3, HEADS, QKV_WIDTH + HEADS * HEAD_DIM, HEAD_DIM)
