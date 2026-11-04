"""Triton FP8 grouped MoE kernel — mirrors vLLM 0.19.0.

Phase 2 of `tasks/fp8_integration/`. Reproduces vLLM's pure-Python +
Triton FP8 grouped MoE path so AFD can run FP8 weights through native
FP8 tensor-core math (vs Phase 1's β-dequant→BF16 path).

------------------------------------------------------------------------
Attribution (vendoring per feedback_vendoring_convention)
------------------------------------------------------------------------
Source: vLLM 0.19.0 (pip-installed; verified 15,747 tok/s @ 8 GPU
TP=1 DP=8 +EP — see commit 9d649bb).

Triton @jit kernel bodies are copied verbatim from upstream:
  - `write_zeros_to_output`
        vllm/model_executor/layers/fused_moe/fused_moe.py:57-74
  - `fused_moe_kernel`
        vllm/model_executor/layers/fused_moe/fused_moe.py:311-572
        (FP8 path: use_fp8_w8a8=True, group_k=128, group_n=128)
  - `_per_token_group_quant_fp8`
        vllm/model_executor/layers/quantization/utils/fp8_utils.py:609-664

Python host wrappers adapt vLLM's `invoke_fused_moe_triton_kernel` and
`fused_experts_impl` to AFD's existing helpers (`moe_align_block_size`,
`moe_sum_reduce_triton`, `silu_and_mul`). The C++ fast path
`torch.ops._C.per_token_group_fp8_quant` and the column-major /
tma-aligned variants are omitted: this file is pure Python + Triton
per the Phase 2 plan §3 "no external compiled extensions on the MoE
path" constraint.

License: Apache-2.0 (vLLM project).
------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import triton
import triton.language as tl


# FP8 e4m3fn constants on Blackwell / GB200 (see vLLM
# `current_platform.fp8_dtype()` and `get_fp8_min_max()`).
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = -448.0
_FP8_MAX = 448.0


# ============================================================================
# @triton.jit kernels — copied VERBATIM from vLLM 0.19.0.
# Do not modify bodies; product-diff edits live in the host wrappers below.
# ============================================================================


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    # Use multiply-by-reciprocal instead of division to match PyTorch's
    # tensor/scalar division precision (GPU fast-division for constexpr
    # divisors can introduce 1-ULP error that flips FP8 quantization at
    # representable-value boundaries).
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


# ============================================================================
# Python host wrappers.
#
# Adapted from vLLM's `invoke_fused_moe_triton_kernel` (line 725) and
# `fused_experts_impl` (line 1633). Three minimal adaptations vs upstream:
#   1. Only the FP8 (use_fp8_w8a8=True) + block-quant (group=128) path is
#      exposed — other quantization schemes (int8/int4/mxfp4/...) are
#      dropped to keep the surface narrow.
#   2. The C++ fast path `torch.ops._C.per_token_group_fp8_quant` is omitted
#      (vLLM-only compiled extension); we always use the Triton fallback.
#   3. The orchestrator (`fused_moe_fp8_triton`) calls AFD's existing
#      helpers — `moe_align_block_size`, `moe_sum_reduce_triton`,
#      `silu_and_mul` — instead of vLLM's `ops.moe_sum` + internal helpers.
#      This is the "minimum patch" that lets vLLM's kernel run inside AFD's
#      existing MoE skeleton (mirrors AFD's `python/minisgl/moe/fused.py`
#      `fused_experts_impl` shape exactly).
# ============================================================================


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    out_q: torch.Tensor | None = None,
    out_s: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row-group dynamic quantization BF16 → FP8 + per-group scale.

    Adapted from vLLM `vllm/model_executor/layers/quantization/utils/fp8_utils.py`
    `per_token_group_quant_fp8` (the Triton fallback path only).

    Args:
        x: Input tensor (M, K) bf16/fp16/fp32; K must be divisible by group_size.
        group_size: Quant group along last dim. 128 matches the weight block
            scale convention for Qwen3-FP8.
        eps: Floor on absmax to avoid divide-by-zero on all-zero groups.
        out_q: Optional preallocated FP8 output (same shape as `x`, dtype
            float8_e4m3fn). When provided the kernel writes in place; when
            None, `torch.empty` allocates a fresh buffer (legacy path).
            Mirrors vLLM's `vllm/.../fp8_utils.py:per_token_group_quant_fp8`
            `out_q` kwarg. Letting the caller preallocate keeps these
            tensors outside AFD's full-graph cudagraph capture pool, which
            otherwise accumulates one transient slot per call × 6 captured
            BS buckets and diverges from the captured pointer table after
            ~470 sustained-decode steps (see
            tasks/fp8_integration/phase_impl/phase3_perf_debug/forensics/
            ANALYSIS_v3_supplement_phase3_4.md §2).
        out_s: Optional preallocated FP32 scale output (shape
            ``x.shape[:-1] + (x.shape[-1] // group_size,)``). Same semantics
            as `out_q`.

    Returns:
        x_q: FP8 e4m3fn tensor, same shape as x. Identity-equal to `out_q`
            if it was provided.
        x_s: Scale tensor (M, K // group_size), float32. Identity-equal to
            `out_s` if it was provided.
    """
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    scale_shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    if out_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=_FP8_DTYPE)
    else:
        assert out_q.shape == x.shape, (
            f"out_q shape {tuple(out_q.shape)} does not match x.shape {tuple(x.shape)}"
        )
        assert out_q.dtype == _FP8_DTYPE, (
            f"out_q dtype {out_q.dtype} must be {_FP8_DTYPE}"
        )
        assert out_q.device == x.device, (
            f"out_q device {out_q.device} must match x.device {x.device}"
        )
        assert out_q.is_contiguous(), "out_q must be contiguous"
        x_q = out_q
    if out_s is None:
        x_s = torch.empty(scale_shape, device=x.device, dtype=torch.float32)
    else:
        assert tuple(out_s.shape) == tuple(scale_shape), (
            f"out_s shape {tuple(out_s.shape)} does not match expected "
            f"{tuple(scale_shape)}"
        )
        assert out_s.dtype == torch.float32, (
            f"out_s dtype {out_s.dtype} must be torch.float32"
        )
        assert out_s.device == x.device, (
            f"out_s device {out_s.device} must match x.device {x.device}"
        )
        assert out_s.is_contiguous(), "out_s must be contiguous"
        x_s = out_s

    M = x.numel() // group_size
    BLOCK = triton.next_power_of_2(group_size)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    # Row stride for 2D inputs; for >2D, x is reshaped through the .view to
    # 2D before quant in vLLM's orchestrator. We accept 2D here as the
    # MoE host always feeds 2D activations (M, K).
    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[-1],
        x.stride(-2) if x.dim() >= 2 else x.shape[-1],
        eps,
        fp8_min=_FP8_MIN,
        fp8_max=_FP8_MAX,
        use_ue8m0=False,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x_q, x_s


def invoke_fused_moe_triton_fp8_kernel(
    A: torch.Tensor,            # FP8 e4m3fn (M, K)
    B: torch.Tensor,            # FP8 e4m3fn (E, N, K)
    C: torch.Tensor,            # bf16 (M, top_k, N)
    A_scale: torch.Tensor,      # fp32 (M, K // group_k)
    B_scale: torch.Tensor,      # fp32 (E, N // group_n, K // group_k)
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: Any,          # tl dtype
    block_shape: tuple[int, int],  # (group_n, group_k), typically (128, 128)
) -> None:
    """Dispatch `fused_moe_kernel` in the FP8 block-quant path.

    Adapted from vLLM `invoke_fused_moe_triton_kernel` keeping only the
    `use_fp8_w8a8=True, group_k=128, group_n=128` branch.
    """
    assert B_scale is not None and topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert triton.cdiv(B.size(-2), block_shape[0]) == B_scale.size(-2)
    assert triton.cdiv(B.size(-1), block_shape[1]) == B_scale.size(-1)

    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size: each token routes to at most top_k
        # distinct experts, so we never need more than that many BLOCK_SIZE_M
        # blocks regardless of the sorted_token_ids padding.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    fused_moe_kernel[grid](
        A,
        B,
        C,
        None,                                  # b_bias_ptr (HAS_BIAS=False)
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale.ndim >= 2 else 0,
        0,                                     # stride_bbe (HAS_BIAS=False)
        0,                                     # stride_bbn (HAS_BIAS=False)
        block_shape[0],                        # group_n
        block_shape[1],                        # group_k
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        naive_block_assignment=False,
        HAS_BIAS=False,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config,
    )


def fused_moe_fp8_triton(
    hidden_states: torch.Tensor,    # (M, K) bf16
    w1: torch.Tensor,                # (E, 2N, K) float8_e4m3fn  (gate+up)
    w1_scale: torch.Tensor,          # (E, 2N // 128, K // 128) fp32
    w2: torch.Tensor,                # (E, K, N) float8_e4m3fn   (down)
    w2_scale: torch.Tensor,          # (E, K // 128, N // 128) fp32
    topk_weights: torch.Tensor,      # (M, top_k) fp32
    topk_ids: torch.Tensor,          # (M, top_k) int32
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    global_num_experts: int | None = None,
    block_shape: tuple[int, int] = (128, 128),
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    """Triton grouped FP8 MoE forward (``config``: Triton tile config; default as below).

    Adapted from vLLM `fused_experts_impl` keeping only the FP8 path; calls
    AFD's existing `moe_align_block_size`, `moe_sum_reduce_triton`, and
    `silu_and_mul` instead of vLLM's internal helpers.

    Pipeline (mirrors vLLM's structure exactly):
      1. moe_align_block_size on topk_ids
      2. per_token_group_quant_fp8(hidden_states)        bf16 → fp8 + scale
      3. fused_moe_kernel w1 (gate+up GEMM)              fp8×fp8 → bf16
      4. silu_and_mul activation                          (M*topk, 2N) → (M*topk, N)
      5. per_token_group_quant_fp8(intermediate)         bf16 → fp8 + scale
      6. fused_moe_kernel w2 (down GEMM, mul_routed=True) fp8×fp8 → bf16
      7. moe_sum_reduce over top_k                       (M, top_k, K) → (M, K)
    """
    from minisgl.kernel import moe_sum_reduce_cuda
    from minisgl.layers import gelu_and_mul, silu_and_mul
    from minisgl.moe.fused import moe_align_block_size, try_get_optimal_moe_config

    assert hidden_states.is_contiguous()
    assert w1.dtype == _FP8_DTYPE, "w1 must be float8_e4m3fn"
    assert w2.dtype == _FP8_DTYPE, "w2 must be float8_e4m3fn"
    assert hidden_states.shape[1] == w1.shape[2], (
        f"K mismatch: hidden_states {hidden_states.shape[1]} vs w1 {w1.shape[2]}"
    )
    assert topk_weights.shape == topk_ids.shape
    if expert_map is not None and global_num_experts is None:
        raise RuntimeError("global_num_experts is required when expert_map is provided")

    num_tokens, K = hidden_states.shape
    E, N, _ = w1.shape                      # N = 2 * moe_intermediate
    top_k_num = topk_ids.shape[1]
    M = num_tokens
    Mt = M * top_k_num

    if config is None:
        try:
            config = try_get_optimal_moe_config(
                w1.shape,
                w2.shape,
                top_k_num,
                M,
                dtype="fp8_w8a8",
                block_shape=block_shape,
            )
        except TypeError:
            config = try_get_optimal_moe_config(w1.shape, w2.shape, top_k_num, M)

    # Preallocate the four FP8-quant output buffers in one block, then thread
    # them into per_token_group_quant_fp8 via the out_q/out_s kwargs. vLLM does
    # the same in `fused_experts_impl` (see ANALYSIS_v3_supplement_phase3_4.md
    # §2). Lifting these allocations outside the per-call torch.empty path
    # prevents the cudagraph memory pool from churning thousands of transient
    # FP8/FP32 slots per forward, which is the proximate cause of the deferred
    # cudaErrorLaunchFailure observed after ~470 sustained-decode steps
    # (phase3_perf_debug/triage_cuda/CUDA_ERROR_TRIAGE.md).
    qhidden_buf = torch.empty(
        (M, K), device=hidden_states.device, dtype=_FP8_DTYPE
    )
    a1q_scale_buf = torch.empty(
        (M, K // block_shape[1]),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    qint2_buf = torch.empty(
        (Mt, N // 2), device=hidden_states.device, dtype=_FP8_DTYPE
    )
    a2q_scale_buf = torch.empty(
        (Mt, (N // 2) // block_shape[1]),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    # Shared cache across w1/w2 outputs (mirrors AFD's `fused_experts_impl`).
    cache = torch.empty(
        Mt * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if expert_map is not None:
        cache.zero_()
    intermediate_cache1 = cache[: Mt * N].view((M, top_k_num, N))
    intermediate_cache2 = torch.empty(
        (Mt, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: Mt * w2.shape[1]].view(
        (M, top_k_num, w2.shape[1])
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    # === step 1: align ============================================
    num_experts_for_align = global_num_experts if expert_map is not None else E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts_for_align
    )
    if expert_map is not None:
        valid = (expert_ids >= 0) & (expert_ids < global_num_experts)
        safe_ids = torch.where(valid, expert_ids, torch.zeros_like(expert_ids))
        expert_ids = torch.where(
            valid,
            expert_map[safe_ids.to(torch.long)],
            torch.full_like(expert_ids, -1),
        )

    # === step 2: quantize activations ============================
    qhidden_states, a1q_scale = per_token_group_quant_fp8(
        hidden_states,
        group_size=block_shape[1],
        out_q=qhidden_buf,
        out_s=a1q_scale_buf,
    )

    # === step 3: w1 GEMM (gate+up) ===============================
    invoke_fused_moe_triton_fp8_kernel(
        qhidden_states,
        w1,
        intermediate_cache1,
        a1q_scale,
        w1_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        top_k_num,
        config,
        compute_type=compute_type,
        block_shape=block_shape,
    )

    # === step 4: activation ======================================
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    if expert_map is not None:
        intermediate_cache3.zero_()

    # === step 5: quantize intermediate ===========================
    qintermediate2, a2q_scale = per_token_group_quant_fp8(
        intermediate_cache2,
        group_size=block_shape[1],
        out_q=qint2_buf,
        out_s=a2q_scale_buf,
    )

    # === step 6: w2 GEMM (down) ==================================
    invoke_fused_moe_triton_fp8_kernel(
        qintermediate2,
        w2,
        intermediate_cache3,
        a2q_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,                                   # top_k=1 for the down GEMM
        config,
        compute_type=compute_type,
        block_shape=block_shape,
    )

    # === step 7: reduce over top_k ===============================
    out_hidden_states = torch.empty_like(hidden_states)
    moe_sum_reduce_cuda(intermediate_cache3, out_hidden_states)
    return out_hidden_states
