"""Gated RMSNorm of the Qwen3.5 GDN output as one Triton kernel (MINISGL_QWEN35_FUSED_GATED_NORM=1).

Forward-only port of the FLA ``layer_norm_fwd_kernel`` with ``IS_RMS_NORM``, ``HAS_Z``,
``NORM_BEFORE_GATE`` and the swish activation, as vLLM ships it in
vllm/third_party/flash_linear_attention/ops/layernorm_guard.py (vLLM: Apache-2.0; the FLA code
there is MIT, Copyright (c) 2024, Tri Dao). Changes for FastAFD: no bias / mean / rstd outputs, and
rows are (token, value head) pairs read through the token and head strides, so a ``z`` that is a
column slice of a merged projection output (e.g. ``[B, 64, 128]`` with strides ``(20480, 128, 1)``)
is read in place, without a ``.contiguous()`` copy.

Semantics are those of ``qwen35_gdn._qwen35_rmsnorm_gated`` (the eager reference), per row of the
last dimension, in fp32: ``y = ((x * rsqrt(mean(x^2) + eps)) * w) * (z * sigmoid(z))``, stored
in ``x.dtype``. The reductions, rsqrt and sigmoid round differently from the eager / inductor
kernels, so the output agrees within bf16 rounding, not bit for bit.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_MAX_BLOCK_N = 1024


@triton.jit
def _qwen35_gated_rmsnorm_kernel(
    X,
    Z,
    W,
    Y,
    stride_x_tok,
    stride_x_head,
    stride_z_tok,
    stride_z_head,
    M,
    H,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    tok = (rows // H).to(tl.int64)
    head = (rows % H).to(tl.int64)
    x_off = tok * stride_x_tok + head * stride_x_head
    z_off = tok * stride_z_tok + head * stride_z_head

    x = tl.load(X + x_off[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    z = tl.load(Z + z_off[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    y *= z * tl.sigmoid(z)
    tl.store(Y + rows.to(tl.int64)[:, None] * N + cols[None, :], y, mask=mask)


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _rows_per_block(num_rows: int, device: torch.device) -> int:
    """vLLM's calc_rows_per_block: next_pow2(cdiv(M, 2 * SMs)), capped at 4."""
    if device.type != "cuda":
        return 4
    sm = _sm_count(device.index if device.index is not None else torch.cuda.current_device())
    return min(triton.next_power_of_2(triton.cdiv(num_rows, 2 * sm)), 4)


def _as_rows(t: torch.Tensor) -> tuple[int, int, int, int]:
    """(tokens, heads, token stride, head stride) of a [T, D] or [T, H, D] tensor."""
    if t.dim() == 2:
        return t.shape[0], 1, t.stride(0), 0
    return t.shape[0], t.shape[1], t.stride(0), t.stride(1)


def qwen35_gated_norm_supported(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() in (2, 3)
        and z.shape == x.shape
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.stride(-1) == 1
        and z.stride(-1) == 1
        and weight.dim() == 1
        and weight.shape[0] == x.shape[-1]
        and weight.stride(0) == 1
        and x.shape[-1] <= _MAX_BLOCK_N
    )


def qwen35_gated_rmsnorm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``_qwen35_rmsnorm_gated(x, z, weight, eps)`` in one launch; returns a contiguous tensor
    shaped like ``x``. ``x`` / ``z`` are read by stride (last dim must be unit-stride)."""
    if not qwen35_gated_norm_supported(x, z, weight):
        raise ValueError(
            f"qwen35_gated_rmsnorm: unsupported x {tuple(x.shape)}/{x.stride()} {x.dtype}, "
            f"z {tuple(z.shape)}/{z.stride()}, weight {tuple(weight.shape)}")
    n = x.shape[-1]
    tokens, heads, sx_t, sx_h = _as_rows(x)
    _, _, sz_t, sz_h = _as_rows(z)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    num_rows = tokens * heads
    if num_rows == 0:
        return out
    block_n = triton.next_power_of_2(n)
    rows_per_block = _rows_per_block(num_rows, x.device)
    _qwen35_gated_rmsnorm_kernel[(triton.cdiv(num_rows, rows_per_block),)](
        x, z, weight, out,
        sx_t, sx_h, sz_t, sz_h,
        num_rows, heads, eps,
        N=n, BLOCK_N=block_n, ROWS_PER_BLOCK=rows_per_block,
        num_warps=min(max(block_n // 256, 1), 8),
    )
    return out


def qwen35_gated_rmsnorm_reference(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The kernel's arithmetic in PyTorch (fp32, sum of squares / N, rsqrt, gate after norm)."""
    xf, zf = x.float(), z.float()
    var = (xf * xf).sum(dim=-1, keepdim=True) / x.shape[-1]
    y = xf * torch.rsqrt(var + eps) * weight.float()
    y = y * (zf * torch.sigmoid(zf))
    return y.to(x.dtype)


__all__ = [
    "qwen35_gated_norm_supported",
    "qwen35_gated_rmsnorm",
    "qwen35_gated_rmsnorm_reference",
]
