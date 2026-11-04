from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from .utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _jit_module() -> Module:
    return load_jit(
        "minimax_qk_norm",
        cuda_files=["minimax_qk_norm.cu"],
        cuda_wrappers=[
            ("launch", "MiniMaxQKNormKernel::run"),
            ("launch_rope_store", "MiniMaxQKNormKernel::run_rope_store"),
            ("launch_rope_store_scalar", "MiniMaxQKNormKernel::run_rope_store_scalar"),
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


def minimax_qk_cross_head_norm_inplace(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    q_dim: int,
    kv_dim: int,
    eps: float,
) -> None:
    """MiniMax M2 in-place cross-head Q/K RMSNorm over a merged QKV tensor.

    This preserves MiniMax's fp32 normalization arithmetic but writes the
    normalized Q and K back into the original bf16/fp16 QKV buffer. V is left
    unchanged. TP>1 still needs the old all-reduce path and should not call
    this helper.
    """
    if not qkv.is_cuda:
        raise RuntimeError("minimax_qk_cross_head_norm_inplace expects CUDA qkv")
    if qkv.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"unsupported qkv dtype: {qkv.dtype}")
    if q_weight.dtype != qkv.dtype or k_weight.dtype != qkv.dtype:
        raise RuntimeError("MiniMax q/k norm weights must match qkv dtype")
    if qkv.stride(-1) != 1:
        raise RuntimeError("qkv last dim must be contiguous")
    _jit_module().launch(qkv, q_weight, k_weight, int(q_dim), int(kv_dim), float(eps))


def minimax_qk_cross_head_norm_rope_store_inplace(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    q_dim: int,
    kv_dim: int,
    head_dim: int,
    eps: float,
) -> None:
    """MiniMax cross-head Q/K RMSNorm + RoPE + KV-cache store.

    This is the MiniMax equivalent of the generic qk_norm_rope_store kernel.
    MiniMax's norm is over all local Q or K heads, so it cannot reuse the
    per-head Qwen kernel.
    """
    if not qkv.is_cuda:
        raise RuntimeError("minimax_qk_cross_head_norm_rope_store_inplace expects CUDA qkv")
    if qkv.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"unsupported qkv dtype: {qkv.dtype}")
    if q_weight.dtype != qkv.dtype or k_weight.dtype != qkv.dtype:
        raise RuntimeError("MiniMax q/k norm weights must match qkv dtype")
    if k_cache.dtype != qkv.dtype or v_cache.dtype != qkv.dtype:
        raise RuntimeError("MiniMax KV cache dtype must match qkv dtype")
    if qkv.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise RuntimeError("MiniMax qkv/cache last dim must be contiguous")
    if positions.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"positions must be int32 or int64, got {positions.dtype}")
    if out_loc.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"out_loc must be int32 or int64, got {out_loc.dtype}")
    if cos_sin_cache.dtype != torch.float32:
        raise RuntimeError(f"cos_sin_cache must be fp32, got {cos_sin_cache.dtype}")
    _jit_module().launch_rope_store(
        qkv,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        k_cache,
        v_cache,
        out_loc,
        int(q_dim),
        int(kv_dim),
        int(head_dim),
        float(eps),
    )


__all__ = [
    "minimax_qk_cross_head_norm_inplace",
    "minimax_qk_cross_head_norm_rope_store_inplace",
]
