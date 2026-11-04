"""vLLM db9527a4's KV-cache store for Qwen3.5-397B's full-attention layers (MINISGL_QWEN35_VLLM_FULLATTN):
reshape_and_cache_flash_kernel<bf16, bf16, kAuto> (CUDA source vendored in csrc/jit/vllm_reshape_and_cache_flash.cu).

vLLM's FlashInfer backend stores K / V with torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, k_cache, v_cache,
slot_mapping, "auto", k_scale, v_scale) (v1/attention/backends/flashinfer.py:2673; bf16 KV, --kv-cache-dtype bfloat16).
Here the caches are FastAFD's NHD MHAKVCache views [num_pages, page_size, kv_heads, head_dim] (named layout difference:
vLLM's cache is (B, H, N, 2*hs), which takes the kernel's HND branch).
"""

from __future__ import annotations

import functools

import torch

_SCALE: dict[torch.device, torch.Tensor] = {}


@functools.cache
def _module():
    from minisgl.kernel.utils import load_jit

    return load_jit(
        "vllm_reshape_and_cache_flash",
        cuda_files=["vllm_reshape_and_cache_flash.cu"],
        cuda_wrappers=[("reshape_and_cache_flash", "VllmReshapeAndCacheFlashKernel::run")],
    )


def _unit_scale(device: torch.device) -> torch.Tensor:
    """layer._k_scale / _v_scale of a bf16 cache: [1] fp32 ones (read only by the FP8 branch)."""
    s = _SCALE.get(device)
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("vllm reshape_and_cache_flash: unit scale first requested inside CUDA-graph capture")
        s = torch.ones(1, dtype=torch.float32, device=device)
        _SCALE[device] = s
    return s


def reshape_and_cache_flash(key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                            slot_mapping: torch.Tensor) -> None:
    """key / value [T, H, D] bf16 (row-strided views allowed), caches [num_pages, page_size, H, D], slot_mapping int64."""
    if slot_mapping.dtype != torch.int64:
        raise ValueError(f"vLLM reshape_and_cache_flash takes int64 slot ids, got {slot_mapping.dtype}")
    s = _unit_scale(key.device)
    _module().reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, s, s)


__all__ = ["reshape_and_cache_flash"]
