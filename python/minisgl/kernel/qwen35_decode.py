# SPDX-License-Identifier: MIT
# Copyright (c) 2026 sgl-project
"""Qwen3.5 decode kernels and layout-preserving adapters (block-FP8 AFD attention side)."""
from __future__ import annotations

import functools
import os


def shared_quant_qkvz(layer, hidden, prequant=None) -> tuple:
    """Return original QKV/z layouts; retain the original prefill dispatch.

    Only host metadata is inspected in decode, so the path is graph-capturable.
    Quantization policy and both prequant GEMMs are the pinned engine's own.
    """
    import torch
    from minisgl.core import get_global_ctx
    from minisgl.kernel import deepgemm
    from minisgl.layers.linear import _dense_fp8_use_bf16_cublas

    qkv, z = layer.in_proj_qkv, layer.in_proj_z
    if not get_global_ctx().batch.is_decode:
        if prequant is not None:
            raise RuntimeError('Prequantized QKV/z input is decode-only')
        return qkv.forward(hidden), z.forward(hidden)
    if (hidden.ndim != 2 or hidden.shape[0] < 1 or hidden.shape[1] != 4096
            or hidden.dtype != torch.bfloat16 or not hidden.is_cuda):
        raise RuntimeError('Shared quant requires CUDA BF16 [B,4096] decode')
    if os.environ.get('MINISGL_FP8_QUANT_BACKEND', 'cuda') != 'cuda':
        raise RuntimeError('Shared quant requires the pinned CUDA quant backend')
    if _dense_fp8_use_bf16_cublas():
        raise RuntimeError('Shared quant cannot use BF16 cublas fallback')
    for projection, width in ((qkv, 12288), (z, 8192)):
        if (projection.weight.shape != (width, 4096)
                or projection.weight.dtype != torch.float8_e4m3fn
                or projection.weight.device != hidden.device or projection.bias is not None
                or projection._use_bf16_cublas_for_fp8):
            raise RuntimeError('Shared quant requires unchanged FP8 TP1 no-bias QKV/z')
    if prequant is None:
        from minisgl.layers.linear import _vllm_qwen35_quant
        hidden_2d = hidden.contiguous().view(-1, hidden.shape[-1])
        if _vllm_qwen35_quant(hidden_2d):
            # MINISGL_QWEN35_VLLM_QUANT: vLLM's QuantFP8 kernel (kernel/vllm_per_token_group_quant.py)
            from minisgl.kernel.vllm_per_token_group_quant import per_token_group_quant_fp8_packed_for_deepgemm
            values, scales = per_token_group_quant_fp8_packed_for_deepgemm(hidden_2d, 128)
        else:
            values, scales = deepgemm.per_token_cast_to_fp8(
                hidden_2d, use_ue8m0=True, gran_k=128, use_packed_ue8m0=True)
    else:
        # From fused_input_norm_quant: the bytes the cast would produce.
        values, scales = prequant
    if os.environ.get('MINISGL_QWEN35_MERGED_QKVZ') == '1':
        # One FP8 GEMM over q|k|v|z, as vLLM's in_proj_qkvz; the
        # outputs are column views (row stride 20480), which the causal conv reads by stride.
        weight, weight_scale = merged_qkvz_fp8(layer)
        output = torch.empty((values.size(0), weight.size(0)), dtype=torch.bfloat16, device=values.device)
        deepgemm.fp8_gemm_nt((values, scales), (weight, weight_scale), output, disable_ue8m0_cast=False)
        return output[:, :12288], output[:, 12288:]
    kwargs = dict(output_shape_prefix=hidden.shape[:-1], output_dtype=hidden.dtype)
    return (qkv.forward_fp8_prequant(values, scales, **kwargs),
            z.forward_fp8_prequant(values, scales, **kwargs))


def merged_qkvz_fp8(layer) -> tuple:
    """Decode QKV|z as one FP8 weight: (rows [20480, 4096], packed scales), built once.

    Both projections were prepared at load (FP8 128x128 blocks, UE8M0 block scales;
    _LinearTPImpl._fp8_weight_for_deepgemm). 12288 and 8192 are multiples of 128, so no
    block straddles the seam: the concatenated rows and block scales are exactly the
    separate ones, and every output column sees the K blocks and scales of its own GEMM.
    The separate projections are rebound to row views of the merged weight (nothing stored
    twice; prefill keeps its two GEMMs on the same bytes) and keep their own packed scales:
    the MN-major TMA-aligned packed layout cannot be sliced, so the merged copy adds
    20480 x 8 int32 per layer. Must run before graph capture (it allocates).
    """
    import torch
    from minisgl.kernel import deepgemm
    merged = getattr(layer, '_merged_qkvz_fp8', None)
    if merged is not None:
        return merged
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('QKV|z FP8 merging must finish before graph capture')
    qkv, z = layer.in_proj_qkv, layer.in_proj_z
    for projection, width in ((qkv, 12288), (z, 8192)):
        projection._fp8_weight_for_deepgemm()  # load-time preparation, cached: no work here
        scale = projection.weight_scale
        if (projection._fp8_scale_format != 'block' or projection._fp8_block_size != (128, 128)
                or projection.weight.shape != (width, 4096) or not projection.weight.is_contiguous()
                or scale.dtype != torch.float32 or scale.shape != (width // 128, 32)):
            raise RuntimeError('Merged QKV|z requires prepared 128x128 block-scaled FP8 QKV/z')
    block_scale = torch.cat((qkv.weight_scale, z.weight_scale), dim=0)
    weight = _merge_rows(qkv, z)
    weight_scale = deepgemm.transform_sf_into_required_layout(
        block_scale.unsqueeze(0), weight.size(0), weight.size(1), (1, 128, 128), 1, False, False,
    ).squeeze(0)
    layer._merged_qkvz_fp8 = (weight, weight_scale)
    return layer._merged_qkvz_fp8

def merged_ba(layer, hidden):
    import torch
    from minisgl.core import get_global_ctx
    if not get_global_ctx().batch.is_decode:
        return layer.in_proj_b.forward(hidden), layer.in_proj_a.forward(hidden)
    if hidden.ndim != 2 or hidden.shape[1] != 4096 or hidden.dtype != torch.bfloat16:
        raise RuntimeError('Expected BF16 Qwen decode [B,4096]')
    b, a = layer.in_proj_b, layer.in_proj_a
    if (b.weight.dtype != torch.bfloat16 or a.weight.dtype != torch.bfloat16 or
            b.weight.shape != (64, 4096) or a.weight.shape != b.weight.shape or
            b.bias is not None or a.bias is not None):
        raise RuntimeError('Expected unchanged BF16 no-bias b/a weights')
    weight = getattr(layer, '_ba_merged_weight', None)
    if weight is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('b/a packing must finish before graph capture')
        # Row views of one allocation: no second copy.
        weight = _merge_rows(b, a)
        layer._ba_merged_weight = weight
    output = torch.nn.functional.linear(hidden, weight)
    return split_ba(output)

def _merge_rows(first, second):
    """Concatenate two projections' weights; each keeps a contiguous row view of the result."""
    import torch
    rows = first.weight.shape[0]
    merged = torch.cat((first.weight, second.weight), dim=0)
    first.weight, second.weight = merged[:rows], merged[rows:]
    return merged

def split_ba(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the N128 GEMM allocation alive through disjoint column views."""
    import torch
    if (output.ndim != 2 or output.shape[0] < 1 or output.shape[1] != 128
            or output.dtype != torch.bfloat16 or output.stride() != (128, 1)):
        raise ValueError('Expected canonical BF16 N128 GEMM output')
    return output[:, :64], output[:, 64:]

@functools.cache
def _moe_residual_norm_module():
    """JIT-load the fused MoE-add/residual/RMSNorm/FP8 kernel once per process."""
    from minisgl.kernel.utils import load_jit
    # FlashInfer builds its norm module with fast math; the kernel mirrors that
    # and pins the quantization's IEEE operations explicitly.
    return load_jit('moe_residual_rmsnorm_fp8_quant',
                    cuda_files=['moe_residual_rmsnorm_fp8_quant.cu'],
                    cuda_wrappers=[('launch', 'MoeResidualRMSNormFp8QuantKernel::run')],
                    extra_cuda_cflags=['--use_fast_math'])

def fused_input_norm_quant(norm, routed, shared, residual) -> tuple:
    """Decode input of one Qwen3.5 AG layer in one kernel instead of three.

    Adds the previous layer's MoE output ``routed + shared`` to ``residual`` in
    place, applies ``norm`` and quantizes the result, producing the same bytes
    as ``routed + shared`` -> FlashInfer ``fused_add_rmsnorm`` ->
    ``per_token_cast_to_fp8`` (CUDA, packed UE8M0); the device unit compares
    every output byte. Returns (BF16 normed, (FP8 values, packed scales)).
    """
    import torch
    from minisgl.core import get_global_ctx
    if not get_global_ctx().batch.is_decode:
        raise RuntimeError('Fused input norm/quant is decode-only')
    if os.environ.get('MINISGL_FP8_QUANT_BACKEND', 'cuda') != 'cuda':
        raise RuntimeError('Fused input norm/quant replaces only the pinned CUDA quant backend')
    if (routed.ndim != 2 or routed.shape[0] < 1 or routed.shape[1] != 4096 or not routed.is_cuda
            or any(tensor.shape != routed.shape or tensor.dtype != torch.bfloat16
                   or tensor.device != routed.device or not tensor.is_contiguous()
                   for tensor in (routed, shared, residual))):
        raise RuntimeError('Fused input norm/quant requires contiguous CUDA BF16 [B,4096] routed/shared/residual')
    weight = norm.weight
    if (weight.shape != (4096,) or weight.dtype != torch.bfloat16 or weight.device != routed.device
            or not weight.is_contiguous()):
        raise RuntimeError('Fused input norm/quant requires the folded BF16 [4096] norm weight')
    rows = routed.shape[0]
    normed = torch.empty_like(routed)
    values = torch.empty((rows, 4096), dtype=torch.float8_e4m3fn, device=routed.device)
    # DeepGEMM's TMA scale layout, identical to per_token_cast_to_fp8_cuda.
    scales = torch.empty_strided((rows, 8), (1, (rows + 3) // 4 * 4), dtype=torch.int32,
                                 device=routed.device)
    _moe_residual_norm_module().launch(residual, routed, shared, weight, normed,
                                       values.view(torch.uint8), scales, float(norm.eps))
    return normed, (values, scales)


@functools.cache
def _full_attn_norm_rope_bf16_store_module():
    """JIT-load the fused q/k norm + RoPE + BF16 KV store kernel once per process."""
    from minisgl.kernel.utils import load_jit
    # FlashInfer builds its norm and RoPE modules with fast math; the kernel mirrors that.
    return load_jit('qwen35_full_attn_norm_rope_bf16_store',
                    cuda_files=['qwen35_full_attn_norm_rope_bf16_store.cu'],
                    cuda_wrappers=[('launch', 'Qwen35FullAttnNormRopeBf16StoreKernel::run')],
                    extra_cuda_cflags=['--use_fast_math'])


def full_attention_decode(layer, qkv, variant: str) -> tuple:
    """Decode attention of one Qwen3.5 full-attention layer on views of its QKV output.

    ``qkv`` rows are [32 x (query 256 | gate 256) | k 512 | v 512]. The unchanged path
    copies query and gate out of the interleaved rows and concatenates q|k|v before
    attention; every consumer can read strided views instead (FlashInfer norm/RoPE and
    the TRT-LLM decode kernel take row and head strides), so no copy is made. Q/K norm, RoPE
    and the BF16 KV store are fused. The query is normalized and rotated in place inside
    ``qkv``. Returns (attention output [B, H, D], gate view [B, H, D]).
    """
    from minisgl.core import get_global_ctx
    ctx = get_global_ctx()
    batch = ctx.batch
    rows, heads, dim = qkv.shape[0], layer.num_qo_heads, layer.head_dim
    if (not batch.is_decode or qkv.ndim != 2 or qkv.stride(1) != 1
            or qkv.shape[1] != 2 * layer.q_size + 2 * layer.kv_size):
        raise RuntimeError('Full-attention view decode requires decode rows of the merged QKV output')
    if variant != 'fused_bf16_kv':
        raise ValueError('Unknown full-attention decode variant: ' + str(variant))
    from minisgl.attention.trtllm import TensorRTLLMBackend
    decode_backend = getattr(ctx.attn_backend, 'decode_backend', ctx.attn_backend)
    if not isinstance(decode_backend, TensorRTLLMBackend):
        # The strided query/key views are what the TRT-LLM decode kernel was validated with.
        raise RuntimeError('Fused Qwen3.5 decode kernels need the TRT-LLM decode backend '
                           f'(--attention-backend trtllm); got {type(decode_backend).__name__}')
    q_gate = qkv[:, :2 * layer.q_size].view(rows, heads, 2 * dim)
    query, gate = q_gate[..., :dim], q_gate[..., dim:]
    key = qkv[:, 2 * layer.q_size:2 * layer.q_size + layer.kv_size]
    value = qkv[:, 2 * layer.q_size + layer.kv_size:]
    return _fused_bf16_kv_decode(ctx, layer, qkv, query, key, value, gate)


def _fused_bf16_kv_decode(ctx, layer, qkv, query, key, value, gate) -> tuple:
    """full_attention_decode 'fused_bf16_kv': norm, RoPE and the BF16 KV store in one kernel."""
    import torch
    batch, attn = ctx.batch, layer.attn
    kvcache = ctx.attn_backend.kvcache
    if kvcache.dtype != torch.bfloat16 or attn.q_norm.eps != attn.k_norm.eps:
        raise RuntimeError('Fused Qwen3.5 decode stores BF16 KV: it needs the BF16 KV pool '
                           'and one q/k norm eps')
    k_cache = kvcache.k_cache(attn.layer_id).view(-1, layer.kv_size)
    v_cache = kvcache.v_cache(attn.layer_id).view(-1, layer.kv_size)
    _full_attn_norm_rope_bf16_store_module().launch(
        qkv, batch.positions, attn.q_norm.weight, attn.k_norm.weight, attn.rotary.cos_sin_cache,
        k_cache, v_cache, batch.out_loc, float(attn.q_norm.eps))
    previous = batch.afd_kv_store_merged
    batch.afd_kv_store_merged = True  # KV already stored by the fused kernel
    try:
        output = ctx.attn_backend.forward(query, key, value, attn.layer_id, batch)
    finally:
        batch.afd_kv_store_merged = previous
    return output.view(qkv.shape[0], layer.num_qo_heads, layer.head_dim), gate
