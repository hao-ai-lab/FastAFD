"""vLLM db9527a4's Qwen3.5-397B full-attention decode sequence on the AFD attention side (sub-switch
MINISGL_QWEN35_VLLM_FULLATTN).

vLLM (Qwen3NextAttention.forward, qwen3_next.py:450-460 + _project_qkv_gate :399-421; FlashInfer backend):
  q, k, gate = fused_qk_rmsnorm_rope_gate(q_gate, k, q_norm.weight, k_norm.weight, cos_sin_cache, positions, eps, ...,
                                          norm_beta=1.0, mrope_section)   -> _fused_qk_rmsnorm_rope_gate_kernel
  reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, "auto", ...)      -> reshape_and_cache_flash_kernel
  FMHA (TRT-LLM gen decode, the same kernel FastAFD runs)
  attn_output * sigmoid(gate)                                                           -> inductor o-gate kernel (ogate.py)
  o_proj (QuantFP8 -> DeepGEMM)
Replaces qwen35_full_attn_norm_rope_bf16_store_kernel (norm + RoPE + KV store in place in qkv, strided q / gate)
and triton_poi_fused_mul_sigmoid_0 (FastAFD's compiled gate).

Inputs as vLLM hands them over (each a contract of vLLM's kernels, checked here):
  * RAW q/k norm weights + norm_beta 1.0 (vLLM adds 1 in fp32 in the kernel; FastAFD's RMSNormOffset folds 1 into a bf16
    weight at load, so layers/norm.py keeps the raw checkpoint tensor as ``raw_weight``);
  * cos_sin_cache in bf16 (vLLM's rotary cache is the model dtype: the vLLM Triton entry reads a bf16 pointer); FastAFD's
    fp32 cache (same formula, layers/rotary.py) is cast once;
  * positions [3, n] for the M-RoPE rotary (config mrope_section [11, 11, 10]: vLLM keeps 2-D positions and takes the
    kernel's HAS_MROPE specialization; text tokens have t = h = w, so FastAFD's 1-D positions are broadcast to 3 rows,
    stride 0). vLLM's positions are int64, FastAFD's AG decode graph buffer is int32 (afd_attention_worker.py:697): the
    kernel loads either and widens to int64 (``.to(tl.int64)``), so the int32 buffer is passed as is (one Triton
    specialization of the same source differs in the positions pointer type; recorded);
  * slot ids int64 (vLLM's slot_mapping; reshape_and_cache_flash_kernel reads ``const int64_t*``). FastAFD's graph buffer
    out_loc is int32 (afd_attention_worker.py:698), so it is cast per full-attention layer (one small cast kernel x 15
    layers per step; a named difference until the call side keeps an int64 slot buffer).
"""

from __future__ import annotations

import torch

from .vllm_fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from .vllm_reshape_and_cache import reshape_and_cache_flash


def _bf16_cos_sin(rotary) -> torch.Tensor:
    t = getattr(rotary, "_vllm_qwen35_cos_sin_bf16", None)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("vLLM full-attention decode: bf16 cos/sin cache first requested inside CUDA-graph capture")
        t = rotary.cos_sin_cache.to(torch.bfloat16).contiguous()
        rotary._vllm_qwen35_cos_sin_bf16 = t
    return t


def _step_ids(batch):
    """(positions [3, n] (stride-0 rows of FastAFD's 1-D buffer, int32 or int64), slot ids int64 [n])."""
    pos, loc = batch.positions, batch.out_loc
    if pos.dtype not in (torch.int32, torch.int64) or loc.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"vLLM full-attention decode: positions {pos.dtype} / out_loc {loc.dtype}")
    return pos.unsqueeze(0).expand(3, -1), (loc if loc.dtype == torch.int64 else loc.to(torch.int64))


def mrope_section_of(config) -> list[int]:
    params = getattr(config.rotary_config, "parameters", None) or {}
    sec = params.get("mrope_section") if isinstance(params, dict) else None
    if not sec or len(sec) != 3:
        raise ValueError("MINISGL_QWEN35_VLLM_FULLATTN needs the checkpoint's rope_parameters.mrope_section (vLLM's M-RoPE path)")
    return [int(x) for x in sec]


def full_attention_decode_vllm(layer, qkv: torch.Tensor, mrope_section) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (attention output [n, H, D] contiguous, raw gate [n, H*D] contiguous) for the o-gate kernel."""
    from minisgl.core import get_global_ctx
    from minisgl.attention.trtllm import TensorRTLLMBackend

    ctx = get_global_ctx()
    batch, attn = ctx.batch, layer.attn
    n, heads, kv_heads, dim = qkv.shape[0], layer.num_qo_heads, layer.num_kv_heads, layer.head_dim
    if not batch.is_decode or qkv.ndim != 2 or qkv.stride(1) != 1 or qkv.shape[1] != 2 * layer.q_size + 2 * layer.kv_size:
        raise RuntimeError("vLLM full-attention decode needs decode rows of the merged QKV output")
    decode_backend = getattr(ctx.attn_backend, "decode_backend", ctx.attn_backend)
    if not isinstance(decode_backend, TensorRTLLMBackend):
        raise RuntimeError("vLLM full-attention decode runs the TRT-LLM decode FMHA (--attention-backend trtllm)")
    kvcache = ctx.attn_backend.kvcache
    if kvcache.dtype != torch.bfloat16 or qkv.dtype != torch.bfloat16:
        raise RuntimeError("vLLM full-attention decode stores a bf16 KV cache (vLLM --kv-cache-dtype bfloat16)")
    q_gate = qkv[:, :2 * layer.q_size]
    k = qkv[:, 2 * layer.q_size:2 * layer.q_size + layer.kv_size]
    v = qkv[:, 2 * layer.q_size + layer.kv_size:]
    pos3, slots = _step_ids(batch)
    q_w = getattr(attn.q_norm, "raw_weight", None)
    k_w = getattr(attn.k_norm, "raw_weight", None)
    if q_w is None or k_w is None:
        raise RuntimeError("vLLM full-attention decode needs the raw (un-offset) q/k norm weights (layers/norm.py raw_weight)")
    q, k_out, gate = fused_qk_rmsnorm_rope_gate(
        q_gate, k, q_w, k_w, _bf16_cos_sin(attn.rotary), pos3, float(attn.q_norm.eps), heads, kv_heads, dim,
        int(attn.rotary.rotary_dim), mrope_section=mrope_section, norm_beta=1.0)
    key_cache = kvcache.k_cache(attn.layer_id)
    value_cache = kvcache.v_cache(attn.layer_id)
    reshape_and_cache_flash(k_out.view(n, kv_heads, dim), v.view(n, kv_heads, dim), key_cache, value_cache, slots)
    previous = batch.afd_kv_store_merged
    batch.afd_kv_store_merged = True   # KV stored above
    try:
        output = ctx.attn_backend.forward(q.view(n, heads, dim), k_out.view(n, kv_heads, dim), v.view(n, kv_heads, dim),
                                          attn.layer_id, batch)
    finally:
        batch.afd_kv_store_merged = previous
    return output.reshape(n, heads, dim), gate


__all__ = ["full_attention_decode_vllm", "mrope_section_of"]
