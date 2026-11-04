"""vLLM db9527a4's Qwen3.5-397B RMSNorms and shared-expert SiLU on the AFD attention side (sub-switches
MINISGL_QWEN35_VLLM_NORM and MINISGL_QWEN35_VLLM_SILU): the inductor kernels CAPTURED from vLLM's own run (kernel/vllm_inductor_*.py,
compiled and launched as inductor does: inductor_launch.py).

vLLM's residual stream (the captured graphs' FX comments; GemmaRMSNorm = raw weight + 1.0 in fp32, eps 1e-6, hidden 4096):
  layer 0 input   first:          normed = rms(x) * (1 + w)                      residual = x (the embedding)
  post-attention  post_gdn/full:  normed = rms(attn + residual) * (1 + w), 3 copies (router / shared expert / experts);
                                  the sum attn + residual is NOT stored
  next input      in_*:           residual' = bf16((routed + shared) + (attn + residual)) (fp32 sums, one rounding),
                                  normed = rms(...) * (1 + w); then QuantFP8 for the FP8 in_proj / qkv_proj
  final norm      final:          normed = rms((routed + shared) + (attn + residual)) * (1 + w), written in place
So the post-attention step hands the NEXT input norm (attn_out, residual), not their sum: the AFD stage keeps a per-layer FIFO
of attention outputs next to its pending shared-expert outputs (models/qwen35_moe_afd.py).
Which in_* kernel: vLLM compiles one graph per attention-op boundary, so the same math has three kernel instances:
  in_gdn (GDN after GDN), in_full (full-attention layer; name ..._per_token_group_fp8_quant_packed_1, the quant is a separate
  CUDA kernel), in_after_full (GDN after a full layer; config INFERRED, see vllm_inductor_in_after_full.py).
"""

from __future__ import annotations

import torch

from . import vllm_inductor as inductor_launch
from . import vllm_inductor_final as final, vllm_inductor_first as first, vllm_inductor_in_after_full as in_after_full
from . import vllm_inductor_in_full as in_full, vllm_inductor_in_gdn as in_gdn, vllm_inductor_post_full as post_full
from . import vllm_inductor_post_gdn as post_gdn, vllm_inductor_silu as silu

HIDDEN = 4096
_IN = {"in_gdn": in_gdn, "in_full": in_full, "in_after_full": in_after_full}


def _rows(*ts):
    n = ts[0].shape[0]
    for t in ts:
        if t.dtype != torch.bfloat16 or t.shape != (n, HIDDEN) or not t.is_contiguous():
            raise ValueError(f"vLLM norm kernels take contiguous bf16 [n, {HIDDEN}] rows, got {tuple(t.shape)} {t.dtype} "
                             f"{t.stride()}")
    return n


def _w(w):
    if w is None or w.dtype != torch.bfloat16 or w.shape != (HIDDEN,) or not w.is_contiguous():
        raise ValueError("vLLM norm kernels take the RAW (un-offset) bf16 [4096] weight (RMSNormOffset*.raw_weight)")
    return w


def input_kind(prev_type: str, this_type: str) -> str:
    if this_type == "full_attention":
        return "in_full"
    return "in_after_full" if prev_type == "full_attention" else "in_gdn"


def first_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = _rows(x)
    out = torch.empty_like(x)
    inductor_launch.launch(first, (n, 1, 1), x, _w(w), out, n, HIDDEN)
    return out


def post_norm(attn_out: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, full: bool):
    """-> (normed x 3): vLLM's moe_forward_shared inputs (router, shared expert, experts)."""
    n = _rows(attn_out, residual)
    o1, o2, o3 = torch.empty_like(attn_out), torch.empty_like(attn_out), torch.empty_like(attn_out)
    inductor_launch.launch(post_full if full else post_gdn, (n, 1, 1), attn_out, residual, _w(w), o1, o2, o3, n, HIDDEN)
    return o1, o2, o3


def input_norm(kind: str, routed, shared, attn_prev, residual, w):
    """-> (residual', normed)."""
    n = _rows(routed, shared, attn_prev, residual)
    res_out, normed = torch.empty_like(residual), torch.empty_like(residual)
    inductor_launch.launch(_IN[kind], (n, 1, 1), routed, shared, attn_prev, residual, _w(w), res_out, normed, n, HIDDEN)
    return res_out, normed


def final_norm(routed, shared, attn_last, residual, w) -> torch.Tensor:
    """vLLM's final norm writes into its MoE-output buffer (in_out_ptr0); FastAFD's routed buffer belongs to the AFD runtime,
    so a copy is normalized in place (one copy kernel per step: a named difference)."""
    n = _rows(routed, shared, attn_last, residual)
    out = routed.clone()
    inductor_launch.launch(final, (n, 1, 1), out, shared, attn_last, residual, _w(w), n, HIDDEN)
    return out


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate_up[:, :I]) * gate_up[:, I:] -> [n, I] (vLLM SiluAndMul, inductor kernel triton_poi_fused_mul_silu_slice_0)."""
    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2 or gate_up.stride(1) != 1 or gate_up.shape[1] % 2:
        raise ValueError(f"vLLM SiLU-mul takes bf16 [n, 2I] rows, got {tuple(gate_up.shape)} {gate_up.dtype} {gate_up.stride()}")
    n, two_i = gate_up.shape
    half = two_i // 2
    out = torch.empty((n, half), dtype=gate_up.dtype, device=gate_up.device)
    xnumel = n * half
    if xnumel:
        xb = silu.CONFIG["XBLOCK"]
        inductor_launch.launch(silu, ((xnumel + xb - 1) // xb, 1, 1), gate_up, out, half, gate_up.stride(0), xnumel)
    return out


__all__ = ["input_kind", "first_norm", "post_norm", "input_norm", "final_norm", "silu_mul"]
