"""Kernels of vLLM db9527a46873 (the vLLM baseline's version) for the Qwen3.5-397B
attention side, behind ONE switch: MINISGL_QWEN35_VLLM_KERNELS=1 (unset / anything else: byte-identical to the default path).

What the switch changes (decode only; prefill, speculative verify and every other model keep their path):
  * GDN layer core = vLLM ``QwenGatedDeltaNetAttention._forward_core_fused_norm_packed`` on the decode path
    (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1766-1892): in-place
    ``_causal_conv1d_update_kernel`` on the packed q|k|v columns, ``b.contiguous()`` / ``a.contiguous()``,
    ``fused_recurrent_gated_delta_rule_packed_decode_kernel`` reading q/k/v by row stride, then
    ``layer_norm_fwd_kernel`` (gated RMSNorm) in place. Replaces qwen35_causal_conv1d_decode, the five
    q/k/v/a/b ``.contiguous()`` copies + _qwen35_gdn_decode_kernel, and _qwen35_gated_rmsnorm_kernel.
  * shared-expert gate (sub-switch MINISGL_QWEN35_VLLM_SHARED_GATE, default = the main switch), decode: vLLM's
    qwen2_moe.py:120-121 line for line and eager (shared_gate.py: F.linear N=1, ATen sigmoid, ATen mul; vLLM runs it
    inside the opaque moe_forward_shared op) instead of _qwen35_shared_gate_kernel / the compiled sigmoid-mul.
  * FP8 activation quant (sub-switch MINISGL_QWEN35_VLLM_QUANT): vLLM's per_token_group_quant_8bit_packed_register_kernel
    (per_token_group_quant.py, CUDA source vendored in csrc/jit/vllm_per_token_group_quant.cu) instead of
    per_token_cast_fp8_packed_kernel at layers/linear.py _forward_fp8 and kernel/qwen35_decode.py (prefill too).
  * router top-k (sub-switch MINISGL_QWEN35_VLLM_ROUTER): vLLM's topkGating<512> (topk_softmax.py, CUDA source vendored in
    csrc/jit/vllm_topk_softmax.cu) instead of _qwen35_softmax_topk_kernel in kernel/moe_topk.py topk_softmax (512 experts).
  * full-attention decode (sub-switch MINISGL_QWEN35_VLLM_FULLATTN; AFD side, BF16 KV ('fused_bf16_kv') only): vLLM's
    _fused_qk_rmsnorm_rope_gate_kernel (fused_qk_norm_rope.py, vendored Triton) + reshape_and_cache_flash_kernel (CUDA
    source, csrc/jit/vllm_reshape_and_cache_flash.cu) + the same FMHA + vLLM's inductor o-gate kernel (ogate.py, captured
    from vLLM's run) instead of qwen35_full_attn_norm_rope_bf16_store_kernel + the compiled sigmoid-mul (full_attn_decode.py).
  * RMSNorms (sub-switch MINISGL_QWEN35_VLLM_NORM; AFD side, decode, block-FP8 fused-norm policy): vLLM's inductor norm kernels
    (layer-0 input, post-attention, input + QuantFP8, final; captured, inductor/*) with vLLM's residual stream (norms.py)
    instead of FlashInfer FusedAddRMSNorm + moe_residual_rmsnorm_fp8_quant_kernel; shared-expert SiLU-mul (sub-switch
    MINISGL_QWEN35_VLLM_SILU): vLLM's inductor triton_poi_fused_mul_silu_slice_0 instead of act_and_mul_kernel.

Vendored files (byte for byte from the vLLM git checkout at db9527a46873; sha256_12 of the upstream file):
  fused_recurrent_packed.py  <- vllm/third_party/flash_linear_attention/ops/fused_recurrent.py   fe6f13110148  (1 edit)
  causal_conv1d_update.py    <- vllm/model_executor/layers/mamba/ops/causal_conv1d.py            cb16cc9250c4
  layernorm_fwd.py           <- vllm/third_party/flash_linear_attention/ops/layernorm_guard.py   b4f696187a13
"""

from __future__ import annotations

import os

SWITCH = "MINISGL_QWEN35_VLLM_KERNELS"

VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"

# (vendored module, upstream path, upstream sha256 (full), [(upstream first line, last line), ...])
VENDORED = (
    ("fused_recurrent_packed.py", "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
     "fe6f13110148", ((255, 485),)),
    ("causal_conv1d_update.py", "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
     "cb16cc9250c4", ((762, 1278),)),
    ("layernorm_fwd.py", "vllm/third_party/flash_linear_attention/ops/layernorm_guard.py",
     "b4f696187a13", ((68, 183), (366, 411), (413, 413), (441, 493))),
    ("fused_qk_norm_rope.py", "vllm/model_executor/layers/fused_qk_norm_rope.py",
     "c201d88fd5e8", ((16, 279),)),
)
# The one deliberate edit (upstream line -> vendored line).
EDITS = {
    "fused_recurrent_packed.py": (("    if state_idx <= 0:", "    if state_idx < 0:"),),
}


def enabled() -> bool:
    return os.environ.get(SWITCH) == "1"


# Sub-switch: the shared-expert gate follows MINISGL_QWEN35_VLLM_KERNELS unless set explicitly.
# "1": vLLM's gate, line-exact and eager in decode (shared_gate.py; replaces the compiled sigmoid-multiply);
# "0": keep _qwen35_shared_gate_kernel (MINISGL_QWEN35_FUSED_SHARED_GATE) while the GDN kernels are vLLM's.
SHARED_GATE_SWITCH = "MINISGL_QWEN35_VLLM_SHARED_GATE"


def shared_gate_enabled() -> bool:
    return _sub_switch(SHARED_GATE_SWITCH)


# sub-switches (same rule as the shared gate: unset / empty = follow MINISGL_QWEN35_VLLM_KERNELS,
# "1" on, anything else off), so each vLLM op can be isolated:
QUANT_SWITCH = "MINISGL_QWEN35_VLLM_QUANT"     # FP8 activation quant (per_token_group_quant.py)
ROUTER_SWITCH = "MINISGL_QWEN35_VLLM_ROUTER"   # router top-k (topk_softmax.py)
FULLATTN_SWITCH = "MINISGL_QWEN35_VLLM_FULLATTN"   # full-attention decode: qk norm + RoPE + gate, KV store, o-gate
NORM_SWITCH = "MINISGL_QWEN35_VLLM_NORM"           # input / post-attention / first / final RMSNorms (inductor kernels, norms.py)
SILU_SWITCH = "MINISGL_QWEN35_VLLM_SILU"           # shared-expert SiLU-mul (inductor kernel, norms.py)
SUB_SWITCHES = (SHARED_GATE_SWITCH, QUANT_SWITCH, ROUTER_SWITCH, FULLATTN_SWITCH, NORM_SWITCH, SILU_SWITCH)

# vendored CUDA sources (csrc/jit/<file>): upstream path, upstream sha256_12, byte-for-byte upstream line ranges
VENDORED_CU = (
    ("vllm_topk_softmax.cu", "csrc/libtorch_stable/moe/topk_softmax_kernels.cu", "5e0eae8dbd76",
     ((41, 42), (44, 74), (278, 625))),
    ("vllm_per_token_group_quant.cu", "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu",
     "380e1f066823", ((530, 695),)),
    ("vllm_reshape_and_cache_flash.cu", "csrc/libtorch_stable/quantization/vectorization.cuh", "6b28cf0cfe9e", ((12, 16),)),
    ("vllm_reshape_and_cache_flash.cu", "csrc/libtorch_stable/quantization/vectorization_utils.cuh", "991d501211c9",
     ((6, 115),)),
    ("vllm_reshape_and_cache_flash.cu", "csrc/attention/dtype_fp8.cuh", "dde54da39de7", ((15, 19),)),
    ("vllm_reshape_and_cache_flash.cu", "csrc/libtorch_stable/cache_kernels.cu", "5ea4d65b32a1", ((251, 263), (325, 411))),
)


def _sub_switch(name: str) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return enabled()
    return value == "1"


def quant_enabled() -> bool:
    return _sub_switch(QUANT_SWITCH)


def router_enabled() -> bool:
    return _sub_switch(ROUTER_SWITCH)


def fullattn_enabled() -> bool:
    return _sub_switch(FULLATTN_SWITCH)


def norm_enabled() -> bool:
    return _sub_switch(NORM_SWITCH)


def silu_enabled() -> bool:
    return _sub_switch(SILU_SWITCH)
