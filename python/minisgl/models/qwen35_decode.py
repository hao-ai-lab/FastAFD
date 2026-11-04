"""Model-owned selection of Qwen3.5 AFD decode implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelConfig


@dataclass(frozen=True)
class Qwen35DecodePolicy:
    """Internal capabilities selected together; not user-configurable switches."""

    shared_quant: bool = False  # FP8 input shared by QKV/z; merged BF16 b/a.
    fused_norm: bool = False  # Deferred shared-expert add, residual and input norm.
    attn_fused: bool = False  # Q/K norm, RoPE and BF16 KV store on merged-QKV views.

    @property
    def full_attention_decode(self) -> str | None:
        return "fused_bf16_kv" if self.attn_fused else None


def decode_policy(
    config: ModelConfig, tp_size: int = 1, *, for_afd: bool = False,
) -> Qwen35DecodePolicy:
    """Select the supported decode path from model, precision and topology: the AFD
    attention side of the block-FP8 checkpoint takes the fused dense decode under
    MINISGL_QWEN35_FP8_FUSED_DECODE=1; everything else runs the generic path."""
    if for_afd and os.environ.get(BLOCK_FP8_SWITCH) == "1":
        return _block_fp8_policy(config, tp_size)
    return Qwen35DecodePolicy()


# Qwen/Qwen3.5-397B-A17B-FP8 (block FP8, 128x128): with this switch set to 1 the AFD
# attention side runs the fused dense decode on it; unset (the default) changes nothing.
# The GDN recurrence kernel and the KV page handling stay generic.
BLOCK_FP8_SWITCH = "MINISGL_QWEN35_FP8_FUSED_DECODE"


def _block_fp8_policy(config: ModelConfig, tp_size: int) -> Qwen35DecodePolicy:
    """Shared QKV/z quant + merged BF16 b/a, fused input norm + FP8 quant, and fused
    full-attention q/k norm + RoPE + BF16 KV store, for the block-FP8 397B at TP1 only."""
    name = BLOCK_FP8_SWITCH
    model_type = getattr(config, "model_type", None)
    if model_type not in ("qwen3_5_moe", "qwen3_5_moe_text"):
        raise ValueError(f"{name}=1 is for the Qwen3.5-397B block-FP8 checkpoint; found model_type={model_type!r}")
    quant = getattr(config, "quant", None)
    if getattr(quant, "method", None) != "fp8" or tuple(quant.weight_block_size) != (128, 128):
        raise ValueError(f"{name}=1 requires the block-FP8 (128x128) checkpoint; found "
                         f"{getattr(quant, 'method', None)!r}")
    if tp_size != 1:
        raise ValueError(f"{name}=1 requires attention TP1; got {tp_size}")
    if os.environ.get("FLASHINFER_USE_CUDA_NORM", "1") != "1":
        raise ValueError(f"{name}=1 requires FLASHINFER_USE_CUDA_NORM=1")
    if os.environ.get("MINISGL_AFD_SPEC", "").strip().lower() not in ("", "0", "none", "off"):
        raise ValueError(f"{name}=1 is qualified without speculative decoding: unset MINISGL_AFD_SPEC")
    if config.hidden_size != 4096:
        raise ValueError(f"{name}=1 requires hidden_size=4096; found {config.hidden_size}")
    gdn_heads = (config.linear_num_key_heads, config.linear_num_value_heads,
                 config.linear_key_head_dim, config.linear_value_head_dim)
    if gdn_heads != (16, 64, 128, 128):
        raise ValueError(f"{name}=1 requires GDN heads (16, 64, 128, 128); found {gdn_heads}")
    full_heads = (config.num_qo_heads, config.num_kv_heads, config.head_dim)
    if full_heads != (32, 2, 256):
        raise ValueError(f"{name}=1 requires full-attention heads (32, 2, 256); found {full_heads}")
    if not {"linear_attention", "full_attention"} <= set(config.layer_types):
        raise ValueError(f"{name}=1 requires linear and full attention layers")
    for layer_id, kind in enumerate(config.layer_types):
        prefix = f"model.layers.{layer_id}"
        if kind == "linear_attention":
            fp8 = [f"{prefix}.linear_attn.{p}" for p in ("in_proj_qkv", "in_proj_z", "out_proj")]
            bf16 = [f"{prefix}.linear_attn.{p}" for p in ("in_proj_b", "in_proj_a")]
        elif kind == "full_attention":
            fp8 = [f"{prefix}.self_attn.{p}_proj" for p in ("q", "k", "v", "o")]
            bf16 = []
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {kind!r}")
        for module in fp8:
            if config.quant_for_modules(module) is not quant:
                raise ValueError(f"{name}=1 requires block-FP8 {module}")
        for module in bf16:
            if config.quant_for_modules(module) is not None:
                raise ValueError(f"{name}=1 requires unquantized BF16 {module}")
    return Qwen35DecodePolicy(shared_quant=True, fused_norm=True, attn_fused=True)


# MINISGL_QWEN35_VLLM_KERNELS=1 runs vLLM db9527a4's GDN decode kernels (kernel/vllm_*.py). It is written for
# exactly the block-FP8 path above: one q|k|v|z FP8 GEMM whose output columns the in-place conv overwrites
# (MINISGL_QWEN35_MERGED_QKVZ=1), merged BF16 b|a, request-table state slots.
VLLM_KERNELS_SWITCH = "MINISGL_QWEN35_VLLM_KERNELS"


def check_vllm_qwen35_layer(config: ModelConfig, policy: Qwen35DecodePolicy, tp_size: int) -> None:
    name = VLLM_KERNELS_SWITCH
    if not (policy.shared_quant and policy.fused_norm):
        raise ValueError(f"{name}=1 requires the block-FP8 decode policy ({BLOCK_FP8_SWITCH}=1); got {policy}")
    if os.environ.get("MINISGL_QWEN35_MERGED_QKVZ") != "1":
        raise ValueError(f"{name}=1 requires MINISGL_QWEN35_MERGED_QKVZ=1 (vLLM's single in_proj_qkvz GEMM layout)")
    if tp_size != 1:
        raise ValueError(f"{name}=1 requires attention TP1; got {tp_size}")
    gdn_heads = (config.linear_num_key_heads, config.linear_num_value_heads,
                 config.linear_key_head_dim, config.linear_value_head_dim)
    if gdn_heads != (16, 64, 128, 128):
        raise ValueError(f"{name}=1 requires GDN heads (16, 64, 128, 128); found {gdn_heads}")
    if not 2 <= config.linear_conv_kernel_dim <= 6:
        raise ValueError(f"{name}=1: vLLM's conv update kernel supports width 2..6; found {config.linear_conv_kernel_dim}")
    if config.hidden_act not in ("silu", "swish"):
        raise ValueError(f"{name}=1: the conv activation must be silu; found {config.hidden_act!r}")
