"""vLLM's Qwen3.5 GDN decode sequence on FastAFD tensors (MINISGL_QWEN35_VLLM_KERNELS=1).

Mirrors, call for call, vLLM db9527a46873 ``QwenGatedDeltaNetAttention.forward_cuda`` with the fused-norm packed
op on a pure-decode, non-speculative batch (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py):

  :917-921   core_attn_out = torch.zeros((n, HV, V), bf16)
  :1783-1789 mixed_qkv, output_gate_flat = mixed_qkvz.split([qkv, z]); output_gate = reshape(n, -1, V)   (views)
  :1789      b, a = split_ba(ba)  (ba.chunk(2, -1), :632-640)                                          (views)
  :1880-1884 _forward_core(mixed_qkv, b.contiguous(), a.contiguous(), core_attn_out)                    (2 copies)
  :1272-1287 -> _forward_core_decode_non_spec (pure decode, no spec masks, packed decode enabled by default,
                envs.py:132 VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=True)
  :1653-1668 causal_conv1d_update(mixed_qkv, conv_state, conv_weights, bias, activation,
                                  conv_state_indices, validate_data=False)  -> in place into mixed_qkv
  :1669-1681 fused_recurrent_gated_delta_rule_packed_decode(..., scale=K**-0.5, out=core_attn_out[:n].unsqueeze(1),
                                                           use_qk_l2norm_in_kernel=True)
  :1885-1892 _rms_norm_gated_cuda(core_attn_out, output_gate, core_attn_out)  (:1815-1847: reshape(-1, V) of the
                gate is a copy for n > 1, layer_norm_fwd(..., is_rms_norm=True, norm_before_gate=True,
                activation='silu', group_size=V), in place)
and returns core_attn_out for the caller's out_proj (:928 out_proj(core_attn_out.flatten(-2))).

FastAFD differences kept on purpose (each one is a contract of FastAFD's state pool, not a kernel choice):
  * conv_state_indices / ssm_state_indices are FastAFD request-table slots (slot 0 real; int32 as vLLM's, set on the call side by
    state_index.py under the switch). The conv call
    passes null_block_id=None (vLLM API, causal_conv1d.py:1272: HAS_NULL_BLOCK=False); the vendored recurrent
    kernel skips only negative slots (the one edit). vLLM passes int32 block ids (gdn_attn.py:145-147, :301).
  * conv_state is FastAFD's [slot, channel, tap] view of a [slot, tap, channel] store (kvcache/hybrid_pool.py:40-52),
    the same strides as vLLM's SD layout after ``.transpose(-1, -2)`` (qwen_gdn_linear_attn.py:1645-1649).
  * FastAFD's own kernels zero the state of a row at position 0; vLLM's never do. A decode row is never at
    position 0 (the prompt sets the state, models/qwen35_moe.py:248-262); only padding rows of the dummy slot are.
"""

from __future__ import annotations

import torch

from .vllm_causal_conv1d import causal_conv1d_update
from .vllm_fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
from .vllm_layernorm_gated import layer_norm_fwd


def check_inputs(mixed_qkv, z, b, a, conv_weight, A_log, dt_bias, norm_weight, conv_state, recurrent_state,
                 state_indices, *, num_key_heads: int, num_value_heads: int, key_dim: int, value_dim: int) -> None:
    """Metadata-only checks (graph-capturable); raise on any layout the vLLM sequence was not written for."""
    n = mixed_qkv.shape[0]
    qkv = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != qkv or mixed_qkv.stride(1) != 1:
        raise ValueError(f"vLLM GDN decode: mixed_qkv must be [n, {qkv}] with unit column stride, got "
                         f"{tuple(mixed_qkv.shape)}/{mixed_qkv.stride()}")
    z_ok = (z.ndim == 2 and tuple(z.shape) == (n, num_value_heads * value_dim)) or (
        z.ndim == 3 and tuple(z.shape) == (n, num_value_heads, value_dim))
    if not z_ok or z.stride(-1) != 1:
        raise ValueError(f"vLLM GDN decode: z must hold [n, {num_value_heads}, {value_dim}], got {tuple(z.shape)}")
    for name, t in (("b", b), ("a", a)):
        if t.ndim != 2 or t.shape != (n, num_value_heads) or t.stride(1) != 1:
            raise ValueError(f"vLLM GDN decode: {name} must be [n, {num_value_heads}] with unit column stride")
    if conv_weight.ndim != 3 or conv_weight.shape[0] != qkv or conv_weight.shape[1] != 1:
        raise ValueError(f"vLLM GDN decode: conv weight must be [{qkv}, 1, W], got {tuple(conv_weight.shape)}")
    if conv_state.ndim != 3 or conv_state.shape[1] != qkv or conv_state.shape[2] != conv_weight.shape[2] - 1:
        raise ValueError(f"vLLM GDN decode: conv state must be [slots, {qkv}, W-1], got {tuple(conv_state.shape)}")
    if (recurrent_state.ndim != 4 or recurrent_state.shape[1:] != (num_value_heads, value_dim, key_dim)
            or recurrent_state.stride(-1) != 1):
        raise ValueError(f"vLLM GDN decode: recurrent state must be [slots, {num_value_heads}, {value_dim}, {key_dim}]")
    if state_indices.ndim != 1 or state_indices.shape[0] != n:
        raise ValueError("vLLM GDN decode: state_indices must be [n]")
    if norm_weight.shape != (value_dim,):
        raise ValueError("vLLM GDN decode: gated-norm weight must be [V]")
    if A_log.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads,):
        raise ValueError("vLLM GDN decode: A_log / dt_bias must be [HV]")


def gdn_decode_vllm(
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    conv_activation: str = "silu",
    norm_activation: str = "silu",
) -> torch.Tensor:
    """Returns core_attn_out [n, HV, V] (BF16, gated-normed). ``mixed_qkv`` is overwritten by the conv output,
    as in vLLM. ``z`` may be the [n, V*HV] column view of the merged q|k|v|z GEMM output (row stride 20480)."""
    check_inputs(mixed_qkv, z, b, a, conv_weight, A_log, dt_bias, norm_weight, conv_state, recurrent_state,
                 state_indices, num_key_heads=num_key_heads, num_value_heads=num_value_heads,
                 key_dim=key_dim, value_dim=value_dim)
    num_tokens = mixed_qkv.shape[0]
    # qwen_gdn_linear_attn.py:917-921 (forward_cuda, use_fused_gdn_decode branch)
    core_attn_out = torch.zeros(
        (num_tokens, num_value_heads, value_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    # :1786-1788 output_gate = output_gate_flat.reshape(output_gate_flat.size(0), -1, self.head_v_dim)
    output_gate = z.reshape(z.size(0), -1, value_dim)
    # :1880-1884 b.contiguous(), a.contiguous()
    b = b.contiguous()
    a = a.contiguous()
    # :1657-1668 _forward_core_decode_non_spec, conv (bias None: ColumnParallelLinear(bias=False), :421-426)
    conv_weights = conv_weight.view(conv_weight.size(0), conv_weight.size(2))
    mixed_qkv_non_spec = causal_conv1d_update(
        mixed_qkv,
        conv_state,
        conv_weights,
        None,
        conv_activation,
        conv_state_indices=state_indices[:num_tokens],
        null_block_id=None,
        validate_data=False,
    )
    # :1669-1681
    out_buf = core_attn_out[:num_tokens].unsqueeze(1)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv_non_spec,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=recurrent_state,
        out=out_buf,
        ssm_state_indices=state_indices[:num_tokens],
        use_qk_l2norm_in_kernel=True,
    )
    # :1885-1892 -> _rms_norm_gated_cuda :1815-1847
    x = core_attn_out[:num_tokens]
    x_shape = x.shape
    gate = output_gate[:num_tokens]
    x_2d = x.reshape(-1, x_shape[-1])
    output_gate_2d = gate.reshape(-1, x_shape[-1])
    out_2d = x.reshape(-1, x_shape[-1])
    layer_norm_fwd(
        x_2d,
        norm_weight.contiguous(),
        None,
        norm_eps,
        z=output_gate_2d,
        out=out_2d,
        group_size=x_shape[-1],
        norm_before_gate=True,
        is_rms_norm=True,
        activation=norm_activation,
    )
    return core_attn_out


__all__ = ["check_inputs", "gdn_decode_vllm"]
