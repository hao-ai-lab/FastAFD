from __future__ import annotations

import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def qwen35_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.5 Q/K L2 normalization, evaluated in FP32."""

    x_f32 = x.float()
    return (x_f32 * torch.rsqrt(x_f32.square().sum(dim=-1, keepdim=True) + eps)).to(
        x.dtype
    )


def qwen35_gdn_gates(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return log forget gate ``g`` and delta update gate ``beta`` in FP32."""

    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    return g, beta


def _expand_qk_heads(
    q: torch.Tensor, k: torch.Tensor, num_value_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_key_heads = q.shape[-2]
    if num_key_heads == num_value_heads:
        return q, k
    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            "GDN value heads must be divisible by key heads: "
            f"{num_value_heads} % {num_key_heads} != 0"
        )
    repeats = num_value_heads // num_key_heads
    return (
        q.repeat_interleave(repeats, dim=-2),
        k.repeat_interleave(repeats, dim=-2),
    )


def qwen35_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float | None = None,
    normalize_qk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Portable GDN oracle/fallback for one sequence.

    Inputs use ``[T, H, D]`` and state uses the FlashInfer-compatible
    k-last layout ``[H_v, D_v, D_k]``.
    """

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [tokens, heads, head_dim]")
    if initial_state.shape != (
        v.shape[1],
        v.shape[2],
        k.shape[2],
    ):
        raise ValueError(
            "invalid recurrent state shape: "
            f"got {tuple(initial_state.shape)}, expected "
            f"{(v.shape[1], v.shape[2], k.shape[2])}"
        )

    if normalize_qk:
        q, k = qwen35_l2norm(q), qwen35_l2norm(k)
    q, k = _expand_qk_heads(q, k, v.shape[1])
    scale = float(scale if scale is not None else q.shape[-1] ** -0.5)
    state = initial_state.float().clone()
    outputs = []
    for token_idx in range(q.shape[0]):
        state.mul_(torch.exp(g[token_idx].float())[:, None, None])
        q_t = q[token_idx].float() * scale
        k_t = k[token_idx].float()
        v_t = v[token_idx].float()
        memory = (state * k_t[:, None, :]).sum(dim=-1)
        delta = (v_t - memory) * beta[token_idx].float()[:, None]
        state.add_(delta[:, :, None] * k_t[:, None, :])
        outputs.append((state * q_t[:, None, :]).sum(dim=-1))
    output = torch.stack(outputs, dim=0).to(q.dtype)
    return output, state


def qwen35_recurrent_gated_delta_rule_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Portable packed-sequence wrapper used by CPU tests and fallback paths."""

    boundaries = [int(x) for x in cu_seqlens.to("cpu").tolist()]
    if len(boundaries) != initial_state.shape[0] + 1:
        raise ValueError("cu_seqlens and initial_state batch size disagree")
    outputs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        output, final_state = qwen35_recurrent_gated_delta_rule(
            q[start:end],
            k[start:end],
            v[start:end],
            g[start:end],
            beta[start:end],
            initial_state[seq_idx],
        )
        outputs.append(output)
        final_states.append(final_state)
    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


def qwen35_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN prefill using FlashInfer's public SM90 API when available."""

    if not q.is_cuda:
        return qwen35_recurrent_gated_delta_rule_packed(
            q, k, v, g, beta, initial_state, cu_seqlens
        )

    device_major = torch.cuda.get_device_capability(q.device)[0]
    if device_major == 9 and q.dtype in (torch.float16, torch.bfloat16):
        from flashinfer import chunk_gated_delta_rule

        q_norm = qwen35_l2norm(q).contiguous()
        k_norm = qwen35_l2norm(k).contiguous()
        return chunk_gated_delta_rule(
            q=q_norm,
            k=k_norm,
            v=v.contiguous(),
            g=torch.exp(g.float()).contiguous(),
            beta=beta.float().contiguous(),
            scale=q.shape[-1] ** -0.5,
            initial_state=initial_state.float().contiguous(),
            output_final_state=True,
            cu_seqlens=cu_seqlens.to(device=q.device, dtype=torch.int64),
            use_qk_l2norm_in_kernel=False,
        )
    return _qwen35_gdn_prefill_triton(q, k, v, g, beta, initial_state, cu_seqlens)


def qwen35_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    first_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Packed depthwise causal convolution and final short-conv state."""

    width = weight.shape[-1]
    if initial_state.shape[-1] != width - 1:
        raise ValueError("convolution state width must equal kernel width - 1")
    boundaries = [int(v) for v in cu_seqlens.to("cpu").tolist()]
    outputs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        reset = (first_positions[seq_idx] == 0).to(initial_state.dtype)
        prefix = initial_state[seq_idx] * (1 - reset)
        sequence = x[start:end].transpose(0, 1)
        window = torch.cat((prefix, sequence), dim=-1)
        output = F.conv1d(
            window.unsqueeze(0),
            weight,
            bias=None,
            groups=x.shape[-1],
        ).squeeze(0).transpose(0, 1)
        outputs.append(F.silu(output))
        final_states.append(window[:, -(width - 1) :])
    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


@triton.jit
def _qwen35_causal_conv1d_decode_kernel(
    x_ptr,
    w_ptr,
    state_ptr,
    idx_ptr,
    pos_ptr,
    out_ptr,
    num_channels,
    x_row_stride,
    state_slot_stride,
    state_channel_stride,
    state_tap_stride,
    w_channel_stride,
    w_tap_stride,
    out_row_stride,
    KERNEL_WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One fused decode step of the Qwen3.5 depthwise causal convolution.

    Replaces index_select + mask + cat + weighted-sum + index_copy_ + SiLU, which
    ran as roughly seven ATen launches per linear-attention layer.

    The retained taps are held in scalar registers instead of a [BLOCK_C, taps]
    tile, mirroring vLLM's `_causal_conv1d_update_kernel`. That removes a second
    read of the state (the shifted window is already in registers) and the
    power-of-two tap padding, which together cost ~1.6x the reference's runtime.
    """
    req = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < num_channels

    slot = tl.load(idx_ptr + req)
    # A request at position 0 starts a fresh sequence, so its retained state is
    # discarded rather than convolved.
    keep = tl.where(tl.load(pos_ptr + req) != 0, 1.0, 0.0)

    state_base = state_ptr + slot * state_slot_stride + offs_c * state_channel_stride
    w_base = w_ptr + offs_c * w_channel_stride
    xv = tl.load(x_ptr + req * x_row_stride + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    # window = [state[0..W-2], x]; accumulate against the matching weight taps.
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    if KERNEL_WIDTH >= 2:
        tap0 = tl.load(state_base, mask=mask_c, other=0.0).to(tl.float32) * keep
        acc += tap0 * tl.load(w_base, mask=mask_c, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        tap1 = tl.load(
            state_base + state_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32) * keep
        acc += tap1 * tl.load(
            w_base + w_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        tap2 = tl.load(
            state_base + 2 * state_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32) * keep
        acc += tap2 * tl.load(
            w_base + 2 * w_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32)
    acc += xv * tl.load(
        w_base + (KERNEL_WIDTH - 1) * w_tap_stride, mask=mask_c, other=0.0
    ).to(tl.float32)

    # Shift the retained window by one: new_state[j] = window[j+1]. Every source
    # tap is already in a register, so the state is never re-read.
    state_ty = state_ptr.dtype.element_ty
    if KERNEL_WIDTH >= 3:
        tl.store(state_base, tap1.to(state_ty), mask=mask_c)
    if KERNEL_WIDTH >= 4:
        tl.store(state_base + state_tap_stride, tap2.to(state_ty), mask=mask_c)
    tl.store(
        state_base + (KERNEL_WIDTH - 2) * state_tap_stride,
        xv.to(state_ty),
        mask=mask_c,
    )

    out = acc * tl.sigmoid(acc)
    tl.store(
        out_ptr + req * out_row_stride + offs_c,
        out.to(out_ptr.dtype.element_ty),
        mask=mask_c,
    )


def qwen35_causal_conv1d_decode_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation retained for parity testing and CPU fallback."""

    state = state_pool.index_select(0, state_indices)
    keep = (positions != 0).to(state.dtype)[:, None, None]
    state = state * keep
    window = torch.cat((state, x.unsqueeze(-1)), dim=-1)
    output = (window * weight.squeeze(1).unsqueeze(0)).sum(dim=-1)
    state_pool.index_copy_(0, state_indices, window[:, :, 1:])
    return F.silu(output)


def qwen35_causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Graph-safe single-token causal convolution with in-place slot updates."""

    # MINISGL_QWEN35_CONV1D_EAGER=1 selects the pre-fusion path so an A/B can be
    # interleaved without editing or checking out source between runs.
    if not x.is_cuda or os.environ.get("MINISGL_QWEN35_CONV1D_EAGER") == "1":
        return qwen35_causal_conv1d_decode_eager(
            x, weight, state_pool, state_indices, positions
        )

    num_requests, num_channels = x.shape
    kernel_width = weight.shape[-1]
    if not 2 <= kernel_width <= 4:
        raise ValueError(
            f"qwen35_causal_conv1d_decode supports kernel_width 2..4, got "
            f"{kernel_width} from weight shape {tuple(weight.shape)}; the decode "
            f"kernel unrolls the retained taps into registers. Extend the "
            f"KERNEL_WIDTH cascade in _qwen35_causal_conv1d_decode_kernel to "
            f"support a wider convolution."
        )
    block_c = 128
    out = torch.empty_like(x)
    grid = (num_requests, triton.cdiv(num_channels, block_c))
    _qwen35_causal_conv1d_decode_kernel[grid](
        x,
        weight,
        state_pool,
        state_indices,
        positions,
        out,
        num_channels,
        x.stride(0),
        state_pool.stride(0),
        state_pool.stride(1),
        state_pool.stride(2),
        weight.stride(0),
        weight.stride(-1),
        out.stride(0),
        KERNEL_WIDTH=kernel_width,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def _qwen35_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    original_dtype = x.dtype
    x_f32 = x.float()
    # z was upcast twice here, which cost an extra full-size copy per layer.
    z_f32 = z.float()
    normalized = x_f32 * torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    gated = normalized * (z_f32 * torch.sigmoid(z_f32))
    return gated.to(original_dtype)


# Eager, this is ~13 ATen launches per linear-attention layer -- 36 of them per
# decode step, each materializing an fp32 intermediate the size of the value
# tensor. vLLM gets the same arithmetic as a single inductor kernel because its
# decoder carries `@support_torch_compile`; minisgl already uses this same local
# pattern for the MiniMax QK norm in `layers/norm.py`.
_qwen35_rmsnorm_gated_compiled = torch.compile(
    _qwen35_rmsnorm_gated, dynamic=True, fullgraph=True
)


def qwen35_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Per-value-head RMSNorm followed by the Qwen3.5 SiLU output gate."""

    # MINISGL_QWEN35_GATE_EAGER=1 selects the pre-compile path so an A/B can be
    # interleaved without editing or checking out source between runs, matching
    # MINISGL_QWEN35_CONV1D_EAGER above. Needed because the compiled reduction is
    # not bitwise-identical, so any correctness difference has to be attributable.
    if not x.is_cuda or os.environ.get("MINISGL_QWEN35_GATE_EAGER") == "1":
        return _qwen35_rmsnorm_gated(x, z, weight, eps)
    return _qwen35_rmsnorm_gated_compiled(x, z, weight, eps)


@triton.jit
def _qwen35_gdn_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    initial_state_ptr,
    cu_seqlens_ptr,
    output_ptr,
    final_state_ptr,
    H_Q: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    value_head = tl.program_id(1)
    value_block = tl.program_id(2)
    key_head = value_head // (H_V // H_Q)
    start = tl.load(cu_seqlens_ptr + sequence_idx)
    end = tl.load(cu_seqlens_ptr + sequence_idx + 1)

    key_offsets = tl.arange(0, BK)
    value_offsets = value_block * BV + tl.arange(0, BV)
    key_mask = key_offsets < K
    value_mask = value_offsets < V
    state_offsets = (
        ((sequence_idx * H_V + value_head) * V + value_offsets[:, None]) * K
        + key_offsets[None, :]
    )
    state_mask = value_mask[:, None] & key_mask[None, :]
    state = tl.load(
        initial_state_ptr + state_offsets, mask=state_mask, other=0.0
    ).to(tl.float32)

    token_idx = start
    while token_idx < end:
        q = tl.load(
            q_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (K**-0.5)
        k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        head_offset = token_idx * H_V + value_head
        state *= tl.exp(tl.load(g_ptr + head_offset).to(tl.float32))
        v = tl.load(
            v_ptr + head_offset * V + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        memory = tl.sum(state * k[None, :], axis=1)
        delta = (v - memory) * tl.load(beta_ptr + head_offset).to(tl.float32)
        state += delta[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], axis=1)
        tl.store(
            output_ptr + head_offset * V + value_offsets,
            output,
            mask=value_mask,
        )
        token_idx += 1

    tl.store(final_state_ptr + state_offsets, state, mask=state_mask)


def _qwen35_gdn_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.float().contiguous()
    beta = beta.float().contiguous()
    initial_state = initial_state.float().contiguous()
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    key_dim = q.shape[-1]
    value_dim = v.shape[-1]
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    grid = (
        initial_state.shape[0],
        v.shape[1],
        triton.cdiv(value_dim, block_value),
    )
    _qwen35_gdn_prefill_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        output,
        final_state,
        H_Q=q.shape[1],
        H_V=v.shape[1],
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        num_warps=4,
        num_stages=2,
    )
    return output, final_state


@triton.jit
def _qwen35_gdn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    state_indices_ptr,
    positions_ptr,
    output_ptr,
    H_Q: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # The value block is the fastest-varying grid axis so that consecutive CTAs
    # walk one request's recurrent state contiguously. With the token axis first,
    # a scheduled wave touched one 16 KiB slice from each of B slots that sit
    # 4 MiB apart, and this bandwidth-bound kernel lost ~13% to the scatter.
    value_block = tl.program_id(0)
    token_idx = tl.program_id(1)
    value_head = tl.program_id(2)
    key_head = value_head // (H_V // H_Q)

    key_offsets = tl.arange(0, BK)
    value_offsets = value_block * BV + tl.arange(0, BV)
    key_mask = key_offsets < K
    value_mask = value_offsets < V

    q = tl.load(
        q_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (K**-0.5)
    k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)

    slot = tl.load(state_indices_ptr + token_idx).to(tl.int64)
    position = tl.load(positions_ptr + token_idx)
    state_offsets = (
        ((slot * H_V + value_head) * V + value_offsets[:, None]) * K
        + key_offsets[None, :]
    )
    state_mask = value_mask[:, None] & key_mask[None, :]
    state = tl.load(state_ptr + state_offsets, mask=state_mask, other=0.0).to(
        tl.float32
    )
    state = tl.where(position == 0, 0.0, state)

    head_offset = token_idx * H_V + value_head
    a = tl.load(a_ptr + head_offset).to(tl.float32)
    b = tl.load(b_ptr + head_offset).to(tl.float32)
    A_log = tl.load(A_log_ptr + value_head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)
    softplus_arg = a + dt_bias
    softplus = tl.where(
        softplus_arg <= 20.0,
        tl.log(1.0 + tl.exp(softplus_arg)),
        softplus_arg,
    )
    state *= tl.exp(-tl.exp(A_log) * softplus)

    v = tl.load(
        v_ptr + (token_idx * H_V + value_head) * V + value_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    memory = tl.sum(state * k[None, :], axis=1)
    delta = (v - memory) * tl.sigmoid(b)
    state += delta[:, None] * k[None, :]
    output = tl.sum(state * q[None, :], axis=1)

    tl.store(state_ptr + state_offsets, state, mask=state_mask)
    tl.store(
        output_ptr + (token_idx * H_V + value_head) * V + value_offsets,
        output,
        mask=value_mask,
    )


def qwen35_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Fused single-token GDN update; mutates selected recurrent-state slots."""

    if not q.is_cuda:
        outputs = []
        for token_idx in range(q.shape[0]):
            slot = int(state_indices[token_idx])
            initial = state_pool[slot]
            if int(positions[token_idx]) == 0:
                initial = torch.zeros_like(initial)
            g, beta = qwen35_gdn_gates(
                a[token_idx], b[token_idx], A_log, dt_bias
            )
            output, final_state = qwen35_recurrent_gated_delta_rule(
                q[token_idx : token_idx + 1],
                k[token_idx : token_idx + 1],
                v[token_idx : token_idx + 1],
                g.unsqueeze(0),
                beta.unsqueeze(0),
                initial,
            )
            state_pool[slot].copy_(final_state)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    # q/k/v arrive as strided views of the fused conv output. Reading them in place
    # by stride was measured and REJECTED: it removes 108 copy launches/step worth
    # ~0.62 ms but costs the bandwidth-bound recurrent kernel 25.65 us per launch
    # (+0.92 ms/step), because materializing them also warms L2 for the read that
    # immediately follows. Net +0.30 ms/step, so the copies stay.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    output = torch.empty_like(v)
    key_dim = q.shape[-1]
    value_dim = v.shape[-1]
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    grid = (triton.cdiv(value_dim, block_value), q.shape[0], v.shape[1])
    _qwen35_gdn_decode_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state_pool,
        state_indices,
        positions,
        output,
        H_Q=q.shape[1],
        H_V=v.shape[1],
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        # Measured across B=32..512: 8 warps beat 4 by 7-12%. This is a
        # read-modify-write of a 4 MiB/request state, so it wants issue width to
        # cover memory latency rather than a narrow tile.
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = [
    "qwen35_causal_conv1d_decode",
    "qwen35_causal_conv1d_prefill",
    "qwen35_gdn_decode",
    "qwen35_gdn_gates",
    "qwen35_gdn_prefill",
    "qwen35_l2norm",
    "qwen35_recurrent_gated_delta_rule",
    "qwen35_recurrent_gated_delta_rule_packed",
    "qwen35_rmsnorm_gated",
]
