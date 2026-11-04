"""vLLM db9527a4's FP8 activation quant for DeepGEMM (MINISGL_QWEN35_VLLM_QUANT): per_token_group_quant_8bit_packed_register_kernel.

vLLM's FP8 block linear quantizes its input with QuantFP8.forward_cuda (quantization/input_quant_fp8.py:83-103,
sha256_12 7ab6276f1e2d) -> ``per_token_group_quant_fp8_packed_for_deepgemm(x, group_size=128, use_ue8m0=True)``
(quantization/utils/fp8_utils.py:685-756, sha256_12 6ab854317510), re-created here line for line: the same output
allocations (FP8 values ``empty_like``, int32 packed UE8M0 scales [mn, ceil(groups/4)] with strides (1, align4(mn))),
eps 1e-10, fp8 min/max = torch.finfo(float8_e4m3fn) (quant_utils.py:94-102, CUDA), and the call into
``torch.ops._C.per_token_group_fp8_quant_packed``, whose CUDA source is vendored in
csrc/jit/vllm_per_token_group_quant.cu (kernel lines 530-695 byte for byte; host side = its lines 699-858 for
bf16 -> e4m3fn, group 128, same grid / block / PDL launch). The vLLM 0.31 compiled op cannot be loaded into
torch 2.10, hence the source build.
Replaces FastAFD's per_token_cast_fp8_packed_kernel (kernel/fp8_quant.py per_token_cast_to_fp8_cuda) at the
attention-side call sites: layers/linear.py _forward_fp8 and kernel/qwen35_decode.py (unfused QKV|z quant).
"""

from __future__ import annotations

import functools

import torch

FP8 = torch.float8_e4m3fn


@functools.cache
def _module():
    from minisgl.kernel.utils import load_jit

    return load_jit(
        "vllm_per_token_group_quant",
        cuda_files=["vllm_per_token_group_quant.cu"],
        cuda_wrappers=[("quant_packed", "VllmPerTokenGroupQuantPackedKernel::run")],
    )


def per_token_group_quant_fp8_packed_for_deepgemm(
    x: torch.Tensor, group_size: int = 128, eps: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8_utils.py:685-756 with use_ue8m0=True: returns (x_q FP8 like x, x_s_packed int32 [mn, ceil(g/4)])."""
    dtype = FP8
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"
    finfo = torch.finfo(dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    hidden_dim = x.shape[-1]
    mn = x.numel() // hidden_dim
    num_groups_per_row = hidden_dim // group_size
    k_num_packed_sf_k = (num_groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4

    x_s_packed = torch.empty_strided(
        (mn, k_num_packed_sf_k),
        (1, tma_aligned_mn),
        device=x.device,
        dtype=torch.int32,
    )

    x_contiguous = x.contiguous()
    x_q_local = torch.empty_like(x_contiguous, device=x.device, dtype=dtype)

    _module().quant_packed(
        x_contiguous.view(mn, hidden_dim),
        x_q_local.view(mn, hidden_dim).view(torch.uint8),
        x_s_packed,
        group_size,
        eps,
        fp8_min,
        fp8_max,
    )
    # return a tensor with the original logical shape.
    x_q = x_q_local.view_as(x)
    return x_q, x_s_packed


def reference_packed(x: torch.Tensor, group_size: int = 128, eps: float = 1e-10,
                     values: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU / torch reference of the vendored kernel's math (per_token_group_quant.cu:530-695, read from the .cu):
    per group of 128: absmax = max(eps, |x|) (fp32), y_s = max(absmax / max_8bit, 1e-10), UE8M0 byte = exponent bits of
    y_s + (mantissa != 0) (a ceil to a power of two), q = clamp(x * (1 / 2^(byte - 127)), min_8bit, max_8bit) -> e4m3fn
    (RNE; |q| <= 448 so the saturating / non-saturating conversions agree); scales packed 4 bytes per int32 (group g ->
    byte g % 4 of word g // 4), column-major [mn, ceil(g/4)] with stride align4(mn)."""
    mn, k = x.shape
    g = k // group_size
    xf = x.float().view(mn, g, group_size)
    amax = torch.maximum(xf.abs().amax(dim=-1), torch.tensor(eps, dtype=torch.float32))
    y_s = torch.maximum(amax / 448.0, torch.tensor(1e-10, dtype=torch.float32))
    bits = y_s.view(torch.int32)
    e8 = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    if values:
        inv = 1.0 / (e8 << 23).view(torch.float32)
        q = (xf * inv.unsqueeze(-1)).clamp(-448.0, 448.0).to(FP8).view(mn, k)
    else:
        q = None
    pk = (g + 3) // 4
    e8p = torch.zeros(mn, pk * 4, dtype=torch.int32)
    e8p[:, :g] = e8
    words = (e8p[:, 0::4] | (e8p[:, 1::4] << 8) | (e8p[:, 2::4] << 16) | (e8p[:, 3::4] << 24))
    tma = ((mn + 3) // 4) * 4
    s = torch.empty_strided((mn, pk), (1, tma), dtype=torch.int32)
    s.copy_(words)
    return q, s


__all__ = ["per_token_group_quant_fp8_packed_for_deepgemm", "reference_packed"]
