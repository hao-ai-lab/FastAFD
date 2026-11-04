"""vLLM db9527a4's attention-output gate for Qwen3.5-397B (MINISGL_QWEN35_VLLM_FULLATTN): the inductor-generated
triton_poi_fused_mul_per_token_group_fp8_quant_packed_sigmoid_view_0, CAPTURED from vLLM's own run and vendored verbatim.

vLLM computes attn_output * torch.sigmoid(gate) (qwen3_next.py:458) inside its compiled graph; inductor emits this
kernel (sigmoid in fp32, product stored bf16) and vLLM then quantizes its output with QuantFP8 for o_proj.
Source: vLLM db9527a4's Qwen3.5-397B decode run, inductor file
<torchinductor cache>/lh/clhq726y75ajdmelkdw6lf3zd5yxpr6zdln2q5hf5gatlf6qox5l.py (sha256_12 cbc22ba117a8), lines from @triton.jit to the end byte
for byte (sha256_12 of that text 2f251ef54254). The inductor decorator above it (@triton_heuristics.pointwise: size hints + the
autotune candidate list) is replaced by a fixed launch with the candidate vLLM's TRACED run used
(nsys traces of vLLM's decode at two batch sizes: block 128, REG 25 = vLLM Triton
entry 72BHD2A5PZOG compiled from this file): XBLOCK 1024, num_warps 4, num_stages 1, grid (xnumel // 1024,).
The kernel has no mask (inductor proved xnumel % XBLOCK == 0: xnumel = tokens x 8192): inputs must be contiguous
[n, 8192] bf16 (vLLM's FMHA output and the contiguous gate of _fused_qk_rmsnorm_rope_gate_kernel).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CAPTURED_FILE = "lh/clhq726y75ajdmelkdw6lf3zd5yxpr6zdln2q5hf5gatlf6qox5l.py"
CAPTURED_SHA12 = "cbc22ba117a8"
JIT_TEXT_SHA12 = "2f251ef54254"
XBLOCK, NUM_WARPS, NUM_STAGES = 1024, 4, 1   # vLLM's traced candidate (Triton entry 72BHD2A5PZOG)


@triton.jit
def triton_poi_fused_mul_per_token_group_fp8_quant_packed_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)


def attn_output_gate(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``attn_output * sigmoid(gate)`` -> bf16 [n, 8192], as vLLM's compiled graph runs it."""
    n = attn_output.shape[0]
    a = attn_output.reshape(n, -1)
    if (a.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16 or not a.is_contiguous() or not gate.is_contiguous()
            or gate.shape != a.shape):
        raise ValueError(f"vLLM o-gate needs contiguous bf16 [n, H*D] attention output and gate, got {tuple(a.shape)} "
                         f"{a.dtype} {a.stride()} / {tuple(gate.shape)} {gate.dtype} {gate.stride()}")
    xnumel = a.numel()
    if xnumel % XBLOCK != 0:
        raise ValueError(f"vLLM o-gate kernel has no mask: numel {xnumel} must be a multiple of {XBLOCK}")
    out = torch.empty_like(a)
    if xnumel:
        triton_poi_fused_mul_per_token_group_fp8_quant_packed_sigmoid_view_0[(xnumel // XBLOCK,)](
            a, gate, out, xnumel, XBLOCK=XBLOCK, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


__all__ = ["attn_output_gate"]
