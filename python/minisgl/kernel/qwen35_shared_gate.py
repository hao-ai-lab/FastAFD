# SPDX-License-Identifier: MIT
"""The Qwen3.5 shared-expert output gate in one kernel.

The AFD router stage computes ``shared * sigmoid(shared_expert_gate(hidden))`` per layer as
a cuBLAS GEMV with N=1 (dot_kernel + reduce_1Block_kernel) and an inductor sigmoid-multiply:
three launches, about 9 us per layer at B=186 (nsys). Here one program
per row computes the dot product in FP32, rounds the logit to BF16 as the BF16 linear does,
and writes ``shared * sigmoid(logit)`` in FP32 rounded to BF16, as the inductor kernel does.
Only the dot product's summation order differs from cuBLAS, so a logit can differ by one
BF16 ulp.
Decode only, behind MINISGL_QWEN35_FUSED_SHARED_GATE=1 (unset: the three-launch path).
"""

from __future__ import annotations

import os

import torch

SWITCH = "MINISGL_QWEN35_FUSED_SHARED_GATE"


def enabled() -> bool:
    return os.environ.get(SWITCH) == "1"


def reference(shared: torch.Tensor, hidden: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """The unfused arithmetic (BF16 linear, then shared * sigmoid in FP32)."""
    logit = torch.nn.functional.linear(hidden, gate_weight)
    return (shared.float() * torch.sigmoid(logit.float())).to(shared.dtype)


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _qwen35_shared_gate_kernel(hidden_ptr, weight_ptr, shared_ptr, out_ptr, logit_ptr,
                                hidden_stride, shared_stride, out_stride, HIDDEN: tl.constexpr,
                                STORE_LOGIT: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, HIDDEN)
        h = tl.load(hidden_ptr + row * hidden_stride + cols).to(tl.float32)
        w = tl.load(weight_ptr + cols).to(tl.float32)
        logit = tl.sum(h * w, axis=0).to(tl.bfloat16)
        if STORE_LOGIT:  # tests only: the BF16 logit the gate used
            tl.store(logit_ptr + row, logit)
        gate = tl.sigmoid(logit.to(tl.float32))
        s = tl.load(shared_ptr + row * shared_stride + cols).to(tl.float32)
        tl.store(out_ptr + row * out_stride + cols, (s * gate).to(tl.bfloat16))

    return _qwen35_shared_gate_kernel


_KERNEL = None


def shared_gate(shared: torch.Tensor, hidden: torch.Tensor, gate_weight: torch.Tensor,
                *, logit_out: torch.Tensor | None = None) -> torch.Tensor:
    """``shared * sigmoid(hidden @ gate_weight.T)`` for BF16 [B, 4096] rows and a [1, 4096] gate.

    ``logit_out`` (tests only): a BF16 [B] buffer that receives the logits the gate used.
    """
    global _KERNEL
    if (hidden.ndim != 2 or shared.shape != hidden.shape or hidden.shape[1] != 4096
            or gate_weight.shape != (1, 4096) or not gate_weight.is_contiguous()
            or any(t.dtype != torch.bfloat16 for t in (shared, hidden, gate_weight))
            or hidden.stride(1) != 1 or shared.stride(1) != 1):
        raise RuntimeError("Fused shared-expert gate requires BF16 [B, 4096] rows and a [1, 4096] gate")
    if logit_out is not None and (logit_out.shape != (hidden.shape[0],) or logit_out.dtype != torch.bfloat16
                                  or logit_out.device != hidden.device or not logit_out.is_contiguous()):
        raise RuntimeError("Fused shared-expert gate logit_out must be a contiguous BF16 [B] buffer")
    if not hidden.is_cuda:
        if logit_out is not None:
            logit_out.copy_(torch.nn.functional.linear(hidden, gate_weight).view(-1))
        return reference(shared, hidden, gate_weight)
    if _KERNEL is None:
        _KERNEL = _kernel()
    out = torch.empty_like(shared, memory_format=torch.contiguous_format)
    if hidden.shape[0] > 0:
        _KERNEL[(hidden.shape[0],)](hidden, gate_weight, shared, out, out if logit_out is None else logit_out,
                                    hidden.stride(0), shared.stride(0), out.stride(0), HIDDEN=4096,
                                    STORE_LOGIT=logit_out is not None, num_warps=8)
    return out
