"""vLLM db9527a4's shared-expert gate for Qwen3.5-397B, line-exact and UNCOMPILED (MINISGL_QWEN35_VLLM_SHARED_GATE).

vLLM builds the Qwen3.5 shared expert as ``Qwen2MoeMLP`` with ``expert_gate = ReplicatedLinear(hidden, 1,
bias=False, quant_config=None)`` (qwen3_next.py:192-216) and hands it to FusedMoE as ``shared_experts``. It runs
inside the opaque custom op ``torch.ops.vllm.moe_forward_shared`` (fused_moe/runner/moe_runner.py:593; the
captured inductor graph of the Qwen3.5-397B run calls it as an extern op), so inductor never fuses it: it runs EAGER.

  vllm/model_executor/models/qwen2_moe.py:120-121 (sha256_12 382904901ed3, git blob_12 1c095d6f091c)
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out

  ``self.expert_gate(x)[0]`` = ReplicatedLinear.forward -> UnquantizedLinearMethod.apply -> the GEMM chosen by
  ``dispatch_unquantized_gemm("auto")`` on CUDA (layers/utils.py:602-614) = ``default_unquantized_gemm``
  (layers/utils.py:83-89, sha256_12 514076151e2a): ``torch.nn.functional.linear(x, weight, bias)``, bias None.

Kernels this launches (vLLM EP4 trace): cuBLAS N=1 GEMM (nvjet splitK + splitKreduce), ATen
sigmoid (bf16 [n, 1] out), ATen broadcast mul ([n, 1] * [n, 4096] -> bf16). FastAFD's compiled gate instead runs
F.linear + ``torch.compile(shared * sigmoid(logit))``: one inductor kernel with the sigmoid kept in fp32 (no bf16
rounding of the gate), i.e. a different kernel sequence and slightly different numerics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

UPSTREAM = "vllm/model_executor/models/qwen2_moe.py"
UPSTREAM_SHA12 = "382904901ed3"
UPSTREAM_LINES = (120, 121)
UPSTREAM_TEXT = (
    "        if self.expert_gate is not None:",
    "            out = F.sigmoid(self.expert_gate(x)[0]) * out",
)


def shared_expert_gate(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``F.sigmoid(self.expert_gate(x)[0]) * out`` with ``self.expert_gate(x)[0] = F.linear(x, weight, None)``.

    x: the shared expert's input [n, hidden] (bf16), weight: shared_expert_gate.weight [1, hidden] (bf16),
    out: the shared expert's output (down_proj) [n, hidden] (bf16). Same operand order as vLLM.
    """
    out = F.sigmoid(F.linear(x, weight, None)) * out
    return out


__all__ = ["shared_expert_gate"]
