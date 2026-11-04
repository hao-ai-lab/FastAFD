"""Environment defaults that must be set before the first `import flashinfer` in any minisgl process.

They must take effect before any flashinfer import: the driver (`python -m minisgl`) and the Ray workers
(actor classes live in minisgl modules, whose flashinfer imports are function-local) import minisgl first.
"""

import os

# FlashInfer 0.6.11 reads FLASHINFER_USE_CUDA_NORM once, when flashinfer.norm is first imported
# (flashinfer/norm/__init__.py:57), and unless it is "1" sends rmsnorm, fused_add_rmsnorm and the
# 3-D (per-head q/k) rmsnorm to CuTe DSL kernels whenever nvidia-cutlass-dsl imports. FlashInfer 0.6.6
# always ran its CUDA JIT kernels, whose source 0.6.11 keeps unchanged, and the fused Qwen3.5 kernels
# (kernel/csrc/jit/moe_residual_rmsnorm_fp8_quant.cu, qwen35_full_attn_norm_rope_bf16_store.cu)
# reproduce those CUDA kernels byte for byte. The 0.6.11 upgrade was for its attention cubins only, so
# keep the CUDA norm kernels by default; FLASHINFER_USE_CUDA_NORM=0 opts into the CuTe DSL kernels.
# Same mechanism as vLLM's env_override.py, imported first by the package __init__.
os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")
