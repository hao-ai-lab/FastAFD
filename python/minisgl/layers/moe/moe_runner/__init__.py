from __future__ import annotations

from .base import MoeA2ABackend, MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from .deepgemm_grouped import DeepGEMMGroupedRunner
from .flashinfer_nvfp4 import FlashInferNvFp4Runner
from .triton import TritonRunner
from .triton_fp8 import TritonFp8Runner


def create_moe_runner(
    runner_backend: str,
    moe_runner_config: MoeRunnerConfig,
) -> MoeRunner:
    backend = MoeRunnerBackend(runner_backend)
    if backend.is_auto() or backend.is_triton():
        return TritonRunner(config=moe_runner_config)
    if backend.is_triton_fp8():
        return TritonFp8Runner(config=moe_runner_config)
    if backend.is_deep_gemm():
        return DeepGEMMGroupedRunner(config=moe_runner_config)
    if backend.is_flashinfer_nvfp4():
        return FlashInferNvFp4Runner(config=moe_runner_config)
    if backend.is_triton_kernels():
        raise NotImplementedError(
            "runner_backend='triton_kernel' is not implemented in the no-DeepEP EP port"
        )
    raise NotImplementedError(f"Unsupported MoE runner backend: {runner_backend!r}")


__all__ = [
    "MoeA2ABackend",
    "MoeRunner",
    "MoeRunnerBackend",
    "MoeRunnerConfig",
    "DeepGEMMGroupedRunner",
    "FlashInferNvFp4Runner",
    "TritonRunner",
    "TritonFp8Runner",
    "create_moe_runner",
]
