from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class MoeA2ABackend(Enum):
    NONE = "none"
    DEEPEP = "deepep"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value!r}")

    def is_none(self) -> bool:
        return self == MoeA2ABackend.NONE

    def is_deepep(self) -> bool:
        return self == MoeA2ABackend.DEEPEP


class MoeRunnerBackend(Enum):
    AUTO = "auto"
    TRITON = "triton"
    TRITON_FP8 = "triton_fp8"
    TRITON_KERNELS = "triton_kernel"
    DEEP_GEMM = "deep_gemm"
    FLASHINFER_NVFP4 = "flashinfer_nvfp4"

    def is_auto(self) -> bool:
        return self == MoeRunnerBackend.AUTO

    def is_triton(self) -> bool:
        return self == MoeRunnerBackend.TRITON

    def is_triton_fp8(self) -> bool:
        return self == MoeRunnerBackend.TRITON_FP8

    def is_triton_kernels(self) -> bool:
        return self == MoeRunnerBackend.TRITON_KERNELS

    def is_deep_gemm(self) -> bool:
        return self == MoeRunnerBackend.DEEP_GEMM

    def is_flashinfer_nvfp4(self) -> bool:
        return self == MoeRunnerBackend.FLASHINFER_NVFP4


@dataclass
class MoeRunnerConfig:
    num_experts: int | None = None
    num_local_experts: int | None = None
    hidden_size: int | None = None
    intermediate_size_per_partition: int | None = None
    top_k: int | None = None
    activation: str = "silu"
    apply_router_weight_on_input: bool = False
    renormalize: bool = True
    ep_size: int = 1
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    params_dtype: torch.dtype | None = None
    runner_backend: str | None = None


class MoeRunner(ABC):
    def __init__(self, config: MoeRunnerConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    @abstractmethod
    def apply(self, dispatch_output: Any, layer: Any) -> torch.Tensor: ...
