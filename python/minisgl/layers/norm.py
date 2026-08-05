from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class RMSNormOffset(RMSNorm):
    """Gemma/Qwen3.5 RMSNorm with checkpoint multiplier ``1 + weight``.

    The offset is folded once while loading so existing FlashInfer and fused
    QK-norm kernels consume the effective multiplier without a per-token add.
    """

    def load_state_dict(self, state_dict, *, prefix="", _internal=False) -> None:
        key = f"{prefix}.weight" if prefix else "weight"
        item = state_dict.pop(key)
        assert self.weight.shape == item.shape and self.weight.dtype == item.dtype
        self.weight = item + 1
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class RMSNormOffsetFused(RMSNormFused):
    """Residual-fused form of :class:`RMSNormOffset`."""

    def load_state_dict(self, state_dict, *, prefix="", _internal=False) -> None:
        key = f"{prefix}.weight" if prefix else "weight"
        item = state_dict.pop(key)
        assert self.weight.shape == item.shape and self.weight.dtype == item.dtype
        self.weight = item + 1
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


def _qk_variance(q: torch.Tensor, k: torch.Tensor):
    q = q.float()
    k = k.float()
    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)
    return q, k, q_var, k_var


def _qk_apply_norm(
    q_f32: torch.Tensor,
    k_f32: torch.Tensor,
    q_var: torch.Tensor,
    k_var: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    orig_dtype: torch.dtype,
):
    # Store MiniMax Q/K norm weights in the model dtype, but match
    # vLLM/SGLang by doing the normalization arithmetic in FP32.
    q = (q_f32 * torch.rsqrt(q_var + eps) * q_weight.float()).to(orig_dtype)
    k = (k_f32 * torch.rsqrt(k_var + eps) * k_weight.float()).to(orig_dtype)
    return q, k


_qk_variance_compiled = torch.compile(_qk_variance, dynamic=True, fullgraph=True)
_qk_apply_norm_compiled = torch.compile(_qk_apply_norm, dynamic=True, fullgraph=True)


class RMSNormCrossHead(BaseOP):
    """MiniMax M2 cross-head RMSNorm.

    Variance is over the local Q/K vector. When tensor parallelism is used the
    scalar variances are all-reduced so every rank applies global normalization.
    """

    def __init__(self, full_dim: int, eps: float = 1e-6) -> None:
        from minisgl.distributed import DistributedCommunicator, get_tp_info

        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        self.eps = eps
        self.weight = torch.empty(full_dim // self.tp_size)
        self._comm = DistributedCommunicator()

    @staticmethod
    def forward_qk(
        q_norm: "RMSNormCrossHead",
        k_norm: "RMSNormCrossHead",
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = q.dtype
        q_f32, k_f32, q_var, k_var = _qk_variance_compiled(q, k)
        if q_norm.tp_size > 1:
            qk_var = torch.cat([q_var, k_var], dim=-1)
            q_norm._comm.all_reduce(qk_var)
            qk_var = qk_var / q_norm.tp_size
            q_var, k_var = qk_var.chunk(2, dim=-1)
        return _qk_apply_norm_compiled(
            q_f32,
            k_f32,
            q_var,
            k_var,
            q_norm.weight,
            k_norm.weight,
            q_norm.eps,
            orig_dtype,
        )
