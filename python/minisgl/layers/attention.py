from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import freeze_rope_scaling, get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=freeze_rope_scaling(rotary_config.scaling),
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def _can_use_fused_qk_norm_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> bool:
        backend = os.environ.get("MINISGL_QK_NORM_ROPE_BACKEND", "auto").lower()
        if backend in ("0", "false", "off", "disabled", "flashinfer"):
            return False
        if backend not in ("auto", "cuda"):
            raise RuntimeError(
                "MINISGL_QK_NORM_ROPE_BACKEND must be 'auto', 'cuda', or 'flashinfer/off', "
                f"got {backend!r}"
            )
        if self.q_norm is None or self.k_norm is None:
            return False
        supported = (
            self.head_dim == 128
            and q.is_cuda
            and k.is_cuda
            and positions.is_cuda
            and positions.dtype in (torch.int32, torch.int64)
            and q.dtype in (torch.bfloat16, torch.float16)
            and k.dtype == q.dtype
            and self.q_norm.weight.is_cuda
            and self.k_norm.weight.is_cuda
            and self.q_norm.weight.dtype == q.dtype
            and self.k_norm.weight.dtype == q.dtype
            and self.q_norm.eps == self.k_norm.eps
            and self.rotary.cos_sin_cache.is_cuda
            and self.rotary.cos_sin_cache.dtype == torch.float32
            and self.rotary.cos_sin_cache.shape[-1] == self.head_dim
            and q.stride(-1) == 1
            and k.stride(-1) == 1
        )
        if not supported and backend == "cuda":
            raise RuntimeError(
                "qk_norm_rope_cuda requested but inputs are unsupported: "
                f"head_dim={self.head_dim}, q_dtype={q.dtype}, k_dtype={k.dtype}, "
                f"positions_dtype={positions.dtype}, q_cuda={q.is_cuda}, k_cuda={k.is_cuda}"
            )
        return supported

    def apply_qk_norm_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._can_use_fused_qk_norm_rope(positions, q, k):
            from minisgl.kernel import qk_norm_rope_cuda

            assert self.q_norm is not None and self.k_norm is not None
            qk_norm_rope_cuda(
                positions,
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary.cos_sin_cache,
                q_heads=self.num_qo_heads,
                kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.q_norm.eps,
            )
            return q, k

        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        return self.rotary.forward(positions, q, k)

    def forward(
        self,
        qkv: torch.Tensor,
        *,
        qk_already_rotary_applied: bool = False,
        kv_store_merged: bool = False,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        merged_store = bool(kv_store_merged)
        if not qk_already_rotary_applied:
            merged_store = self._try_fused_rope_store_kv(ctx, q, k, v)
        if not qk_already_rotary_applied and not merged_store:
            q, k = self.apply_qk_norm_rope(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        if merged_store:
            prev_flag = ctx.batch.afd_kv_store_merged
            ctx.batch.afd_kv_store_merged = True
            try:
                o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
            finally:
                ctx.batch.afd_kv_store_merged = prev_flag
        else:
            o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)

    def _try_fused_rope_store_kv(self, ctx, q, k, v) -> bool:
        """qk_norm_rope + KV-cache store in ONE kernel (replaces the separate
        store_kv_cache launch).  Falls back to the split path whenever any
        precondition fails."""
        positions = ctx.batch.positions
        out_loc = getattr(ctx.batch, "out_loc", None)
        kvcache = getattr(ctx.attn_backend, "kvcache", None)
        if out_loc is None or kvcache is None:
            return False
        if not self._can_use_fused_qk_norm_rope(positions, q, k):
            return False
        try:
            k_cache = kvcache.k_cache(self.layer_id)
            v_cache = kvcache.v_cache(self.layer_id)
        except Exception:
            return False
        if k_cache.dtype != q.dtype or v.dtype != q.dtype:
            return False
        kv_dim = self.num_kv_heads * self.head_dim
        if k_cache.numel() % kv_dim != 0:
            return False
        from minisgl.kernel import qk_norm_rope_store_kv_cuda

        assert self.q_norm is not None and self.k_norm is not None
        qk_norm_rope_store_kv_cuda(
            ctx.batch.positions, q, k, v,
            self.q_norm.weight, self.k_norm.weight,
            self.rotary.cos_sin_cache,
            k_cache.view(-1, kv_dim), v_cache.view(-1, kv_dim),
            out_loc,
            q_heads=self.num_qo_heads,
            kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            eps=self.q_norm.eps,
        )
        return True
