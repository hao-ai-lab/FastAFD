"""DeepGEMM grouped MoE runner.

Vendored design reference: vLLM
submodules/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
(DeepGemmExperts) and experts/batched_deep_gemm_moe.py.
"""
from __future__ import annotations

from math import ceil
from typing import Any

import torch

from minisgl.kernel import deepgemm as deep_gemm
from minisgl.kernel.deepgemm_fused_quant import (
    persistent_masked_m_silu_mul_quant,
    persistent_psum_silu_mul_quant,
)
from minisgl.layers.activation import glm4_silu_and_mul, silu_and_mul

from .base import MoeRunner, MoeRunnerBackend, MoeRunnerConfig


def _expected_m_per_expert(
    *,
    n_tokens: int,
    top_k: int,
    num_experts: int,
    m_pad: int,
) -> int:
    """Static per-bucket M hint for DeepGEMM masked grouped heuristics."""
    if num_experts <= 0:
        return max(1, int(m_pad))
    expected = ceil((max(1, n_tokens) * max(1, top_k) / num_experts) * 4.0)
    return max(1, min(int(m_pad), int(expected)))


def _requant_weight_ue8m0_inplace(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_shape: tuple[int, int],
) -> None:
    """Mirror vLLM's DeepGEMM FP8 weight postprocess for SM100 UE8M0."""
    block_m, block_k = block_shape
    w_view = weight.reshape(-1, weight.shape[-2], weight.shape[-1])
    s_view = weight_scale.reshape(-1, *weight_scale.shape[-2:])
    for idx in range(w_view.size(0)):
        w_q = w_view[idx]
        s_old = s_view[idx]
        m_cur, k_cur = w_q.shape
        s_float = s_old.to(torch.float32)
        s_exp_r = torch.repeat_interleave(s_float, block_m, dim=0)
        s_exp = torch.repeat_interleave(s_exp_r, block_k, dim=1)
        w_dq = w_q.to(torch.float32) * s_exp[:m_cur, :k_cur]
        w_requant, s_requant = deep_gemm.per_block_cast_to_fp8(
            w_dq,
            use_ue8m0=True,
            gran_k=block_m,
        )
        w_q.copy_(w_requant)
        s_old.copy_(s_requant)


def _requant_channel_weight_ue8m0_inplace(
    weight: torch.Tensor,
    channel_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert GLM per-output-channel FP8 weights to DeepGEMM block-scale FP8.

    GLM compressed-tensors checkpoints store one scale per output row:
    ``weight.shape == [..., M, K]`` and ``scale.shape == [..., M, 1]``.
    DeepGEMM's SM100 path expects the weight still in FP8, but with 128x128
    UE8M0 block scales in the layout consumed by transform_sf_into_required_layout.
    Repack once per layer after loading, keeping the runtime path fully FP8.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"channel FP8 repack expects fp8 weight, got {weight.dtype}")
    if channel_scale.shape[:-1] != weight.shape[:-1] or channel_scale.shape[-1] != 1:
        raise RuntimeError(
            f"channel FP8 scale shape mismatch: weight={tuple(weight.shape)} "
            f"scale={tuple(channel_scale.shape)}"
        )
    w_view = weight.reshape(-1, weight.shape[-2], weight.shape[-1])
    s_view = channel_scale.reshape(-1, channel_scale.shape[-2], channel_scale.shape[-1])
    repacked_scales: list[torch.Tensor] = []
    for idx in range(w_view.size(0)):
        w_q = w_view[idx]
        s = s_view[idx].to(torch.float32)
        w_dq = w_q.to(torch.float32) * s
        w_requant, s_requant = deep_gemm.per_block_cast_to_fp8(
            w_dq,
            use_ue8m0=True,
            gran_k=128,
        )
        w_q.copy_(w_requant)
        repacked_scales.append(s_requant.contiguous())
    scale = torch.stack(repacked_scales, dim=0)
    return scale.reshape(*weight.shape[:-2], scale.shape[-2], scale.shape[-1])


class DeepGEMMGroupedRunner(MoeRunner):
    """DeepGEMM grouped MoE over masked or psum-contiguous expert layout."""

    def __init__(self, config: MoeRunnerConfig) -> None:
        super().__init__(config)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM

    def _invoke_bf16_grouped(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        expected_m: int,
    ) -> None:
        assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
        assert C.dtype == torch.bfloat16
        assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
        if expert_num_tokens.dtype != torch.int32:
            expert_num_tokens = expert_num_tokens.to(torch.int32)
        if not expert_num_tokens.is_contiguous():
            expert_num_tokens = expert_num_tokens.contiguous()

        kernel = getattr(deep_gemm, "bf16_m_grouped_gemm_nt_masked", None)
        if kernel is None:
            kernel = getattr(deep_gemm, "m_grouped_bf16_gemm_nt_masked", None)
        if kernel is None:
            raise RuntimeError(
                "deep_gemm is missing bf16/m_grouped_bf16_gemm_nt_masked"
            )
        kernel(
            A,
            B,
            C,
            expert_num_tokens,
            int(expected_m),
        )

    def _invoke_bf16_contiguous(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        psum_tokens_per_expert: torch.Tensor,
        expected_m: int,
    ) -> None:
        assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
        assert C.dtype == torch.bfloat16
        assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
        psum_tokens_per_expert = psum_tokens_per_expert.to(
            device=A.device,
            dtype=torch.int32,
        )
        if not psum_tokens_per_expert.is_contiguous():
            psum_tokens_per_expert = psum_tokens_per_expert.contiguous()

        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            A,
            B,
            C,
            psum_tokens_per_expert,
            use_psum_layout=True,
            expected_m_for_psum_layout=int(expected_m),
        )

    def _invoke_fp8_grouped(
        self,
        *,
        A: torch.Tensor,
        A_scale: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        C: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        expected_m: int,
    ) -> None:
        assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
        assert A_scale.dtype == torch.int32 and B_scale.dtype == torch.int32
        assert C.dtype == torch.bfloat16
        if expert_num_tokens.dtype != torch.int32:
            expert_num_tokens = expert_num_tokens.to(torch.int32)
        if not expert_num_tokens.is_contiguous():
            expert_num_tokens = expert_num_tokens.contiguous()

        kernel = getattr(deep_gemm, "fp8_m_grouped_gemm_nt_masked", None)
        if kernel is None:
            kernel = getattr(deep_gemm, "m_grouped_fp8_gemm_nt_masked", None)
        if kernel is None:
            raise RuntimeError("deep_gemm is missing fp8/m_grouped_fp8_gemm_nt_masked")
        kernel(
            (A, A_scale),
            (B, B_scale),
            C,
            expert_num_tokens,
            int(expected_m),
            disable_ue8m0_cast=False,
        )

    def _invoke_fp8_contiguous(
        self,
        *,
        A: torch.Tensor,
        A_scale: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        C: torch.Tensor,
        psum_tokens_per_expert: torch.Tensor,
        expected_m: int,
    ) -> None:
        assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
        assert C.dtype == torch.bfloat16
        psum_tokens_per_expert = psum_tokens_per_expert.to(
            device=A.device,
            dtype=torch.int32,
        )
        if not psum_tokens_per_expert.is_contiguous():
            psum_tokens_per_expert = psum_tokens_per_expert.contiguous()

        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A, A_scale),
            (B, B_scale),
            C,
            psum_tokens_per_expert,
            disable_ue8m0_cast=False,
            use_psum_layout=True,
            expected_m_for_psum_layout=int(expected_m),
        )

    def _get_fp8_weights_for_deepgemm(
        self,
        layer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = getattr(layer, "_deepgemm_fp8_weight_cache", None)
        if cached is not None:
            return cached

        quant_method = getattr(getattr(layer, "quant", None), "method", None)
        block_shape = tuple(getattr(layer.quant, "weight_block_size", (128, 128)))

        with torch.no_grad():
            if quant_method == "fp8_channel" or getattr(layer, "_fp8_scale_format", None) == "channel":
                layer.gate_up_proj_scale = _requant_channel_weight_ue8m0_inplace(
                    layer.gate_up_proj,
                    layer.gate_up_proj_scale,
                )
                layer.down_proj_scale = _requant_channel_weight_ue8m0_inplace(
                    layer.down_proj,
                    layer.down_proj_scale,
                )
                layer._fp8_scale_format = "block"
                block_shape = (128, 128)
            else:
                if block_shape != (128, 128):
                    raise RuntimeError(
                        f"DeepGEMM FP8 path requires 128x128 blocks, got {block_shape}"
                    )
                _requant_weight_ue8m0_inplace(
                    layer.gate_up_proj,
                    layer.gate_up_proj_scale,
                    block_shape,
                )
                _requant_weight_ue8m0_inplace(
                    layer.down_proj,
                    layer.down_proj_scale,
                    block_shape,
                )
            w1_scale = deep_gemm.transform_sf_into_required_layout(
                layer.gate_up_proj_scale,
                layer.gate_up_proj.size(1),
                layer.gate_up_proj.size(2),
                (1, 128, 128),
                layer.gate_up_proj.size(0),
                False,
                disable_ue8m0_cast=False,
            )
            w2_scale = deep_gemm.transform_sf_into_required_layout(
                layer.down_proj_scale,
                layer.down_proj.size(1),
                layer.down_proj.size(2),
                (1, 128, 128),
                layer.down_proj.size(0),
                False,
                disable_ue8m0_cast=False,
            )
        cached = (layer.gate_up_proj, w1_scale, layer.down_proj, w2_scale)
        layer._deepgemm_fp8_weight_cache = cached
        return cached

    @staticmethod
    def _contiguous_alignment() -> int:
        return int(deep_gemm.get_mk_alignment_for_contiguous_layout())

    @staticmethod
    def _activation_fn(name: str):
        """Look up the expert activation, refusing rather than defaulting.

        An unrecognised activation (e.g. `swigluoai`, a different function
        with an asymmetric clamp and an `(up + 1)` term) must not silently
        become plain SiLU; `moe/fused.py`'s FN_MAP raises on an unknown name
        too.
        """
        fns = {"silu": silu_and_mul, "glm4_silu": glm4_silu_and_mul}
        if name not in fns:
            raise NotImplementedError(
                f"DeepGEMM grouped runner has no {name!r} activation; it "
                f"implements {sorted(fns)}. Defaulting to SiLU would return a "
                "plausible but wrong tensor."
            )
        return fns[name]

    @staticmethod
    def _apply_expanded_topk_weights(output: torch.Tensor, dispatch_output: Any) -> torch.Tensor:
        topk_weights = getattr(dispatch_output, "topk_weights", None)
        if topk_weights is None:
            return output
        output.mul_(topk_weights.to(output.dtype).view(-1, 1))
        return output

    def _apply_psum_layout(self, dispatch_output: Any, layer: Any) -> torch.Tensor:
        x = dispatch_output.hidden_states
        psum_tokens_per_expert = dispatch_output.recv_count
        if psum_tokens_per_expert is None:
            raise RuntimeError("DeepGEMM psum layout requires recv_count metadata")
        if x.ndim != 2:
            raise RuntimeError(f"DeepGEMM psum layout expects 2D input, got {x.shape}")
        M, H = x.shape
        E = int(psum_tokens_per_expert.shape[0])
        w1 = layer.gate_up_proj
        w2 = layer.down_proj
        if M == 0:
            return x.new_empty((0, H), dtype=torch.bfloat16)
        if w1.shape[0] != E or w2.shape[0] != E:
            raise RuntimeError(
                f"weight expert count mismatch: w1.E={w1.shape[0]} w2.E={w2.shape[0]} "
                f"vs dispatch E={E}"
            )
        N = w1.shape[1]
        I = w2.shape[2]
        expected_m = max(1, int(M) // max(1, E))
        device = x.device
        intermediate_cache1 = torch.empty((M, N), dtype=torch.bfloat16, device=device)
        output = torch.empty((M, H), dtype=torch.bfloat16, device=device)

        if w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16:
            intermediate_cache2 = torch.empty((M, I), dtype=torch.bfloat16, device=device)
            self._invoke_bf16_contiguous(
                A=x.contiguous(),
                B=w1,
                C=intermediate_cache1,
                psum_tokens_per_expert=psum_tokens_per_expert,
                expected_m=expected_m,
            )
            self._activation_fn(layer.activation)(intermediate_cache1, out=intermediate_cache2)
            self._invoke_bf16_contiguous(
                A=intermediate_cache2,
                B=w2,
                C=output,
                psum_tokens_per_expert=psum_tokens_per_expert,
                expected_m=expected_m,
            )
            return self._apply_expanded_topk_weights(output, dispatch_output)

        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "DeepGEMM runner supports BF16 or FP8 weights; "
                f"got w1={w1.dtype}, w2={w2.dtype}"
            )

        if x.dtype == torch.float8_e4m3fn:
            x_q = x.contiguous()
            x_scale = getattr(dispatch_output, "hidden_states_scale", None)
            if x_scale is None:
                raise RuntimeError(
                    "DeepGEMM FP8 psum path requires hidden_states_scale"
                )
        else:
            x_q, x_scale = deep_gemm.per_token_cast_to_fp8(
                x.contiguous(),
                use_ue8m0=True,
                gran_k=128,
                use_packed_ue8m0=True,
            )
        w1, w1_scale, w2, w2_scale = self._get_fp8_weights_for_deepgemm(layer)
        self._invoke_fp8_contiguous(
            A=x_q,
            A_scale=x_scale,
            B=w1,
            B_scale=w1_scale,
            C=intermediate_cache1,
            psum_tokens_per_expert=psum_tokens_per_expert,
            expected_m=expected_m,
        )

        a2q, a2q_scale = persistent_psum_silu_mul_quant(
            intermediate_cache1,
            psum_tokens_per_expert,
            alignment=self._contiguous_alignment(),
            topk_weights=getattr(dispatch_output, "topk_weights", None),
            group_size=128,
        )
        self._invoke_fp8_contiguous(
            A=a2q,
            A_scale=a2q_scale,
            B=w2,
            B_scale=w2_scale,
            C=output,
            psum_tokens_per_expert=psum_tokens_per_expert,
            expected_m=expected_m,
        )
        return output

    def apply(self, dispatch_output: Any, layer: Any) -> torch.Tensor:
        x = dispatch_output.hidden_states
        if x.ndim == 2:
            return self._apply_psum_layout(dispatch_output, layer)
        expert_num_tokens = dispatch_output.recv_count
        topk_idx = getattr(
            dispatch_output,
            "topk_idx",
            getattr(dispatch_output, "topk_ids", None),
        )
        if topk_idx is None:
            raise RuntimeError("DeepGEMMGroupedRunner requires topk_idx/topk_ids metadata")

        if x.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "DeepGEMMGroupedRunner supports BF16 or FP8 activations; "
                f"got {x.dtype}"
            )

        E, M_pad, H = x.shape
        w1 = layer.gate_up_proj
        w2 = layer.down_proj
        if x.dtype == torch.bfloat16 and (w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16):
            raise ValueError(
                "DeepGEMMGroupedRunner BF16 path supports BF16 weights only; "
                f"got w1={w1.dtype}, w2={w2.dtype}"
            )
        if x.dtype == torch.float8_e4m3fn and (
            w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn
        ):
            raise ValueError(
                "DeepGEMMGroupedRunner FP8 path requires FP8 weights; "
                f"got w1={w1.dtype}, w2={w2.dtype}"
            )
        if w1.shape[0] != E or w2.shape[0] != E:
            raise RuntimeError(
                f"weight expert count mismatch: w1.E={w1.shape[0]} w2.E={w2.shape[0]} "
                f"vs dispatch E={E}"
            )

        N = w1.shape[1]
        I = w2.shape[2]
        expected_m = _expected_m_per_expert(
            n_tokens=int(topk_idx.shape[0]),
            top_k=int(getattr(layer, "top_k", self.config.top_k or 1)),
            num_experts=int(getattr(layer, "num_experts", self.config.num_experts or E)),
            m_pad=int(M_pad),
        )

        device = x.device
        intermediate_cache1 = torch.empty((E, M_pad, N), dtype=torch.bfloat16, device=device)
        output = torch.empty((E, M_pad, H), dtype=torch.bfloat16, device=device)

        if x.dtype == torch.bfloat16:
            intermediate_cache2 = torch.empty((E, M_pad, I), dtype=torch.bfloat16, device=device)
            self._invoke_bf16_grouped(
                A=x.contiguous(),
                B=w1,
                C=intermediate_cache1,
                expert_num_tokens=expert_num_tokens,
                expected_m=expected_m,
            )

            self._activation_fn(layer.activation)(
                intermediate_cache1.view(E * M_pad, N),
                out=intermediate_cache2.view(E * M_pad, I),
            )

            self._invoke_bf16_grouped(
                A=intermediate_cache2,
                B=w2,
                C=output,
                expert_num_tokens=expert_num_tokens,
                expected_m=expected_m,
            )
            return output

        x_scale = getattr(dispatch_output, "hidden_states_scale", None)
        if x_scale is None:
            raise RuntimeError("DeepGEMM FP8 path requires DeepEP-LL activation scales")
        w1, w1_scale, w2, w2_scale = self._get_fp8_weights_for_deepgemm(layer)

        self._invoke_fp8_grouped(
            A=x.contiguous(),
            A_scale=x_scale,
            B=w1,
            B_scale=w1_scale,
            C=intermediate_cache1,
            expert_num_tokens=expert_num_tokens,
            expected_m=expected_m,
        )

        a2q, a2q_scale = persistent_masked_m_silu_mul_quant(
            intermediate_cache1,
            expert_num_tokens,
            group_size=128,
        )

        self._invoke_fp8_grouped(
            A=a2q,
            A_scale=a2q_scale,
            B=w2,
            B_scale=w2_scale,
            C=output,
            expert_num_tokens=expert_num_tokens,
            expected_m=expected_m,
        )

        return output


__all__ = ["DeepGEMMGroupedRunner"]
