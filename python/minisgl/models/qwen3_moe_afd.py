"""Qwen3-MoE decomposed for the afd AG/EG architecture.

afd keeps dense attention/router work on the AG side and expert work on the
EG side:
  - AG (attention stage): embed + LOCAL attention (qkv->attn->o) +
    router gate + norm + lm_head. It only routes and dispatches — never runs experts.
  - EG (experts only): the grouped expert FFN over dispatched tokens.

So we re-compose the SAME building blocks (RopeAttn, the router gate, MoELayer's
expert side, VocabParallelEmbedding, ParallelLMHead, RMSNormFused) into AG/EG
stages.

The actual dispatch/combine COMM is NOT here — it lives in the EG/AG GPU runners
(via DeepEPM2NAdapter). This module only produces the per-stage tensors:
  AG: forward_attention(h)->h, route(h)->(topk_ids, topk_weights)
  EG: run_experts(dispatch_output)->expert_out
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from minisgl.env import afd_env
from minisgl.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.layers.moe import MoELayer
from minisgl.models.afd import ModelStageState
from minisgl.utils import nvtx_annotate

from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3AfdTopK(NamedTuple):
    topk_ids: torch.Tensor | None      # [tokens, top_k] real model expert ids
    topk_weights: torch.Tensor | None  # [tokens, top_k] fp32
    router_logits: torch.Tensor | None = None
    renormalize: bool = False
    deepep_topk: bool = False
    dispatch_fp8: bool = True


def _qwen3_dispatch_uses_fp8(config: ModelConfig) -> bool:
    return bool(config.quant is not None and config.quant.method.startswith("fp8"))


def _use_fused_route(num_tokens: int, gate_weight: torch.Tensor) -> bool:
    """Tensor-core fused router (gate_topk: mma.sync GEMM + select kernels),
    replacing the cublas nvjet + splitK + topk_gating triple: 8.05us vs
    9.3us of kernel time per layer.  Validated e2e on the 235B b1104
    workload: 38.03ms vs 38.52ms with the host route (top10 = 1.0)."""
    if afd_env("MOE_BACKEND", "deepep") != "megamoe_m2n":
        return False
    max_tokens = 512
    return (
        num_tokens <= max_tokens
        and gate_weight.dtype == torch.bfloat16
        and gate_weight.shape[0] <= 256
        and gate_weight.shape[1] % 256 == 0
    )


def qwen3_afd_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    dispatch_fp8: bool = True,
) -> Qwen3AfdTopK:
    if afd_env("MOE_BACKEND", "deepep") != "megamoe_m2n":
        return Qwen3AfdTopK(
            topk_ids=None,
            topk_weights=None,
            router_logits=router_logits.contiguous(),
            renormalize=renormalize,
            dispatch_fp8=dispatch_fp8,
        )

    from minisgl.kernel import topk_softmax

    num_tokens = router_logits.shape[0]
    weights = torch.empty(
        (num_tokens, top_k),
        dtype=torch.float32,
        device=router_logits.device,
    )
    ids = torch.empty(
        (num_tokens, top_k),
        dtype=torch.int32,
        device=router_logits.device,
    )
    topk_softmax(
        weights,
        ids,
        router_logits.contiguous(),
        renormalize=renormalize,
    )
    return Qwen3AfdTopK(
        topk_ids=ids,
        topk_weights=weights,
        renormalize=renormalize,
        dispatch_fp8=dispatch_fp8,
    )



# ============================ AG: attention + router ============================


class _Qwen3AfdRouterGate(BaseOP):
    """Router gate at the legacy `.mlp.gate` path (no experts on the AG side)."""

    def __init__(self, config: ModelConfig):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)


class Qwen3AfdAGLayer(BaseOP):
    """AG layer: local attention + router gate. No experts."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        # router gate lives under `.mlp.gate` so checkpoint keys match the legacy
        # layout (model.layers.{i}.mlp.gate.weight) and load_weight finds them.
        self.mlp = _Qwen3AfdRouterGate(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.dispatch_fp8 = _qwen3_dispatch_uses_fp8(config)
        self._layer_id = layer_id

    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> ModelStageState:
        # norm -> qkv -> qk_norm+rope -> core attn (KV via ctx metadata) -> o_proj -> norm
        h, residual = self.input_layernorm.forward(hidden_states, residual)
        h = self.self_attn.forward(h)
        h, residual = self.post_attention_layernorm.forward(h, residual)
        return ModelStageState(hidden_states=h, residual=residual)

    @nvtx_annotate("AFD_AG_Route_{}", layer_id_field="_layer_id")
    def route(self, hidden_states: torch.Tensor) -> Qwen3AfdTopK:
        h = hidden_states.view(-1, hidden_states.shape[-1])
        if _use_fused_route(h.shape[0], self.mlp.gate.weight):
            # ONE kernel: gate GEMV + softmax + topk (+ renormalize) — replaces
            # the cublas gate GEMM + splitK + topk_gating launches (megamoe
            # decode path).
            from minisgl.kernel import gate_topk

            ids, weights = gate_topk(
                h, self.mlp.gate.weight, self.top_k,
                renormalize=self.renormalize,
            )
            return Qwen3AfdTopK(
                topk_ids=ids,
                topk_weights=weights,
                renormalize=self.renormalize,
                dispatch_fp8=self.dispatch_fp8,
            )
        logits = self.mlp.gate.forward(h)
        return qwen3_afd_topk(
            logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            dispatch_fp8=self.dispatch_fp8,
        )


class Qwen3AfdDenseRouterStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [Qwen3AfdAGLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        return ModelStageState(hidden_states=self.embed_tokens.forward(input_ids), residual=None)

    def forward_attention(self, layer_id: int, h: torch.Tensor, residual=None) -> ModelStageState:
        return self.layers.op_list[layer_id].forward_attention(h, residual)

    def route(self, layer_id: int, h: torch.Tensor) -> Qwen3AfdTopK:
        return self.layers.op_list[layer_id].route(h)

    def finalize_hidden(self, h: torch.Tensor, residual=None) -> torch.Tensor:
        return self.norm.forward(h, residual)[0]


class Qwen3AfdDenseRouterForCausalLM(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3AfdDenseRouterStage(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward_attention(self, layer_id, h, residual=None):
        return self.model.forward_attention(layer_id, h, residual)

    def route(self, layer_id, h):
        return self.model.route(layer_id, h)

    def finalize_hidden(self, h, residual=None):
        return self.model.finalize_hidden(h, residual)

    def forward_lm_head(self, h):
        return self.lm_head.forward(h)


# ================================ EG: experts only ================================


class _Qwen3AfdExpertMLP(BaseOP):
    """Experts at the legacy `.mlp.experts` path (EG side has no gate/attn)."""

    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant=config.quant,
            a2a_backend="none",
        )


class Qwen3AfdEGLayer(BaseOP):
    """EG layer: grouped expert FFN over dispatched tokens.

    Reuses MoELayer purely as the expert weight holder + DeepGEMM runner; its
    internal dispatcher is unused (the AG/EG comm is done by DeepEPM2NAdapter in
    the runner). `run_experts` runs pre_permute -> runner -> post_permute on a
    dispatch output and returns the expert-output tensor to send back via combine.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        # experts under `.mlp.experts` to match the legacy checkpoint layout
        # (model.layers.{i}.mlp.experts.gate_up_proj / down_proj).
        self.mlp = _Qwen3AfdExpertMLP(config)
        self._layer_id = layer_id

    @nvtx_annotate("AFD_EG_Experts_{}", layer_id_field="_layer_id")
    def run_experts(self, dispatch_output) -> torch.Tensor:
        combine_input = self.mlp.experts._run_moe_core(dispatch_output)
        return combine_input.hidden_states


class _Qwen3AfdExpertStageInner(BaseOP):
    def __init__(self, config: ModelConfig):
        self.layers = OPList(
            [Qwen3AfdEGLayer(config, i) for i in range(config.num_layers)]
        )

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.layers.op_list[layer_id].run_experts(dispatch_output)


class Qwen3AfdExpertStage(BaseOP):
    def __init__(self, config: ModelConfig):
        # `.model` wrapper so keys are model.layers.{i}.mlp.experts.* (matches loader).
        self.model = _Qwen3AfdExpertStageInner(config)
        self._dispatch_fp8 = _qwen3_dispatch_uses_fp8(config)

    def dispatch_uses_fp8(self) -> bool:
        return self._dispatch_fp8

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.model.run_experts(layer_id, dispatch_output)


__all__ = [
    "Qwen3AfdTopK",
    "qwen3_afd_topk",
    "Qwen3AfdAGLayer",
    "Qwen3AfdDenseRouterStage",
    "Qwen3AfdDenseRouterForCausalLM",
    "Qwen3AfdEGLayer",
    "Qwen3AfdExpertStage",
]
