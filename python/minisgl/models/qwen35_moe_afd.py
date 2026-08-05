from __future__ import annotations

from typing import NamedTuple

import torch
from minisgl.layers import (
    BaseOP,
    LinearReplicated,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNormOffsetFused,
    VocabParallelEmbedding,
)
from minisgl.models.afd import ModelStageState
from minisgl.utils import nvtx_annotate

from .config import ModelConfig
from .qwen35_moe import (
    Qwen3_5FullAttention,
    Qwen3_5LinearAttention,
    Qwen3_5SharedExpert,
    qwen35_shared_output,
)
from .qwen3_moe_afd import qwen3_afd_topk


class Qwen3_5AfdTopK(NamedTuple):
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    router_logits: torch.Tensor | None = None
    renormalize: bool = True
    deepep_topk: bool = False
    shared_output: torch.Tensor | None = None
    dispatch_fp8: bool = False


def _qwen35_dispatch_uses_fp8(config: ModelConfig) -> bool:
    return bool(config.quant is not None and config.quant.method.startswith("fp8"))


class _Qwen3_5AfdGateAndShared(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = Qwen3_5SharedExpert(config, layer_id)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)


class Qwen3_5AfdAGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_type = config.layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5LinearAttention(config, layer_id)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config, layer_id)
        else:
            raise ValueError(f"Invalid Qwen3.5 layer type: {self.layer_type}")
        self.mlp = _Qwen3_5AfdGateAndShared(config, layer_id)
        self.input_layernorm = RMSNormOffsetFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormOffsetFused(
            config.hidden_size, config.rms_norm_eps
        )
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.dispatch_fp8 = _qwen35_dispatch_uses_fp8(config)
        self._layer_id = layer_id

    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> ModelStageState:
        hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn.forward(hidden_states)
        else:
            hidden_states = self.self_attn.forward(hidden_states)
        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        return ModelStageState(hidden_states=hidden_states, residual=residual)

    @nvtx_annotate("AFD_AG_Route_{}", layer_id_field="_layer_id")
    def route(self, hidden_states: torch.Tensor) -> Qwen3_5AfdTopK:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        base = qwen3_afd_topk(
            self.mlp.gate.forward(hidden_states),
            top_k=self.top_k,
            renormalize=self.renormalize,
            dispatch_fp8=self.dispatch_fp8,
        )
        shared = qwen35_shared_output(
            self.mlp.shared_expert.forward(hidden_states),
            self.mlp.shared_expert_gate.forward(hidden_states),
        )
        return Qwen3_5AfdTopK(
            topk_ids=base.topk_ids,
            topk_weights=base.topk_weights,
            router_logits=base.router_logits,
            renormalize=base.renormalize,
            deepep_topk=base.deepep_topk,
            shared_output=shared,
            dispatch_fp8=self.dispatch_fp8,
        )


class Qwen3_5AfdDenseRouterStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen3_5AfdAGLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormOffsetFused(config.hidden_size, config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        return ModelStageState(self.embed_tokens.forward(input_ids), residual=None)

    def forward_attention(self, layer_id: int, hidden_states, residual=None):
        return self.layers.op_list[layer_id].forward_attention(hidden_states, residual)

    def route(self, layer_id: int, hidden_states):
        return self.layers.op_list[layer_id].route(hidden_states)

    def finalize_hidden(self, hidden_states, residual=None):
        return self.norm.forward(hidden_states, residual)[0]


class Qwen3_5AfdDenseRouterForCausalLM(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5AfdDenseRouterStage(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward_attention(self, layer_id, hidden_states, residual=None):
        return self.model.forward_attention(layer_id, hidden_states, residual)

    def route(self, layer_id, hidden_states):
        return self.model.route(layer_id, hidden_states)

    def finalize_hidden(self, hidden_states, residual=None):
        return self.model.finalize_hidden(hidden_states, residual)

    def forward_lm_head(self, hidden_states):
        return self.lm_head.forward(hidden_states)


class _Qwen3_5AfdExpertMLP(BaseOP):
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


class Qwen3_5AfdEGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.mlp = _Qwen3_5AfdExpertMLP(config)
        self._layer_id = layer_id

    def run_experts(self, dispatch_output) -> torch.Tensor:
        return self.mlp.experts._run_moe_core(dispatch_output).hidden_states


class _Qwen3_5AfdExpertStageInner(BaseOP):
    def __init__(self, config: ModelConfig):
        self.layers = OPList(
            [Qwen3_5AfdEGLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.layers.op_list[layer_id].run_experts(dispatch_output)


class Qwen3_5AfdExpertStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = _Qwen3_5AfdExpertStageInner(config)
        self._dispatch_fp8 = _qwen35_dispatch_uses_fp8(config)

    def dispatch_uses_fp8(self) -> bool:
        return self._dispatch_fp8

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.model.run_experts(layer_id, dispatch_output)


__all__ = [
    "Qwen3_5AfdAGLayer",
    "Qwen3_5AfdDenseRouterForCausalLM",
    "Qwen3_5AfdExpertStage",
    "Qwen3_5AfdTopK",
]
