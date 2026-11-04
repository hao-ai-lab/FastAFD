from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.layers import (
    BaseOP,
    LinearReplicated,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.models.afd import ModelStageState
from minisgl.utils import nvtx_annotate

from .minimax_m2 import MiniMaxM2Attn, MiniMaxM2TopK, minimax_m2_topk

if TYPE_CHECKING:
    from .config import ModelConfig


MiniMaxM2AfdTopK = MiniMaxM2TopK


class _MiniMaxM2RouterGate(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = (
            torch.empty(config.num_experts, dtype=torch.float32)
            if config.use_routing_bias
            else None
        )
        # The gate again in the order the router kernel reads with float4 loads (same
        # logits; 3 MB per layer). Not part of the state dict.
        self._gate_lane_major: torch.Tensor | None = None

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        weight = self.gate.weight
        self._gate_lane_major = None
        if weight.is_cuda and weight.dtype == torch.float32 and weight.shape[1] % 128 == 0:
            from minisgl.kernel.moe_topk import minimax_route_lane_major

            self._gate_lane_major = minimax_route_lane_major(weight)


class MiniMaxM2AfdAGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = MiniMaxM2Attn(config, layer_id)
        self.mlp = _MiniMaxM2RouterGate(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.renormalize = True
        self._layer_id = layer_id

    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> ModelStageState:
        h, residual = self.input_layernorm.forward(hidden_states, residual)
        h = self.self_attn.forward(h)
        h, residual = self.post_attention_layernorm.forward(h, residual)
        return ModelStageState(hidden_states=h, residual=residual)

    @nvtx_annotate("AFD_AG_Route_{}", layer_id_field="_layer_id")
    def route(
        self,
        hidden_states: torch.Tensor,
        *,
        quant_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> MiniMaxM2AfdTopK:
        h = hidden_states.view(-1, hidden_states.shape[-1])
        if (
            h.is_cuda
            and h.dtype == torch.bfloat16
            and self.mlp.gate.weight.dtype == torch.float32
            and self.scoring_func == "sigmoid"
            and self.mlp.e_score_correction_bias is not None
        ):
            from minisgl.kernel import minimax_route_topk_fp32_dot_topk

            ids, weights = minimax_route_topk_fp32_dot_topk(
                h,
                self.mlp.gate.weight,
                self.mlp.e_score_correction_bias,
                self.top_k,
                renormalize=self.renormalize,
                quant_out=quant_out,
                pre_sigmoid=quant_out is not None,
                gate_weight_lane_major=self.mlp._gate_lane_major,
            )
            return MiniMaxM2AfdTopK(
                topk_ids=ids,
                topk_weights=weights,
                renormalize=self.renormalize,
                external_quant=quant_out is not None,
            )
        router_logits = self.mlp.gate.forward(h.to(torch.float32))
        return minimax_m2_topk(
            router_logits,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            score_correction_bias=self.mlp.e_score_correction_bias,
            renormalize=self.renormalize,
        )

    def route_for_megamoe(
        self,
        hidden_states: torch.Tensor,
        adapter,
        *,
        lane: int = 0,
    ) -> MiniMaxM2AfdTopK:
        h = hidden_states.view(-1, hidden_states.shape[-1])
        buf = adapter.buffers[int(lane) % len(adapter.buffers)]
        n = int(h.shape[0])
        quant_out = (
            buf.x[:n],
            buf.x_sf[:n],
            buf.topk_weights[:n, : self.top_k],
        )
        return self.route(hidden_states, quant_out=quant_out)


class MiniMaxM2AfdDenseRouterStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [MiniMaxM2AfdAGLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        return ModelStageState(hidden_states=self.embed_tokens.forward(input_ids), residual=None)

    def forward_attention(self, layer_id: int, h: torch.Tensor, residual=None) -> ModelStageState:
        return self.layers.op_list[layer_id].forward_attention(h, residual)

    def route(self, layer_id: int, h: torch.Tensor) -> MiniMaxM2AfdTopK:
        return self.layers.op_list[layer_id].route(h)

    def route_for_megamoe(
        self,
        layer_id: int,
        h: torch.Tensor,
        adapter,
        *,
        lane: int = 0,
    ) -> MiniMaxM2AfdTopK:
        return self.layers.op_list[layer_id].route_for_megamoe(
            h,
            adapter,
            lane=lane,
        )

    def finalize_hidden(self, h: torch.Tensor, residual=None) -> torch.Tensor:
        return self.norm.forward(h, residual)[0]


class MiniMaxM2AfdDenseRouterForCausalLM(BaseOP):
    # route() is the router alone, local to this rank (no TP collective, shared expert or
    # dense layer): the AFD MB pipeline may run it on the lane stream (MINISGL_AFD_ROUTE_ON_LANE)
    afd_route_is_local = True

    def __init__(self, config: ModelConfig):
        self.model = MiniMaxM2AfdDenseRouterStage(config)
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

    def route_for_megamoe(self, layer_id, h, adapter, *, lane: int = 0):
        return self.model.route_for_megamoe(layer_id, h, adapter, lane=lane)

    def finalize_hidden(self, h, residual=None):
        return self.model.finalize_hidden(h, residual)

    def forward_lm_head(self, h):
        return self.lm_head.forward(h)


class _MiniMaxM2AfdExpertMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            quant=config.quant,
            a2a_backend="none",
        )


class MiniMaxM2AfdEGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.mlp = _MiniMaxM2AfdExpertMLP(config)
        self._layer_id = layer_id

    @nvtx_annotate("AFD_EG_Experts_{}", layer_id_field="_layer_id")
    def run_experts(self, dispatch_output) -> torch.Tensor:
        combine_input = self.mlp.experts._run_moe_core(dispatch_output)
        return combine_input.hidden_states


class _MiniMaxM2AfdExpertStageInner(BaseOP):
    def __init__(self, config: ModelConfig):
        self.layers = OPList(
            [MiniMaxM2AfdEGLayer(config, i) for i in range(config.num_layers)]
        )

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.layers.op_list[layer_id].run_experts(dispatch_output)


class MiniMaxM2AfdExpertStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = _MiniMaxM2AfdExpertStageInner(config)

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.model.run_experts(layer_id, dispatch_output)


__all__ = [
    "MiniMaxM2AfdTopK",
    "MiniMaxM2AfdAGLayer",
    "MiniMaxM2AfdDenseRouterStage",
    "MiniMaxM2AfdDenseRouterForCausalLM",
    "MiniMaxM2AfdEGLayer",
    "MiniMaxM2AfdExpertStage",
]
