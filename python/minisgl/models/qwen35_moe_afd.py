from __future__ import annotations

from collections import deque
from typing import NamedTuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel import qwen35_shared_gate
from minisgl.kernel.vllm_kernels import norm_enabled as vllm_qwen35_norm_enabled
from minisgl.kernel.vllm_kernels import silu_enabled as vllm_qwen35_silu_enabled
from minisgl.kernel.vllm_kernels import shared_gate_enabled as vllm_qwen35_shared_gate_enabled
from minisgl.kernel.vllm_shared_gate import shared_expert_gate as vllm_shared_expert_gate
from minisgl.kernel.qwen35_decode import fused_input_norm_quant
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
from .qwen35_decode import decode_policy
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
        self.shared_expert._vllm_silu = vllm_qwen35_silu_enabled()   # AFD attention side only
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)


class Qwen3_5AfdAGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_type = config.layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5LinearAttention(config, layer_id, for_afd=True)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config, layer_id, for_afd=True)
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

    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention_fused(
        self, routed: torch.Tensor, shared: torch.Tensor, residual: torch.Tensor
    ) -> ModelStageState:
        """Decode form of forward_attention; the previous layer's MoE add is fused in."""
        normed, prequant = fused_input_norm_quant(self.input_layernorm, routed, shared, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn.forward(normed, prequant=prequant)
        else:
            hidden_states = self.self_attn.forward(normed, prequant=prequant)
        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        return ModelStageState(hidden_states=hidden_states, residual=residual)

    # MINISGL_QWEN35_VLLM_NORM: vLLM's residual stream with its inductor norm kernels
    # (kernel/vllm_norms.py). Each returns (state, attention output); the stage keeps the attention output for the
    # next layer's input norm, which recomputes attn + residual as vLLM's does (the post-attention norm stores no sum).
    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention_vllm_first(self, hidden_states: torch.Tensor):
        from minisgl.kernel import vllm_norms as vnorm
        normed = vnorm.first_norm(hidden_states, self.input_layernorm.raw_weight)
        return self._attention_post_vllm(normed, None, hidden_states)

    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention_vllm(self, kind: str, routed, shared, attn_prev, residual):
        from minisgl.kernel import vllm_norms as vnorm
        residual, normed = vnorm.input_norm(kind, routed, shared, attn_prev, residual, self.input_layernorm.raw_weight)
        # the quant of the normed rows that feeds the FP8 in_proj_qkvz / qkv_proj: vLLM's QuantFP8 under
        # MINISGL_QWEN35_VLLM_QUANT, else FastAFD's per_token_cast (so each sub-switch stays isolated)
        from minisgl.layers.linear import _vllm_qwen35_quant
        if _vllm_qwen35_quant(normed):
            from minisgl.kernel.vllm_per_token_group_quant import per_token_group_quant_fp8_packed_for_deepgemm
            prequant = per_token_group_quant_fp8_packed_for_deepgemm(normed, 128)
        else:
            from minisgl.kernel import deepgemm
            prequant = deepgemm.per_token_cast_to_fp8(normed, use_ue8m0=True, gran_k=128, use_packed_ue8m0=True)
        return self._attention_post_vllm(normed, prequant, residual)

    def _attention_post_vllm(self, normed, prequant, residual):
        from minisgl.kernel import vllm_norms as vnorm
        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn.forward(normed, prequant=prequant)
        else:
            attn_out = self.self_attn.forward(normed, prequant=prequant)
        attn_out = attn_out.view(-1, attn_out.shape[-1])
        hidden_states = vnorm.post_norm(attn_out, residual, self.post_attention_layernorm.raw_weight,
                                        full=self.layer_type == "full_attention")[0]
        return ModelStageState(hidden_states=hidden_states, residual=residual), attn_out

    @nvtx_annotate("AFD_AG_Route_{}", layer_id_field="_layer_id")
    def route(self, hidden_states: torch.Tensor) -> Qwen3_5AfdTopK:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        base = qwen3_afd_topk(
            self.mlp.gate.forward(hidden_states),
            top_k=self.top_k,
            renormalize=self.renormalize,
            dispatch_fp8=self.dispatch_fp8,
        )
        shared_hidden = self.mlp.shared_expert.forward(hidden_states)
        # MINISGL_QWEN35_VLLM_SHARED_GATE (default: = MINISGL_QWEN35_VLLM_KERNELS), decode: vLLM's gate line for line and
        # eager (qwen2_moe.py:120-121 runs inside the opaque moe_forward_shared op): F.linear, ATen sigmoid, ATen mul.
        if vllm_qwen35_shared_gate_enabled() and get_global_ctx().batch.is_decode:
            shared = vllm_shared_expert_gate(
                hidden_states, self.mlp.shared_expert_gate.weight, shared_hidden)
        elif qwen35_shared_gate.enabled() and get_global_ctx().batch.is_decode:
            # MINISGL_QWEN35_FUSED_SHARED_GATE=1: the N=1 GEMV and sigmoid-multiply in one kernel.
            shared = qwen35_shared_gate.shared_gate(
                shared_hidden, hidden_states, self.mlp.shared_expert_gate.weight)
        else:
            shared = qwen35_shared_output(
                shared_hidden, self.mlp.shared_expert_gate.forward(hidden_states))
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
        # Fused input norm: in decode, route() keeps the shared
        # expert output here instead of letting the AFD runtime add it to the
        # routed output; the next layer's input kernel (or finalize_hidden) adds
        # it. One FIFO per layer keeps micro-batches in production order.
        self._fused_input_norm = decode_policy(config, get_tp_info().size, for_afd=True).fused_norm
        self._pending_shared = [deque() for _ in range(config.num_layers)]
        # MINISGL_QWEN35_VLLM_NORM: vLLM's norm kernels / residual stream in decode; the attention output of
        # each layer waits here (one FIFO per layer, like _pending_shared) for the next layer's input norm (or the final norm).
        self._layer_types = list(config.layer_types)
        self._vllm_norm = vllm_qwen35_norm_enabled()
        if self._vllm_norm and (not self._fused_input_norm or get_tp_info().size != 1
                                or config.hidden_size != 4096):
            raise ValueError("MINISGL_QWEN35_VLLM_NORM needs the block-FP8 fused-norm decode policy at TP1 (Qwen3.5-397B, hidden 4096)")
        self._pending_attn = [deque() for _ in range(config.num_layers)]

    def _take_pending_attn(self, layer_id: int, routed: torch.Tensor) -> torch.Tensor:
        pending = self._pending_attn[layer_id]
        if not pending:
            raise RuntimeError(f"No pending attention output for Qwen3.5 layer {layer_id}")
        attn = pending.popleft()
        if attn.shape != routed.shape:
            raise RuntimeError(f"Pending attention output does not match layer {layer_id} routed output")
        return attn

    def _fused_decode(self, residual) -> bool:
        return (self._fused_input_norm and residual is not None
                and get_global_ctx().batch.is_decode)

    def _take_pending_shared(self, layer_id: int, routed: torch.Tensor) -> torch.Tensor:
        pending = self._pending_shared[layer_id]
        if not pending:
            raise RuntimeError(f"No pending shared-expert output for Qwen3.5 layer {layer_id}")
        shared = pending.popleft()
        if shared is None or shared.shape != routed.shape:
            raise RuntimeError(f"Pending shared-expert output does not match layer {layer_id} routed output")
        return shared

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        if any(self._pending_shared) or any(self._pending_attn):
            raise RuntimeError("Stale pending shared-expert / attention outputs from an incomplete forward")
        return ModelStageState(self.embed_tokens.forward(input_ids), residual=None)

    def forward_attention(self, layer_id: int, hidden_states, residual=None):
        if self._vllm_norm and get_global_ctx().batch.is_decode:
            layer = self.layers.op_list[layer_id]
            if residual is None:
                state, attn_out = layer.forward_attention_vllm_first(hidden_states)
            else:
                from minisgl.kernel.vllm_norms import input_kind
                shared = self._take_pending_shared(layer_id - 1, hidden_states)
                attn_prev = self._take_pending_attn(layer_id - 1, hidden_states)
                kind = input_kind(self._layer_types[layer_id - 1], self._layer_types[layer_id])
                state, attn_out = layer.forward_attention_vllm(kind, hidden_states, shared, attn_prev, residual)
            self._pending_attn[layer_id].append(attn_out)
            return state
        if self._fused_decode(residual):
            shared = self._take_pending_shared(layer_id - 1, hidden_states)
            return self.layers.op_list[layer_id].forward_attention_fused(
                hidden_states, shared, residual)
        return self.layers.op_list[layer_id].forward_attention(hidden_states, residual)

    def route(self, layer_id: int, hidden_states):
        topk = self.layers.op_list[layer_id].route(hidden_states)
        if self._fused_input_norm and get_global_ctx().batch.is_decode:
            self._pending_shared[layer_id].append(topk.shared_output)
            topk = topk._replace(shared_output=None)
        return topk

    def finalize_hidden(self, hidden_states, residual=None):
        if self._vllm_norm and residual is not None and get_global_ctx().batch.is_decode:
            from minisgl.kernel import vllm_norms as vnorm
            last = len(self.layers.op_list) - 1
            shared = self._take_pending_shared(last, hidden_states)
            attn = self._take_pending_attn(last, hidden_states)
            return vnorm.final_norm(hidden_states.view(-1, hidden_states.shape[-1]), shared, attn, residual,
                                    self.norm.raw_weight).view(hidden_states.shape)
        if self._fused_decode(residual):
            # Same operand order as afd_moe_output: routed + shared.
            hidden_states = hidden_states + self._take_pending_shared(
                len(self.layers.op_list) - 1, hidden_states)
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
    def __init__(self, config: ModelConfig, layer_id: int):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            # Resolved per module: the checkpoint names the modules it leaves unquantized.
            quant=config.quant_for_modules(f"model.layers.{layer_id}.mlp.experts"),
            a2a_backend="none",
        )


class Qwen3_5AfdEGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.mlp = _Qwen3_5AfdExpertMLP(config, layer_id)
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
