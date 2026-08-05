from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel.qwen35_gdn import (
    qwen35_causal_conv1d_decode,
    qwen35_causal_conv1d_prefill,
    qwen35_gdn_decode,
    qwen35_gdn_gates,
    qwen35_gdn_prefill,
    qwen35_rmsnorm_gated,
)
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVGatedMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormOffset,
    RMSNormOffsetFused,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class _Qwen3_5DepthwiseConv1d(BaseOP):
    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Qwen3.5 depthwise convolution requires request state")


class Qwen3_5LinearAttention(BaseOP):
    """TP-sharded Qwen3.5 Gated DeltaNet with request-scoped recurrent state."""

    def __init__(self, config: ModelConfig, layer_id: int):
        tp_size = get_tp_info().size
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        local_key_heads = div_even(config.linear_num_key_heads, tp_size)
        local_value_heads = div_even(config.linear_num_value_heads, tp_size)
        local_conv_dim = (
            2 * local_key_heads * config.linear_key_head_dim
            + local_value_heads * config.linear_value_head_dim
        )
        module_prefix = f"model.layers.{layer_id}.linear_attn"

        self.in_proj_qkv = LinearColParallelMerged(
            config.hidden_size,
            [key_dim, key_dim, value_dim],
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.in_proj_qkv"),
        )
        self.in_proj_z = LinearColParallelMerged(
            config.hidden_size,
            [value_dim],
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.in_proj_z"),
        )
        self.in_proj_b = LinearColParallelMerged(
            config.hidden_size,
            [config.linear_num_value_heads],
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.in_proj_b"),
        )
        self.in_proj_a = LinearColParallelMerged(
            config.hidden_size,
            [config.linear_num_value_heads],
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.in_proj_a"),
        )
        self.conv1d = _Qwen3_5DepthwiseConv1d(
            local_conv_dim, config.linear_conv_kernel_dim
        )
        self.dt_bias = torch.empty(local_value_heads)
        self.A_log = torch.empty(local_value_heads, dtype=torch.float32)
        self.norm = RMSNorm(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearOProj(
            value_dim,
            config.hidden_size,
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.out_proj"),
        )
        self._num_key_heads = local_key_heads
        self._num_value_heads = local_value_heads
        self._key_head_dim = config.linear_key_head_dim
        self._value_head_dim = config.linear_value_head_dim
        self._local_key_dim = local_key_heads * self._key_head_dim
        self._local_value_dim = local_value_heads * self._value_head_dim
        self._norm_eps = config.rms_norm_eps
        self._layer_id = layer_id

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        state_pool = ctx.gdn_state_pool
        if state_pool is None:
            raise RuntimeError("Qwen3.5 GDN state pool is not initialized")
        layer_state = state_pool.layer_state(self._layer_id)

        mixed_qkv = self.in_proj_qkv.forward(hidden_states)
        z = self.in_proj_z.forward(hidden_states).view(
            -1, self._num_value_heads, self._value_head_dim
        )
        b = self.in_proj_b.forward(hidden_states).view(-1, self._num_value_heads)
        a = self.in_proj_a.forward(hidden_states).view(-1, self._num_value_heads)

        if batch.is_decode:
            mixed_qkv = qwen35_causal_conv1d_decode(
                mixed_qkv,
                self.conv1d.weight,
                layer_state.conv,
                batch.state_indices,
                batch.positions,
            )
        else:
            cu_seqlens, sequence_slots, first_positions = self._prefill_state_plan(batch)
            initial_conv = layer_state.conv.index_select(0, sequence_slots)
            mixed_qkv, final_conv = qwen35_causal_conv1d_prefill(
                mixed_qkv,
                self.conv1d.weight,
                initial_conv,
                cu_seqlens,
                first_positions,
            )
            layer_state.conv.index_copy_(0, sequence_slots, final_conv)

        query, key, value = mixed_qkv.split(
            [self._local_key_dim, self._local_key_dim, self._local_value_dim],
            dim=-1,
        )
        query = query.view(-1, self._num_key_heads, self._key_head_dim)
        key = key.view(-1, self._num_key_heads, self._key_head_dim)
        value = value.view(-1, self._num_value_heads, self._value_head_dim)

        if batch.is_decode:
            core_output = qwen35_gdn_decode(
                query,
                key,
                value,
                a,
                b,
                self.A_log,
                self.dt_bias,
                layer_state.recurrent,
                batch.state_indices,
                batch.positions,
            )
        else:
            g, beta = qwen35_gdn_gates(a, b, self.A_log, self.dt_bias)
            initial_recurrent = layer_state.recurrent.index_select(0, sequence_slots)
            keep = (first_positions != 0).to(initial_recurrent.dtype)
            initial_recurrent = initial_recurrent * keep[:, None, None, None]
            core_output, final_recurrent = qwen35_gdn_prefill(
                query,
                key,
                value,
                g,
                beta,
                initial_recurrent,
                cu_seqlens,
            )
            layer_state.recurrent.index_copy_(0, sequence_slots, final_recurrent)

        gated = qwen35_rmsnorm_gated(
            core_output, z, self.norm.weight, self._norm_eps
        )
        return self.out_proj.forward(gated.flatten(1))

    @staticmethod
    def _prefill_state_plan(batch):
        cached = getattr(batch, "_qwen35_gdn_prefill_plan", None)
        if cached is not None:
            return cached
        lengths = [int(req.extend_len) for req in batch.padded_reqs]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu_seqlens = torch.tensor(
            boundaries, dtype=torch.int32, device=batch.positions.device
        )
        starts = cu_seqlens[:-1].to(torch.int64)
        sequence_slots = batch.state_indices.index_select(0, starts)
        first_positions = batch.positions.index_select(0, starts)
        plan = (cu_seqlens, sequence_slots, first_positions)
        batch._qwen35_gdn_prefill_plan = plan
        return plan


class Qwen3_5FullAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.q_size = self.num_qo_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        module_prefix = f"model.layers.{layer_id}.self_attn"
        self.qkv_proj = LinearQKVGatedMerged(
            config.hidden_size,
            config.head_dim,
            config.num_qo_heads,
            config.num_kv_heads,
            has_bias=False,
            quant=config.quant_for_modules(
                f"{module_prefix}.q_proj",
                f"{module_prefix}.k_proj",
                f"{module_prefix}.v_proj",
            ),
        )
        self.q_norm = RMSNormOffset(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormOffset(config.head_dim, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            config.num_qo_heads * config.head_dim,
            config.hidden_size,
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.o_proj"),
        )

    def split_query_gate(self, q_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading = q_gate.shape[:-1]
        per_head = q_gate.view(*leading, self.num_qo_heads, 2 * self.head_dim)
        query, gate = per_head.chunk(2, dim=-1)
        return query.reshape(*leading, self.q_size), gate.reshape(*leading, self.q_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(hidden_states)
        q_gate, key, value = qkv.split(
            [2 * self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        query, gate = self.split_query_gate(q_gate)
        attn_output = self.attn.forward(torch.cat((query, key, value), dim=-1))
        return self.o_proj.forward(attn_output * torch.sigmoid(gate))


class Qwen3_5SharedExpert(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        intermediate_size = config.shared_expert_intermediate_size
        module_prefix = f"model.layers.{layer_id}.mlp.shared_expert"
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
            quant=config.quant_for_modules(
                f"{module_prefix}.gate_proj",
                f"{module_prefix}.up_proj",
            ),
        )
        self.down_proj = LinearRowParallel(
            intermediate_size,
            config.hidden_size,
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.down_proj"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden_states)))


def qwen35_shared_output(
    shared_hidden: torch.Tensor, gate_logits: torch.Tensor
) -> torch.Tensor:
    return shared_hidden * torch.sigmoid(gate_logits)


class Qwen3_5MoeMLP(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant=config.quant,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = Qwen3_5SharedExpert(config, layer_id)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        routed = self.experts.forward(
            hidden_states=hidden_states,
            router_logits=self.gate.forward(hidden_states),
        )
        shared = qwen35_shared_output(
            self.shared_expert.forward(hidden_states),
            self.shared_expert_gate.forward(hidden_states),
        )
        return routed + shared


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_type = config.layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5LinearAttention(config, layer_id)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config, layer_id)
        else:
            raise ValueError(f"Invalid Qwen3.5 layer type: {self.layer_type}")
        self.mlp = Qwen3_5MoeMLP(config, layer_id)
        self.input_layernorm = RMSNormOffsetFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormOffsetFused(
            config.hidden_size, config.rms_norm_eps
        )
        self._layer_id = layer_id

    def forward_token_mixer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            return self.linear_attn.forward(hidden_states)
        return self.self_attn.forward(hidden_states)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
        hidden_states = self.forward_token_mixer(hidden_states)
        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        return self.mlp.forward(hidden_states), residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormOffsetFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens.forward(input_ids)
        residual = None
        for layer in self.layers.op_list:
            hidden_states, residual = layer.forward(hidden_states, residual)
        return self.norm.forward(hidden_states, residual)[0]


class Qwen3_5MoeForConditionalGeneration(BaseLLMModel):
    """Text-only Qwen3.5-MoE implementation for the multimodal checkpoint."""

    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        hidden_states = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden_states)


__all__ = [
    "Qwen3_5FullAttention",
    "Qwen3_5LinearAttention",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5SharedExpert",
    "qwen35_shared_output",
]
