from __future__ import annotations
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Dict, Optional, Tuple
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
    parameters: Dict[str, Any] | None = None
    partial_rotary_factor: float | None = None


@dataclass(frozen=True)
class QuantConfig:
    method: str
    activation_scheme: str
    weight_block_size: Tuple[int, int]
    group_size: int | None = None
    ignored_modules: Tuple[str, ...] = ()
    kv_cache_dtype: str | None = None

    def is_ignored(self, module_name: str) -> bool:
        candidates = [module_name]
        language_prefix = "model.language_model."
        if module_name.startswith(language_prefix):
            candidates.append("model." + module_name.removeprefix(language_prefix))
        elif module_name.startswith("model."):
            candidates.append(language_prefix + module_name.removeprefix("model."))
        return any(
            fnmatchcase(candidate, pattern)
            for pattern in self.ignored_modules
            for candidate in candidates
        )


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    use_routing_bias: bool
    scoring_func: str
    model_type: str
    architectures: list[str]
    quant: Optional[QuantConfig] = None
    use_qk_norm: bool = False
    rope_parameters: Dict[str, Any] | None = None
    rope_scaling: Dict[str, Any] | None = None
    rope_theta: float = 10000.0
    partial_rotary_factor: float | None = None
    layer_types: Tuple[str, ...] = ()
    attn_output_gate: bool = False
    linear_conv_kernel_dim: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    shared_expert_intermediate_size: int = 0
    mamba_ssm_dtype: str = "float32"
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def num_full_attention_layers(self) -> int:
        return sum(layer_type == "full_attention" for layer_type in self.layer_types)

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(layer_type == "linear_attention" for layer_type in self.layer_types)

    @property
    def is_hybrid_attention(self) -> bool:
        return self.num_linear_attention_layers > 0

    def quant_for_modules(self, *module_names: str) -> QuantConfig | None:
        if self.quant is None:
            return None
        if any(self.quant.is_ignored(name) for name in module_names):
            return None
        return self.quant

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        def value(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        top = config
        text_config = value(config, "text_config", None)
        if text_config is not None:
            config = text_config

        num_attention_heads = int(value(config, "num_attention_heads"))
        hidden_size = int(value(config, "hidden_size"))
        num_kv_heads = value(config, "num_key_value_heads", num_attention_heads)
        head_dim = value(config, "head_dim", None) or hidden_size // num_attention_heads
        tie_word_embeddings = value(
            config,
            "tie_word_embeddings",
            value(top, "tie_word_embeddings", False),
        )
        model_type = value(config, "model_type", value(top, "model_type", "llama"))
        n_routed_experts = value(config, "n_routed_experts", None)
        if n_routed_experts is None:
            num_experts = value(
                config, "num_local_experts", value(config, "num_experts", 0)
            )
            n_routed_experts = num_experts
        else:
            num_experts = n_routed_experts
        num_experts_per_tok = value(config, "num_experts_per_tok", 0)
        moe_intermediate_size = int(
            value(config, "moe_intermediate_size", 0)
            or value(config, "intermediate_size", 0)
        )
        intermediate_size = int(
            value(config, "intermediate_size", 0) or moe_intermediate_size
        )
        qwen35_model_types = {"qwen3_5_moe", "qwen3_5_moe_text"}
        norm_topk_prob = value(config, "norm_topk_prob", model_type in qwen35_model_types)
        use_routing_bias = value(config, "use_routing_bias", False)
        scoring_func = value(config, "scoring_func", "softmax")
        if (
            model_type == "minimax_m2"
            and scoring_func == "sigmoid"
            and value(config, "norm_topk_prob", None) is None
        ):
            # vLLM hardcodes MiniMax M2 sigmoid-routing weights to be
            # renormalized; the HF config currently omits norm_topk_prob.
            norm_topk_prob = True
        architectures = value(
            config,
            "architectures",
            value(top, "architectures", ["LlamaForCausalLM"]),
        )
        use_qk_norm = value(config, "use_qk_norm", model_type in qwen35_model_types)
        num_layers = int(value(config, "num_hidden_layers"))
        layer_types_raw = value(config, "layer_types", None)
        if layer_types_raw is None:
            layer_types = ("full_attention",) * num_layers
        else:
            layer_types = tuple(str(item) for item in layer_types_raw)
            if len(layer_types) != num_layers:
                raise ValueError(
                    "layer_types length must equal num_hidden_layers: "
                    f"{len(layer_types)} != {num_layers}"
                )
            invalid_layer_types = set(layer_types) - {
                "full_attention",
                "linear_attention",
            }
            if invalid_layer_types:
                raise ValueError(f"Unsupported layer_types: {sorted(invalid_layer_types)}")
        model_extra: Dict[str, Any] = {}
        if model_type == "glm4_moe" or "Glm4MoeForCausalLM" in architectures:
            for attr, default in (
                ("n_shared_experts", 0),
                ("n_group", 0),
                ("topk_group", 0),
                ("first_k_dense_replace", 0),
                ("routed_scaling_factor", 1.0),
                ("attention_bias", False),
            ):
                value = getattr(config, attr, default)
                model_extra[attr] = default if value is None else value

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        # MiniMax M2 also uses partial rotary via rotary_dim.
        rope_scaling = value(config, "rope_scaling", value(top, "rope_scaling", None))
        rope_parameters = value(
            config, "rope_parameters", value(top, "rope_parameters", None)
        )
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", None) or value(
                config, "rope_theta", value(top, "rope_theta", None)
            )
            rope_scaling = rope_parameters
        else:
            rope_theta = value(config, "rope_theta", value(top, "rope_theta", None))
            if rope_theta is None and rope_scaling is not None:
                rope_theta = rope_scaling["rope_theta"]
            if rope_theta is None:
                rope_theta = 10000.0
        partial_rotary_factor = value(config, "partial_rotary_factor", None)
        if partial_rotary_factor is None and isinstance(rope_parameters, dict):
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is None and isinstance(rope_scaling, dict):
            partial_rotary_factor = rope_scaling.get("partial_rotary_factor")
        rotary_dim = value(config, "rotary_dim", None)
        if rotary_dim is None:
            rotary_dim = (
                int(head_dim * float(partial_rotary_factor))
                if partial_rotary_factor is not None
                else head_dim
            )

        quant: Optional[QuantConfig] = None
        quant_config = value(
            config,
            "quantization_config",
            value(top, "quantization_config", None),
        )
        if quant_config is not None:
            quant_dict = (
                quant_config
                if isinstance(quant_config, dict)
                else getattr(quant_config, "to_dict", lambda: {})()
            )
            quant_method = quant_dict.get("quant_method", "") or ""
            if "fp8" in quant_method:
                block_size = quant_dict.get("weight_block_size", [128, 128])
                quant = QuantConfig(
                    method="fp8",
                    activation_scheme=quant_dict.get("activation_scheme", "dynamic"),
                    weight_block_size=(int(block_size[0]), int(block_size[1])),
                    ignored_modules=tuple(
                        quant_dict.get("modules_to_not_convert", ())
                    ),
                )
            elif (
                quant_method == "compressed-tensors"
                and quant_dict.get("format") == "float-quantized"
            ):
                # GLM FP8 checkpoints store per-output-channel FP8 weights plus
                # `.weight_scale`.  Runtime code repacks these channel-scale
                # tensors into DeepGEMM's 128x128 block-scale FP8 format once
                # after loading.
                quant = QuantConfig(
                    method="fp8_channel",
                    activation_scheme="dynamic",
                    weight_block_size=(1, 1),
                )
            elif quant_method == "modelopt" and quant_dict.get("quant_algo") == "NVFP4":
                groups = quant_dict.get("config_groups", {})
                group0 = groups.get("group_0", {}) if isinstance(groups, dict) else {}
                weights = group0.get("weights", {}) if isinstance(group0, dict) else {}
                group_size = int(weights.get("group_size", 16))
                kv_scheme = quant_dict.get("kv_cache_scheme", {})
                quant = QuantConfig(
                    method="modelopt_nvfp4",
                    activation_scheme="static",
                    weight_block_size=(1, group_size),
                    group_size=group_size,
                    ignored_modules=tuple(quant_dict.get("ignore", ())),
                    kv_cache_dtype=(
                        f"fp{int(kv_scheme.get('num_bits', 8))}"
                        if isinstance(kv_scheme, dict) and kv_scheme
                        else None
                    ),
                )

        return cls(
            num_layers=num_layers,
            num_qo_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=int(value(config, "vocab_size")),
            intermediate_size=intermediate_size,
            hidden_act=str(value(config, "hidden_act")),
            rms_norm_eps=float(value(config, "rms_norm_eps")),
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=int(value(config, "max_position_embeddings")),
                base=rope_theta,
                scaling=rope_scaling,
                parameters=rope_parameters,
                partial_rotary_factor=partial_rotary_factor,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            use_routing_bias=use_routing_bias,
            scoring_func=scoring_func,
            model_type=model_type,
            architectures=architectures,
            quant=quant,
            use_qk_norm=use_qk_norm,
            rope_parameters=rope_parameters,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            layer_types=layer_types,
            attn_output_gate=bool(value(config, "attn_output_gate", False)),
            linear_conv_kernel_dim=int(value(config, "linear_conv_kernel_dim", 0)),
            linear_key_head_dim=int(value(config, "linear_key_head_dim", 0)),
            linear_value_head_dim=int(value(config, "linear_value_head_dim", 0)),
            linear_num_key_heads=int(value(config, "linear_num_key_heads", 0)),
            linear_num_value_heads=int(value(config, "linear_num_value_heads", 0)),
            shared_expert_intermediate_size=int(
                value(config, "shared_expert_intermediate_size", 0)
            ),
            mamba_ssm_dtype=str(value(config, "mamba_ssm_dtype", "float32")),
            model_extra=model_extra,
        )
