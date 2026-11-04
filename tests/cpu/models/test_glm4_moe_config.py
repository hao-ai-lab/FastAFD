"""GLM-4 MoE (GLM-4.5 / 4.6 / 4.7) config mapping: the fields the AFD model reads."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import minisgl


def _import_alone(name: str):
    """Import one minisgl module without running its package's __init__ (the
    `minisgl.models` package imports Triton kernels, which the CPU lane lacks)."""
    pkg = name.rpartition(".")[0]
    saved = sys.modules.get(pkg)
    if saved is None:
        stub = types.ModuleType(pkg)
        stub.__path__ = [str(Path(minisgl.__file__).parent.joinpath(*pkg.split(".")[1:]))]
        sys.modules[pkg] = stub
    try:
        return importlib.import_module(name)
    finally:
        if saved is None:
            del sys.modules[pkg]


_config = _import_alone("minisgl.models.config")

# A scaled-down GLM-4.6 config.json (same fields and value types).
GLM46 = {
    "architectures": ["Glm4MoeForCausalLM"],
    "model_type": "glm4_moe",
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 1000,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 1000000,
    "partial_rotary_factor": 0.5,
    "rope_scaling": None,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "attention_bias": True,
    "use_qk_norm": False,
    "tie_word_embeddings": False,
}


def test_glm4_moe_config_keeps_the_glm_fields() -> None:
    cfg = _config.ModelConfig.from_hf(dict(GLM46))
    extra = cfg.model_extra
    assert extra["n_shared_experts"] == 1
    assert extra["first_k_dense_replace"] == 1
    assert extra["routed_scaling_factor"] == 2.5
    assert extra["attention_bias"] is True
    assert (extra["n_group"], extra["topk_group"]) == (1, 1)
