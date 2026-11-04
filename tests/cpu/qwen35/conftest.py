"""Pinned Qwen3.5 production checkpoint fixtures for isolated CPU tests."""

from __future__ import annotations

import copy
from functools import partial
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

import minisgl.distributed.info as dist_info
from minisgl.distributed.info import DistributedInfo

NUM_LAYERS = 60
LAYER_TYPES = [
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(NUM_LAYERS)
]


def _modules_to_not_convert(layer_types: list[str]) -> list[str]:
    """The BF16 modules of Qwen/Qwen3.5-397B-A17B-FP8 (block FP8 128x128): the GDN conv
    and b/a projections, the routers and the shared-expert gates."""
    skip = ["lm_head", "model.language_model.embed_tokens"]
    for i, kind in enumerate(layer_types):
        prefix = f"model.language_model.layers.{i}."
        if kind == "linear_attention":
            skip += [prefix + "linear_attn.conv1d", prefix + "linear_attn.in_proj_a",
                     prefix + "linear_attn.in_proj_b"]
        skip += [prefix + "mlp.gate", prefix + "mlp.shared_expert_gate"]
    return skip


# The fields of the public Qwen/Qwen3.5-397B-A17B-FP8 config.json that ModelConfig
# reads; the decode kernels are qualified for these shapes.
QWEN35_397B_FP8 = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "attention_bias": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "layer_types": LAYER_TYPES,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "moe_intermediate_size": 1024,
        "num_attention_heads": 32,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "num_hidden_layers": NUM_LAYERS,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "shared_expert_intermediate_size": 1024,
        "tie_word_embeddings": False,
        "vocab_size": 248320,
    },
    "quantization_config": {
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_per_tensor": False,
        "act_per_tensor": False, "weight_block_size": [128, 128],
        "modules_to_not_convert": _modules_to_not_convert(LAYER_TYPES),
    },
}


def _model_config_class():
    """Load models/config.py alone.

    The `minisgl.models` package imports Triton kernels at package scope, which the
    CPU lane does not install; config.py itself needs only transformers, so the
    real `ModelConfig.from_hf` mapping stays under test.
    """
    name = "_fastafd_cpu_qwen35_model_config"
    module = sys.modules.get(name)
    if module is None:
        path = Path(dist_info.__file__).resolve().parents[1] / "models" / "config.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
        spec.loader.exec_module(module)
    return module.ModelConfig


@pytest.fixture
def qwen_config():
    """Build the declared config, optionally overriding text_config or quantization fields."""
    model_config = _model_config_class()

    def build(*, text=None, quant=None, drop_quant: bool = False):
        raw = copy.deepcopy(QWEN35_397B_FP8)
        raw["text_config"].update(text or {})
        if drop_quant:
            del raw["quantization_config"]
        else:
            raw["quantization_config"].update(quant or {})
        return model_config.from_hf(raw)

    return build


@pytest.fixture
def decode():
    """Load pure model policy without the CUDA-dependent models package."""
    name = "_fastafd_cpu_qwen35_decode"
    path = Path(dist_info.__file__).resolve().parents[1] / "models" / "qwen35_decode.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass annotation resolution
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def afd_decode_flags(decode):
    return partial(decode.decode_policy, for_afd=True)

@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank=0, size=1))


_GDN_SOURCE = Path(dist_info.__file__).resolve().parents[1] / "kernel" / "qwen35_gdn.py"


@pytest.fixture(scope="module")
def gdn():
    """minisgl/kernel/qwen35_gdn.py loaded with a Triton stand-in scoped to its own import.

    The CPU lane has no Triton. The module needs it only for the @triton.jit decorators and the
    triton.cdiv / next_power_of_2 calls in its launch wrappers, which is all the stand-in provides;
    it is removed from sys.modules again so nothing else in the session sees it.
    """
    # The module calls torch.compile at import; build one wrapper first so torch's own optional-Triton
    # probes (inductor/dynamo imports) run against the real absence of Triton, not the stand-in.
    torch.compile(lambda t: t, dynamic=True, fullgraph=True)

    stub, language = ModuleType("triton"), ModuleType("triton.language")
    stub.jit = lambda fn: fn
    stub.cdiv = lambda a, b: -(-a // b)
    stub.next_power_of_2 = lambda n: 1 << max(n - 1, 0).bit_length()
    language.constexpr = object
    stub.language = language
    saved = {name: sys.modules.get(name) for name in ("triton", "triton.language")}
    sys.modules.update({"triton": stub, "triton.language": language})
    try:
        spec = importlib.util.spec_from_file_location("_qwen35_gdn_cpu_lane", _GDN_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module
