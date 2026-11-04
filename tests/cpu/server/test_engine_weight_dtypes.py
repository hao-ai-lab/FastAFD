"""Colocated engine weight casts: quantization scales load as float32, as on the AFD path."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch

import minisgl


def _import_alone(name: str):
    """Import one minisgl module without running its package's __init__: the
    `minisgl.models` package imports Triton kernels at package scope, which the CPU
    lane does not install."""
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


cast = _import_alone("minisgl.models.weight").cast_checkpoint_tensor


def test_bf16_block_scales_load_as_float32() -> None:
    # Qwen3-235B-A22B-Instruct-2507-FP8 stores its dense block scales in bf16
    bf16 = torch.bfloat16
    assert cast("model.layers.0.self_attn.qkv_proj.weight_scale", torch.ones(1, 1, dtype=bf16), bf16).dtype \
        == torch.float32
    assert cast("model.layers.0.mlp.experts.w13_weight_scale", torch.ones(2, 1, 1, dtype=torch.float16),
                bf16).dtype == torch.float32
    assert cast("model.layers.0.self_attn.o_proj.weight_scale", torch.ones(1, 1), bf16).dtype == torch.float32


def test_packed_tensors_are_left_alone() -> None:
    fp8 = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    assert cast("model.layers.0.self_attn.qkv_proj.weight", fp8, torch.bfloat16).dtype == torch.float8_e4m3fn
    packed = torch.zeros(2, dtype=torch.uint8)
    assert cast("model.layers.0.mlp.experts.w13_weight_scale", packed, torch.bfloat16).dtype == torch.uint8


def test_other_tensors_take_the_model_dtype_unless_kept_float32() -> None:
    w = torch.ones(4, dtype=torch.float32)
    assert cast("model.layers.0.input_layernorm.weight", w, torch.bfloat16).dtype == torch.bfloat16
    assert cast("model.layers.0.mlp.gate.weight", w.bfloat16(), torch.bfloat16, float32=True).dtype \
        == torch.float32
