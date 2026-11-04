"""The block-FP8 Qwen3.5-397B switches, CPU contracts.

MINISGL_QWEN35_FP8_FUSED_DECODE=1 selects the dense decode fusions for
Qwen/Qwen3.5-397B-A17B-FP8 (block FP8 128x128) on the AFD attention side; unset, the
policy is the generic one. MINISGL_QWEN35_MERGED_QKVZ=1 runs decode
QKV|z as one FP8 GEMM: here the merged operands are checked structurally against the
separate ones in tests/kernel/qwen35/test_block_fp8_merge.py (the kernel lane: it
imports Triton); the bit identity of the GEMM itself is a device test,
tests/kernel/test_qwen35_fp8_gdn_in_proj.py.
"""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SWITCH = "MINISGL_QWEN35_FP8_FUSED_DECODE"


def _fixtures():
    path = Path(__file__).resolve().parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_qwen35_declared_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DECLARED = _fixtures()


def _block_fp8_raw(*, skip_extra=(), skip_drop=(), block=(128, 128)) -> dict:
    """The fields of Qwen/Qwen3.5-397B-A17B-FP8's config.json that ModelConfig reads."""
    raw = copy.deepcopy(_DECLARED.QWEN35_397B_FP8)
    skip = _DECLARED._modules_to_not_convert(_DECLARED.LAYER_TYPES)
    skip = [name for name in skip if name not in skip_drop] + list(skip_extra)
    raw["quantization_config"] = {
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_per_tensor": False,
        "act_per_tensor": False, "weight_block_size": list(block), "modules_to_not_convert": skip,
    }
    return raw


@pytest.fixture
def block_fp8():
    model_config = _DECLARED._model_config_class()
    return lambda **kw: model_config.from_hf(_block_fp8_raw(**kw))


@pytest.fixture(autouse=True)
def _switch_unset(monkeypatch):
    for key in (SWITCH, "FLASHINFER_USE_CUDA_NORM", "MINISGL_QWEN35_MERGED_QKVZ", "MINISGL_AFD_SPEC"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("value", ["", "0", "none", "off"])
def test_switch_accepts_spec_decoding_off(monkeypatch, block_fp8, decode, value):
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv("MINISGL_AFD_SPEC", value)
    assert decode.decode_policy(block_fp8(), 1, for_afd=True).shared_quant


# --- policy -------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_unset_switch_keeps_the_generic_policy(monkeypatch, block_fp8, decode, value):
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    assert decode.decode_policy(block_fp8(), 1, for_afd=True) == decode.Qwen35DecodePolicy()


def test_switch_selects_the_dense_fusions_only(monkeypatch, block_fp8, decode):
    monkeypatch.setenv(SWITCH, "1")
    policy = decode.decode_policy(block_fp8(), 1, for_afd=True)
    assert policy == decode.Qwen35DecodePolicy(shared_quant=True, fused_norm=True, attn_fused=True)
    assert policy.full_attention_decode == "fused_bf16_kv"


def test_switch_leaves_colocated_execution_alone(monkeypatch, block_fp8, decode):
    monkeypatch.setenv(SWITCH, "1")
    assert decode.decode_policy(block_fp8(), 1, for_afd=False) == decode.Qwen35DecodePolicy()


def test_switch_refuses_other_checkpoints(monkeypatch, block_fp8, decode):
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(ValueError, match="block-FP8"):
        decode.decode_policy(block_fp8(block=(64, 64)), 1, for_afd=True)
    with pytest.raises(ValueError, match="model_type"):
        decode.decode_policy(replace(block_fp8(), model_type="qwen3_moe"), 1, for_afd=True)
    with pytest.raises(ValueError, match="block-FP8"):
        decode.decode_policy(replace(block_fp8(), quant=None), 1, for_afd=True)


@pytest.mark.parametrize("tp_size", [0, 2, 4])
def test_switch_requires_tp1(monkeypatch, block_fp8, decode, tp_size):
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(ValueError, match="TP1"):
        decode.decode_policy(block_fp8(), tp_size, for_afd=True)


@pytest.mark.parametrize(("key", "value", "message"), [
    ("FLASHINFER_USE_CUDA_NORM", "0", "FLASHINFER_USE_CUDA_NORM"),
    ("MINISGL_AFD_SPEC", "eagle3", "speculative decoding"),
])
def test_switch_refuses_other_kernel_environments(monkeypatch, block_fp8, decode, key, value, message):
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=message):
        decode.decode_policy(block_fp8(), 1, for_afd=True)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("hidden_size", 3072, "hidden_size"),
    ("linear_num_key_heads", 8, "GDN heads"),
    ("linear_num_value_heads", 32, "GDN heads"),
    ("linear_key_head_dim", 64, "GDN heads"),
    ("linear_value_head_dim", 64, "GDN heads"),
    ("num_qo_heads", 16, "full-attention heads"),
    ("num_kv_heads", 4, "full-attention heads"),
    ("head_dim", 128, "full-attention heads"),
])
def test_switch_checks_kernel_shapes(monkeypatch, block_fp8, decode, field, value, message):
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(ValueError, match=message):
        decode.decode_policy(replace(block_fp8(), **{field: value}), 1, for_afd=True)


@pytest.mark.parametrize("layers", [["linear_attention"] * 4, ["full_attention"] * 4])
def test_switch_requires_both_layer_kinds(monkeypatch, block_fp8, decode, layers):
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(ValueError, match="linear and full attention"):
        decode.decode_policy(replace(block_fp8(), layer_types=layers), 1, for_afd=True)


def test_switch_does_not_skip_unknown_layers(monkeypatch, block_fp8, decode):
    monkeypatch.setenv(SWITCH, "1")
    config = block_fp8()
    layers = list(config.layer_types)
    layers[-2] = "sliding_attention"
    with pytest.raises(ValueError, match="Unsupported Qwen3.5 layer type"):
        decode.decode_policy(replace(config, layer_types=layers), 1, for_afd=True)


@pytest.mark.parametrize("module", [
    "layers.58.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z", "layers.57.linear_attn.out_proj",
    "layers.59.self_attn.q_proj", "layers.3.self_attn.k_proj", "layers.3.self_attn.v_proj",
    "layers.55.self_attn.o_proj",
])
def test_every_fp8_projection_of_every_layer_is_checked(monkeypatch, block_fp8, decode, module):
    monkeypatch.setenv(SWITCH, "1")
    config = block_fp8(skip_extra=["model.language_model." + module])
    with pytest.raises(ValueError, match="requires block-FP8 model.layers"):
        decode.decode_policy(config, 1, for_afd=True)


@pytest.mark.parametrize("module", ["layers.0.linear_attn.in_proj_b", "layers.58.linear_attn.in_proj_a"])
def test_quant_and_a_must_stay_bf16(monkeypatch, block_fp8, decode, module):
    monkeypatch.setenv(SWITCH, "1")
    config = block_fp8(skip_drop=["model.language_model." + module])
    with pytest.raises(ValueError, match="unquantized BF16"):
        decode.decode_policy(config, 1, for_afd=True)


def test_the_switches_reach_the_ray_workers():
    from minisgl import ray_env_keys

    for key in (SWITCH, "MINISGL_QWEN35_MERGED_QKVZ", "MINISGL_QWEN35_FUSED_SHARED_GATE"):
        assert key in ray_env_keys._BOTH_HOPS


# --- D: fused shared-expert gate ----------------------------------------------------------

def _gate_inputs(rows: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    shared = (torch.randn((rows, 4096), generator=generator) * 0.05).to(torch.bfloat16)
    hidden = torch.randn((rows, 4096), generator=generator).to(torch.bfloat16)
    weight = (torch.randn((1, 4096), generator=generator) * 0.02).to(torch.bfloat16)
    return shared, hidden, weight


@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_shared_gate_switch_is_off_unless_exactly_1(monkeypatch, value):
    from minisgl.kernel import qwen35_shared_gate

    if value is not None:
        monkeypatch.setenv("MINISGL_QWEN35_FUSED_SHARED_GATE", value)
    assert not qwen35_shared_gate.enabled()
    monkeypatch.setenv("MINISGL_QWEN35_FUSED_SHARED_GATE", "1")
    assert qwen35_shared_gate.enabled()


@pytest.mark.parametrize("rows", [1, 3, 62])
def test_shared_gate_host_path_is_the_unfused_arithmetic(rows):
    from minisgl.kernel import qwen35_shared_gate

    shared, hidden, weight = _gate_inputs(rows, seed=rows)
    logit = torch.nn.functional.linear(hidden, weight)
    want = (shared.float() * torch.sigmoid(logit.float())).to(torch.bfloat16)
    got = qwen35_shared_gate.shared_gate(shared, hidden, weight)
    assert got.dtype == torch.bfloat16 and torch.equal(got, want)


@pytest.mark.parametrize("broken", ["gate_shape", "fp32_hidden", "shared_shape", "width", "column_stride"])
def test_shared_gate_refuses_other_layouts(broken):
    from minisgl.kernel import qwen35_shared_gate

    shared, hidden, weight = _gate_inputs(4, seed=9)
    if broken == "gate_shape":
        weight = weight.view(4096, 1)
    elif broken == "fp32_hidden":
        hidden = hidden.float()
    elif broken == "shared_shape":
        shared = shared[:3]
    elif broken == "width":
        shared, hidden, weight = shared[:, :2048], hidden[:, :2048], weight[:, :2048].contiguous()
    else:
        hidden = hidden.t().contiguous().t()
    with pytest.raises(RuntimeError, match="Fused shared-expert gate requires"):
        qwen35_shared_gate.shared_gate(shared, hidden, weight)


def test_route_uses_the_fused_gate_only_with_the_switch_in_decode():
    source = (Path(__file__).resolve().parents[3] / "python/minisgl/models/qwen35_moe_afd.py").read_text()
    assert ("if qwen35_shared_gate.enabled() and get_global_ctx().batch.is_decode:\n"
            "            # MINISGL_QWEN35_FUSED_SHARED_GATE=1:") in source
    assert ("shared = qwen35_shared_output(\n"
            "                shared_hidden, self.mlp.shared_expert_gate.forward(hidden_states))") in source


def test_merged_b_a_rebinds_row_views(monkeypatch):
    """merged_ba keeps one b|a allocation (row views, as merged_in_proj) instead of a copy."""
    import minisgl.core as core
    from minisgl.kernel import qwen35_decode as align

    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(batch=SimpleNamespace(is_decode=True)))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)  # CUDA is hidden
    generator = torch.Generator().manual_seed(4)
    b = SimpleNamespace(weight=torch.randn((64, 4096), generator=generator).to(torch.bfloat16), bias=None)
    a = SimpleNamespace(weight=torch.randn((64, 4096), generator=generator).to(torch.bfloat16), bias=None)
    originals = (b.weight.clone(), a.weight.clone())
    layer = SimpleNamespace(in_proj_b=b, in_proj_a=a)
    x = torch.randn((3, 4096), generator=generator).to(torch.bfloat16)
    got_b, got_a = align.merged_ba(layer, x)
    merged = layer._ba_merged_weight
    assert b.weight.data_ptr() == merged.data_ptr() and a.weight.data_ptr() == merged.data_ptr() + 64 * 4096 * 2
    assert torch.equal(b.weight, originals[0]) and torch.equal(a.weight, originals[1])
    output = torch.nn.functional.linear(x, torch.cat(originals, dim=0))
    assert torch.equal(got_b, output[:, :64]) and torch.equal(got_a, output[:, 64:])
