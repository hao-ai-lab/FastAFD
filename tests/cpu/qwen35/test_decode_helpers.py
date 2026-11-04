"""Qwen3.5 production decode adapters: bounds, pool contracts and exact host math."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
from minisgl.kernel import qwen35_decode as alignment

def _decode_ctx(monkeypatch, decode: bool):
    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(batch=SimpleNamespace(is_decode=decode)))


def _norm_inputs(rows: int = 2):
    norm = SimpleNamespace(weight=torch.ones(4096, dtype=torch.bfloat16), eps=1e-6)
    tensors = [torch.zeros(rows, 4096, dtype=torch.bfloat16) for _ in range(3)]
    return norm, tensors


def test_fused_input_norm_quant_is_decode_only(monkeypatch):
    _decode_ctx(monkeypatch, decode=False)
    norm, (routed, shared, residual) = _norm_inputs()
    with pytest.raises(RuntimeError, match="decode-only"):
        alignment.fused_input_norm_quant(norm, routed, shared, residual)


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_fused_input_norm_quant_replaces_only_the_cuda_quant_backend(monkeypatch, backend):
    _decode_ctx(monkeypatch, decode=True)
    monkeypatch.setenv("MINISGL_FP8_QUANT_BACKEND", backend)
    norm, (routed, shared, residual) = _norm_inputs()
    with pytest.raises(RuntimeError, match="pinned CUDA quant backend"):
        alignment.fused_input_norm_quant(norm, routed, shared, residual)


def test_fused_input_norm_quant_rejects_host_or_misshaped_inputs(monkeypatch):
    _decode_ctx(monkeypatch, decode=True)
    norm, (routed, shared, residual) = _norm_inputs()
    with pytest.raises(RuntimeError, match="contiguous CUDA BF16"):
        alignment.fused_input_norm_quant(norm, routed, shared, residual)
    with pytest.raises(RuntimeError, match="contiguous CUDA BF16"):
        alignment.fused_input_norm_quant(norm, routed[:, :2048], shared, residual)


def _bf16_norm_inputs(rows: int = 2):
    norm = SimpleNamespace(weight=torch.ones(3072, dtype=torch.bfloat16), eps=1e-6)
    return norm, [torch.zeros(rows, 3072, dtype=torch.bfloat16) for _ in range(3)]


def _attention_layer():
    return SimpleNamespace(attn=SimpleNamespace(), num_qo_heads=32, head_dim=256, q_size=8192, kv_size=512)


def test_full_attention_views_are_decode_only(monkeypatch):
    _decode_ctx(monkeypatch, decode=False)
    qkv = torch.zeros(2, 2 * 8192 + 2 * 512, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="decode rows of the merged QKV output"):
        alignment.full_attention_decode(_attention_layer(), qkv, "fused_bf16_kv")


def test_full_attention_views_reject_other_layouts(monkeypatch):
    _decode_ctx(monkeypatch, decode=True)
    with pytest.raises(RuntimeError, match="decode rows of the merged QKV output"):
        alignment.full_attention_decode(_attention_layer(), torch.zeros(2, 8192), "fused_bf16_kv")
    with pytest.raises(ValueError, match="Unknown full-attention decode variant"):
        alignment.full_attention_decode(
            _attention_layer(), torch.zeros(2, 2 * 8192 + 2 * 512, dtype=torch.bfloat16), "copy"
        )


def _trtllm_backend(kvcache=None):
    from minisgl.attention.trtllm import TensorRTLLMBackend

    backend = TensorRTLLMBackend.__new__(TensorRTLLMBackend)  # the type is what is checked
    backend.kvcache = kvcache
    return backend


def _hybrid(prefill, decode):
    from minisgl.attention.base import HybridBackend

    return HybridBackend(prefill, decode)


@pytest.mark.parametrize("kind", ["other", "hybrid_with_other_decode"])
def test_full_attention_views_need_the_trtllm_decode_backend(monkeypatch, kind):
    backend = SimpleNamespace() if kind == "other" else _hybrid(_trtllm_backend(), SimpleNamespace())
    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(
        batch=SimpleNamespace(is_decode=True), attn_backend=backend))
    qkv = torch.zeros(2, 2 * 8192 + 2 * 512, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="need the TRT-LLM decode backend"):
        alignment.full_attention_decode(_attention_layer(), qkv, "fused_bf16_kv")


def test_bf16_kv_fused_full_attention_requires_the_bf16_pool(monkeypatch):
    kvcache = SimpleNamespace(dtype=torch.float8_e4m3fn)
    backend = _hybrid(SimpleNamespace(), _trtllm_backend(kvcache))
    backend.kvcache = kvcache
    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(
        batch=SimpleNamespace(is_decode=True), attn_backend=backend))
    layer = _attention_layer()
    layer.attn = SimpleNamespace(q_norm=SimpleNamespace(eps=1e-6), k_norm=SimpleNamespace(eps=1e-6))
    qkv = torch.zeros(2, 2 * 8192 + 2 * 512, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="BF16 KV pool"):
        alignment.full_attention_decode(layer, qkv, "fused_bf16_kv")
    kvcache.dtype = torch.bfloat16
    layer.attn.k_norm.eps = 1e-5
    with pytest.raises(RuntimeError, match="one q/k norm eps"):
        alignment.full_attention_decode(layer, qkv, "fused_bf16_kv")


def test_split_ba_returns_disjoint_views_of_one_allocation():
    output = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128).to(torch.bfloat16)
    b, a = alignment.split_ba(output)
    assert b.data_ptr() == output.data_ptr()
    assert torch.equal(b, output[:, :64]) and torch.equal(a, output[:, 64:])


@pytest.mark.parametrize(
    "output",
    [
        torch.zeros(3, 64, dtype=torch.bfloat16),
        torch.zeros(3, 128, dtype=torch.float32),
        torch.zeros(0, 128, dtype=torch.bfloat16),
        torch.zeros(128, 3, dtype=torch.bfloat16).t(),
    ],
    ids=["width", "dtype", "empty", "strided"],
)
def test_split_ba_rejects_non_canonical_outputs(output):
    with pytest.raises(ValueError, match="canonical BF16 N128"):
        alignment.split_ba(output)
