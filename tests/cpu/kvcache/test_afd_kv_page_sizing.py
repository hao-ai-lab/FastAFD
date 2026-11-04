"""AFD attention-side KV sizing prices a page as the pool allocates it."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import minisgl.kvcache.mha_pool as mha_pool

GIB = 1024**3

# Qwen3.5-397B-A17B: 60 layers, every fourth full attention, 2 KV heads of 256.
QWEN35_KV_LAYERS = tuple(range(3, 60, 4))


@pytest.fixture
def state_cls(monkeypatch):
    """AfdAttentionState, imported with stub Ray and engine modules (no CUDA/Triton)."""
    before = set(sys.modules)
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    engine = ModuleType("minisgl.engine.engine")
    engine._adjust_config = lambda config: None
    engine._init_tp_communication = lambda config, dtype: None
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        patch.setitem(sys.modules, "minisgl.engine.engine", engine)
        from minisgl.afd_attention_worker import AfdAttentionState
    yield AfdAttentionState
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


def _model_config(*, kv_layers=QWEN35_KV_LAYERS, head_dim=256, kv_heads=2):
    return SimpleNamespace(
        head_dim=head_dim,
        num_kv_heads=kv_heads,
        kv_cache_layer_ids=tuple(kv_layers),
    )


def _num_pages(
    state_cls,
    monkeypatch,
    *,
    model_config,
    old_free: int,
    state_bytes: int,
    num_page_override: int | None = None,
) -> int:
    monkeypatch.delenv("MINISGL_AFD_SPEC", raising=False)
    state = state_cls.__new__(state_cls)
    state.dtype = torch.bfloat16
    new_free = old_free - state_bytes
    state._sync_get_memory = lambda: (new_free, new_free)
    config = SimpleNamespace(
        model_config=model_config,
        tp_info=SimpleNamespace(size=1),
        page_size=16,
        memory_ratio=0.78,
        num_page_override=num_page_override,
        max_running_req=256,
        max_seq_len=8192,
    )
    return state._determine_num_pages(old_free, config)


def _budget(old_free: int, state_bytes: int) -> int:
    return int(0.78 * old_free) - state_bytes


def _pool_bytes_per_page(monkeypatch, model_config) -> int:
    """What the real pool allocates per page, measured on a one-page CPU pool."""
    monkeypatch.setattr(mha_pool, "get_tp_info", lambda: SimpleNamespace(size=1))
    pool = mha_pool.MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_layers=max(model_config.kv_cache_layer_ids) + 1,
        head_dim=model_config.head_dim,
        num_pages=1,
        page_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layer_ids=model_config.kv_cache_layer_ids,
    )
    return pool._kv_buffer.numel() * pool._kv_buffer.element_size()


@pytest.mark.parametrize("old_free_gib", [40, 120, 183])
@pytest.mark.parametrize("kv_layers", [QWEN35_KV_LAYERS, tuple(range(62))])
def test_sized_pool_fills_the_budget_without_exceeding_it(state_cls, monkeypatch, old_free_gib, kv_layers):
    model_config = _model_config(kv_layers=kv_layers)
    old_free, state_bytes = old_free_gib * GIB, 8 * GIB
    pages = _num_pages(
        state_cls, monkeypatch, model_config=model_config,
        old_free=old_free, state_bytes=state_bytes,
    )
    per_page = _pool_bytes_per_page(monkeypatch, model_config)
    budget = _budget(old_free, state_bytes)

    assert pages * per_page <= budget < (pages + 1) * per_page


def test_num_page_override_is_honoured_below_the_cap(state_cls, monkeypatch):
    pages = _num_pages(
        state_cls, monkeypatch, model_config=_model_config(),
        old_free=120 * GIB, state_bytes=8 * GIB, num_page_override=1000,
    )

    assert pages == 1000
