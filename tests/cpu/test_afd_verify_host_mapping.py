"""Speculative verify steps: the vectorized host mapping equals the per-request loop."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

DUMMY = 9


@pytest.fixture
def runtime_cls(monkeypatch):
    """AfdAttentionRuntime, imported with a stub Ray (afd_support imports it at module scope)."""
    before = set(sys.modules)
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        from minisgl.afd_attention_runtime import AfdAttentionRuntime
    yield AfdAttentionRuntime
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


def _runtime(cls):
    runtime = cls.__new__(cls)
    runtime._gdn_sentinel = False
    runtime._dummy_req = SimpleNamespace(table_idx=DUMMY)
    runtime._verify_maps_cache = {}
    runtime._claim_input_mapping_stage = lambda size: (
        0, torch.full((size,), -99, dtype=torch.int32), torch.full((size,), -99, dtype=torch.int32)
    )
    table = torch.arange(16 * 64, dtype=torch.int32).reshape(16, 64)
    runtime.state = SimpleNamespace(device=torch.device("cpu"), page_table=table)
    runtime.page_table = table  # the runtime's default: the state's own table
    return runtime


def _req(table_idx: int, cached: int, length: int):
    return SimpleNamespace(table_idx=table_idx, cached_len=cached, device_len=cached + length)


def _loop_reference(reqs):
    positions, tables = [], []
    for req in reqs:
        positions += list(range(req.cached_len, req.device_len))
        tables += [req.table_idx] * (req.device_len - req.cached_len)
    return positions, tables


# Block 3: real requests and dummy verify slots, every slot extends by three tokens.
VERIFY = [_req(0, 5, 3), _req(4, 11, 3), _req(DUMMY, 0, 3), _req(2, 7, 3), _req(DUMMY, 0, 3)]


def test_uniform_blocks_match_the_loop(runtime_cls):
    runtime = _runtime(runtime_cls)
    _, positions, tables = runtime._build_attention_input_mapping(VERIFY)
    assert runtime._mapping_uniform_len == 3
    assert (positions.tolist(), tables.tolist()) == _loop_reference(VERIFY)


def test_uneven_extends_keep_the_loop(runtime_cls):
    reqs = [_req(3, 0, 4), _req(0, 2, 2), _req(1, 5, 3)]
    runtime = _runtime(runtime_cls)
    _, positions, tables = runtime._build_attention_input_mapping(reqs)
    assert runtime._mapping_uniform_len == 0
    assert (positions.tolist(), tables.tolist()) == _loop_reference(reqs)


@pytest.mark.parametrize("verify", [True, False])
def test_verify_gets_one_table_slot_per_request(runtime_cls, verify):
    runtime = _runtime(runtime_cls)
    batch = SimpleNamespace(padded_reqs=VERIFY, afd_req_table_indices_gpu=None)
    runtime._attach_input_mapping(batch, SimpleNamespace(phase="prefill", spec_verify=verify))
    per_req = batch.afd_req_table_indices_gpu
    if verify:
        assert per_req.tolist() == [req.table_idx for req in VERIFY]
    else:
        assert per_req is None
