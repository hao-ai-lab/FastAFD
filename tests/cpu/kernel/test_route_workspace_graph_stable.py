"""Router workspaces must outlive a CUDA graph that captured them.

A larger call replaces the cached workspace; the replaced tensor is kept alive,
because a graph captured against it still writes through its raw pointer.
"""

from __future__ import annotations

import torch

from minisgl.kernel import moe_topk


def test_minimax_route_logits_workspace_keeps_the_replaced_buffer(monkeypatch):
    monkeypatch.setattr(moe_topk, "_MINIMAX_ROUTE_LOGITS_WS", {})
    monkeypatch.setattr(moe_topk, "_RETIRED_WS", [])
    cpu = torch.device("cpu")
    small = moe_topk._minimax_route_logits_workspace(cpu, 16, 256)
    assert small.shape == (16, 256)
    assert moe_topk._minimax_route_logits_workspace(cpu, 512, 256).data_ptr() == small.data_ptr()
    big = moe_topk._minimax_route_logits_workspace(cpu, 4096, 256)
    assert big.shape == (4096, 256)
    assert big.data_ptr() != small.data_ptr()
    assert any(t.data_ptr() == small.data_ptr() for t in moe_topk._RETIRED_WS)


def test_gate_topk_workspace_keeps_the_replaced_buffers(monkeypatch):
    monkeypatch.setattr(moe_topk, "_GATE_TOPK_WS", {})
    monkeypatch.setattr(moe_topk, "_RETIRED_WS", [])
    cpu = torch.device("cpu")
    partials, counters = moe_topk._gate_topk_workspace(cpu, 4)
    assert moe_topk._gate_topk_workspace(cpu, 32)[0].data_ptr() == partials.data_ptr()
    grown, _ = moe_topk._gate_topk_workspace(cpu, 64)
    assert grown.data_ptr() != partials.data_ptr()
    assert [t[0].data_ptr() for t in moe_topk._RETIRED_WS] == [partials.data_ptr()]
