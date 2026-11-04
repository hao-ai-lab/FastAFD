"""Shared fixtures for the AFD paging and prefix cache CPU tests."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

PAGE = 4


def _compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """minisgl.kernel.fast_compare_key (a JIT C++ extension the CPU lane cannot build):
    the length of the common prefix of two 1-D int tensors."""
    n = min(len(x), len(y))
    differ = (x[:n] != y[:n]).nonzero()
    return int(differ[0]) if len(differ) else n


@pytest.fixture(scope="module")
def afd():
    """The coordinator, its scheduler, the step plan builder and the attention runtime, imported with a stub Ray (``remote`` is the identity decorator) and a stub
    ``minisgl.engine.engine``: both are imported at module scope by the AFD files and
    pull in Ray and Triton, which the CPU lane does not install, and nothing under test
    calls into either. The radix key compare runs in Python. The modules are dropped
    again after the test module."""
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    engine = ModuleType("minisgl.engine.engine")
    engine._adjust_config = engine._init_tp_communication = None
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        patch.setitem(sys.modules, "minisgl.engine.engine", engine)
        patch.setattr(importlib.import_module("minisgl.kernel"), "fast_compare_key", _compare_key)
        scheduler = importlib.import_module("minisgl.afd_scheduler")
        yield SimpleNamespace(
            Scheduler=scheduler.CentralizedAfdDpScheduler,
            Coordinator=importlib.import_module("minisgl.afd_coordinator").AfdCoordinator,
            build_plan=importlib.import_module("minisgl.afd_protocol").build_afd_ag_plan,
            Runtime=importlib.import_module("minisgl.afd_attention_runtime").AfdAttentionRuntime,
            ChunkedReq=importlib.import_module("minisgl.scheduler.prefill").ChunkedReq,
        )
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


class CpuWorker:
    """The paging part of AfdAttentionRuntime on CPU: applies a step plan's page grants."""

    def __init__(self, afd, sched):
        rt = object.__new__(afd.Runtime)
        rt.config = SimpleNamespace(page_size=PAGE)
        rt.state = SimpleNamespace(page_table=torch.zeros_like(sched.page_table))
        rt.page_table = rt.state.page_table
        rt._free_pages = list(range(0, sched.cache_manager.num_pages * PAGE, PAGE))
        rt._table_cached_len, rt._table_pages, rt._table_reqs = {}, {}, {}
        rt._granted_tables = set()
        rt._req_input_ids = torch.empty(1, dtype=torch.int32)
        rt._page_to_token_tensor = lambda pages: (
            torch.tensor(pages, dtype=torch.int32).unsqueeze(1) + torch.arange(PAGE, dtype=torch.int32)
        ).flatten()
        self.rt = rt
        self.pending_free: tuple[int, ...] = ()

    def apply(self, plan):
        self.rt._free_tables(plan.free_table_indices)
        offset = 0
        for t, ext, cached, n in zip(plan.table_indices, plan.extend_lens, plan.cached_lens, plan.grant_counts):
            self.rt._make_stateful_req(t, ext, cached_len=cached, pages=plan.grant_pages[offset:offset + n])
            offset += n


@pytest.fixture
def cpu_worker(afd):
    return lambda sched: CpuWorker(afd, sched)
