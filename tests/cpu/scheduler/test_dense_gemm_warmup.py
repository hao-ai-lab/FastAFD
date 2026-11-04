"""MINISGL_DG_GEMM_WARMUP: the FP8 dense GEMM sweep finds every linear of a minisgl model."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

import minisgl


def _import_alone(name: str):
    """Import one minisgl module without running its package's __init__: the
    `minisgl.layers` packages import Triton kernels at package scope, which the CPU
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


linear = _import_alone("minisgl.layers.linear")
_base = _import_alone("minisgl.layers.base")
BaseOP, OPList = _base.BaseOP, _base.OPList


class _FakeFp8Linear(linear._LinearTPImpl):
    """A linear with an FP8-looking CUDA weight that records the token counts it ran."""

    def __init__(self, n: int, k: int) -> None:
        self.weight = SimpleNamespace(dtype=torch.float8_e4m3fn, is_cuda=True, shape=(n, k),
                                      device="cpu", dim=lambda: 2, numel=lambda: n * k)
        self._calls: list[int] = []

    def _forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        self._calls.append(x.shape[0])
        return x


class _Block(BaseOP):
    def __init__(self, qkv: BaseOP, o: BaseOP) -> None:
        self.qkv_proj = qkv
        self.o_proj = o


class _Model(BaseOP):
    def __init__(self, blocks: list[BaseOP], head: BaseOP) -> None:
        self.layers = OPList(blocks)
        self._lm_head = head  # private attributes are walked too


def test_warmup_walks_a_minisgl_model(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    qkv0, o0, qkv1, o1, head = (_FakeFp8Linear(*s) for s in
                                ((48, 16), (16, 32), (48, 16), (16, 32), (64, 16)))
    model = _Model([_Block(qkv0, o0), _Block(qkv1, o1)], head)
    assert not hasattr(model, "modules")

    shapes, calls = linear.warmup_fp8_dense_gemms(model, 20)

    # one module per distinct weight shape, each swept over M = 1..20
    assert shapes == 3
    assert calls == 3 * 20
    ran = [m for m in (qkv0, o0, qkv1, o1, head) if m._calls]
    assert len(ran) == 3 and all(m._calls == list(range(1, 21)) for m in ran)


def test_walk_visits_each_op_once() -> None:
    shared = _FakeFp8Linear(8, 8)
    model = _Model([_Block(shared, shared), _Block(shared, _FakeFp8Linear(8, 4))], shared)
    ops = list(linear._walk_ops(model))
    assert len(ops) == len({id(op) for op in ops})
    assert sum(isinstance(op, _FakeFp8Linear) for op in ops) == 2
