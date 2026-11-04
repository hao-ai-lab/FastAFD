"""MegaMoE M2N protocol switches on the host side: P2P, the AG wait split, EG early start.

The kernels are JIT-built on GPU hosts only, so these tests check what the adapter and
the kernel wrapper hand the kernels: which AG launches a layer makes and in what order
with which waits between them, the early-start launch arguments, the EG protocol
arguments and their layer-numbering guard, the EG prefetch default, and that the
wrapper's P2P control words sit where the kernel's layout puts them.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import minisgl.core

_PKG = Path(minisgl.core.__file__).resolve().parent
_LAYOUT = _PKG / "kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/layout/mega_moe_m2n.cuh"
_KERNEL = _PKG / "kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/sm100_mega_moe_m2n.cuh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def m2n_afd(monkeypatch):
    """moe/megamoe_m2n_afd.py alone, its Triton-importing deps stubbed; the stub
    kernel records the AG launches and the waits between them in order."""
    for name in ("minisgl.layers", "minisgl.layers.moe", "minisgl.layers.moe.moe_runner"):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    runner = ModuleType("minisgl.layers.moe.moe_runner.deepgemm_grouped")
    runner._requant_channel_weight_ue8m0_inplace = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    kernel = ModuleType("minisgl.kernel.megamoe_m2n_mega")
    kernel.calls = []
    kernel.mega_moe_m2n_ag = lambda y, x, ids, w, buf, **kw: kernel.calls.append(("ag", kw))
    kernel.stream_wait_ge = lambda flag, value: kernel.calls.append(("waitvalue", flag, value))
    kernel.wait_ge_cta = lambda flag, value, reset=False, tag=0: kernel.calls.append(
        ("waitcta", flag, value, reset))
    kernel.wait_ge_cta_mono = lambda flag, consumed, value, tag=0: kernel.calls.append(
        ("waitmono", flag, consumed, value))
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    module = _load(_PKG / "moe" / "megamoe_m2n_afd.py", "_fastafd_cpu_megamoe_m2n_protocol")
    for name in ("MINISGL_MEGAMOE_CHECK_INPUTS", "MINISGL_MEGAMOE_AG_TIMING",
                 "MINISGL_MEGAMOE_LOG_LAUNCH", "MINISGL_MEGAMOE_EG_PREFETCH_MB"):
        monkeypatch.delenv(name, raising=False)
    return module, kernel


class _Buffer:
    """The two P2P views ag_moe waits on, by layer parity."""

    def combine_count_view(self, parity):
        return ("count", parity)

    def combine_consumed_view(self, parity):
        return ("consumed", parity)


def _ag_adapter(adapter_cls, *, p2p=False, split="fused", waited=False, mono=False):
    adapter = object.__new__(adapter_cls)
    adapter.is_ag = True
    adapter.rank = 1
    adapter.num_lanes = 1
    adapter.buffers = [_Buffer()]
    adapter.bucket = 64
    adapter.hidden_size = 256
    adapter._y_cache = {}
    adapter._eager_buffers = {}
    adapter.routed_top_k = adapter.top_k = 4
    adapter.shared_experts_per_rank = 0
    adapter.routed_scaling_factor = 1.0
    adapter.real_num_experts = adapter.num_experts = 16
    adapter.gate_renormalize = True
    adapter.ag_num_sms, adapter.ag_dispatch_sms, adapter.ag_combine_sms = 24, 40, 16
    adapter.eg_size = 4
    adapter.p2p, adapter.ag_split = p2p, split
    adapter.ag_waited_combine, adapter.ag_wait_mono = waited, mono
    return adapter


def _layer(adapter, layer):
    tokens = 8
    adapter.ag_moe(torch.randn(tokens, 256, dtype=torch.bfloat16),
                   torch.zeros((tokens, 4), dtype=torch.int32),
                   torch.full((tokens, 4), 0.25), layer=layer)


def test_default_is_one_fused_barrier_launch(m2n_afd):
    module, kernel = m2n_afd
    _layer(_ag_adapter(module.MegaMoEM2NAfdAdapter), layer=3)
    ((kind, kw),) = kernel.calls
    assert kind == "ag" and kw["phase"] == 0 and kw["num_sms"] == 24
    assert not kw["p2p"] and not kw["push_dispatch"] and not kw["waited_combine"]


def test_p2p_needs_the_layer_and_passes_its_parity(m2n_afd):
    module, kernel = m2n_afd
    adapter = _ag_adapter(module.MegaMoEM2NAfdAdapter, p2p=True)
    with pytest.raises(RuntimeError, match="layer index"):
        _layer(adapter, layer=None)
    _layer(adapter, layer=5)
    _layer(adapter, layer=6)
    assert [kw["layer_parity"] for _, kw in kernel.calls] == [1, 0]
    assert all(kw["p2p"] for _, kw in kernel.calls)


@pytest.mark.parametrize("split,waited,mono,between,combine_waited", [
    ("spin", False, False, None, False),
    ("waitvalue", False, False, ("waitvalue", ("count", 1), 5), False),
    ("waitcta", False, False, ("waitcta", ("count", 1), 5, False), False),
    ("waitcta", True, False, ("waitcta", ("count", 1), 5, True), True),
    ("waitcta", False, True, ("waitmono", ("count", 1), ("consumed", 1), 5), True),
])
def test_split_launches_dispatch_wait_then_combine(m2n_afd, split, waited, mono, between, combine_waited):
    module, kernel = m2n_afd
    adapter = _ag_adapter(module.MegaMoEM2NAfdAdapter, p2p=True, split=split, waited=waited, mono=mono)
    _layer(adapter, layer=7)
    dispatch, *rest = kernel.calls
    assert dispatch[0] == "ag" and dispatch[1]["phase"] == 1 and dispatch[1]["num_sms"] == 40
    assert not dispatch[1]["waited_combine"]
    if between is not None:
        # the wait is on this rank's counter of the layer's parity, at eg_size + 1
        wait, rest = rest[0], rest[1:]
        assert wait == between
    ((kind, combine),) = rest
    assert kind == "ag" and combine["phase"] == 2 and combine["num_sms"] == 16
    assert combine["waited_combine"] == combine_waited


def test_eg_early_spec(m2n_afd):
    module, _ = m2n_afd
    parse = module.parse_eg_early
    assert parse("") == dict(module.EG_EARLY_OFF) and parse("")["early_max_late"] == 0
    fixed = parse("1:20")
    assert (fixed["early_max_late"], fixed["early_policy"], fixed["early_wait_cycles"]) == (1, 0, 38000)
    auto = parse("2:auto")
    assert (auto["early_max_late"], auto["early_policy"], auto["early_margin_pct"]) == (2, 1, 15)
    assert (auto["early_pass_cycles"], auto["early_expert_cycles"]) == (19000, 2280)
    tuned = parse("1:auto:30:5:2")
    assert (tuned["early_margin_pct"], tuned["early_pass_cycles"], tuned["early_expert_cycles"]) == (30, 9500, 3800)
    for bad in ("auto", "0:5", "x:5"):
        with pytest.raises(ValueError):
            parse(bad)


def _eg_adapter(adapter_cls, *, p2p, layers):
    adapter = object.__new__(adapter_cls)
    adapter.p2p = p2p
    adapter._eg_global_to_compact = layers
    return adapter


def test_eg_protocol_arguments_and_layer_numbering_guard(m2n_afd):
    module, _ = m2n_afd
    off = _eg_adapter(module.MegaMoEM2NAfdAdapter, p2p=False, layers={1: 0, 2: 1})
    kwargs = off._eg_protocol_kwargs(7)
    assert kwargs["p2p"] is False and kwargs["layer_base"] == 7 and kwargs["push_dispatch"] is False
    assert kwargs["early_max_late"] == 0
    # P2P indexes counters by layer parity: AG counts model layers, EG MoE layers
    whole = _eg_adapter(module.MegaMoEM2NAfdAdapter, p2p=True, layers={0: 0, 1: 1})
    assert whole._eg_protocol_kwargs(0)["p2p"]  # the persistent launch, from layer 0
    assert whole._eg_protocol_kwargs(1, 1)["layer_base"] == 1  # layer 1's own launch
    with pytest.raises(RuntimeError, match="every model layer"):
        whole._eg_protocol_kwargs(1)  # a launch from compact 0 numbered as model layer 1
    with pytest.raises(RuntimeError, match="every model layer"):
        _eg_adapter(module.MegaMoEM2NAfdAdapter, p2p=True, layers={1: 0, 2: 1})._eg_protocol_kwargs(0)


def test_p2p_control_words_match_the_kernel_layout(monkeypatch):
    """The wrapper seeds and reads P2P control words by offset from the buffer end;
    the kernel's layout::P2PRegionsM2N must put them there."""
    package = ModuleType("minisgl.kernel")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "minisgl.kernel", package)
    monkeypatch.setitem(sys.modules, "minisgl.kernel.deepgemm", ModuleType("minisgl.kernel.deepgemm"))
    mm = _load(_PKG / "kernel" / "megamoe_m2n_mega.py", "minisgl.kernel.megamoe_m2n_mega")
    layout = _LAYOUT.read_text()
    struct = layout[layout.index("struct P2PRegionsM2N"):]
    struct = struct[: struct.index("\n};")]
    ctrl_bytes = int(re.search(r"kNumCtrlBytes = (\d+);", struct).group(1))
    assert mm.P2P_CTRL_BYTES == ctrl_bytes
    words = {m.group(1): m.group(2) for m in re.finditer(
        r"int\* (get_\w+)\(.*?\) const \{\s*return static_cast<int\*>\(ctrl_base\)( \+ [^;]+)?;", struct, re.S)}
    assert words["get_credit_ptr"] is None
    assert words["get_combine_count_ptr"].strip() == "+ 1 + parity"

    buf = SimpleNamespace(num_bytes=4096, buffer=torch.zeros(4096, dtype=torch.int8))
    count = mm.MegaMoEM2NSymmBuffer.combine_count_view(buf, 1)
    consumed = mm.MegaMoEM2NSymmBuffer.combine_consumed_view(buf, 0)
    base = 4096 - ctrl_bytes
    assert count.data_ptr() - buf.buffer.data_ptr() == base + 4 * 2
    assert consumed.data_ptr() - buf.buffer.data_ptr() == base + 4 * 8
    # the consumed words are not one of the layout's other control words
    used = {0, 1, 2} | {int(w.split("+")[1]) for g, w in words.items() if w and g != "get_combine_count_ptr"}
    assert not {8, 9} & used


def test_eg_kernel_prints_only_through_the_hang_guard():
    """A printf in the EG kernel spills its warp roles' registers (nvcc 13.0: an
    EG instance went from 36/24 to 108/120 spill bytes with one in the
    pull loop), so it may only appear inside DG_HANG_GUARD, which compiles to
    nothing unless MINISGL_MEGAMOE_HANG_GUARD=1."""
    source = _KERNEL.read_text()
    eg = source[source.index("sm100_mega_moe_m2n_eg_impl("):]
    code = "\n".join(line.split("//", 1)[0] for line in eg.splitlines())
    assert "printf(" not in code
