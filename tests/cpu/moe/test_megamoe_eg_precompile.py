"""Every MegaMoE EG kernel instance a step can select must be built before the first step.

DeepGEMM JIT-compiles an EG instance the first time a step selects it, on the EG host,
while the AG ranks have already launched their half of that step's M2N collective. The
adapter therefore builds, at startup, one instance per expected-token value the step
path can hand the kernel. The DeepGEMM extension is replaced by a recorder here: these
tests check which launches the adapter makes and with which instance inputs, not what
the C++ heuristic does with them (that is a pure function of the same inputs).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

import minisgl.core
from minisgl import ray_env_keys

BUCKET = 2048  # a prefill-sized per-rank dispatch bucket
NUM_LAYERS = 3


@pytest.fixture
def m2n_afd(monkeypatch):
    """Load moe/megamoe_m2n_afd.py alone, with its two Triton-importing deps stubbed.

    The stub kernel records every EG call and, for a compile-only call, returns a
    source string naming the instance inputs, as the real extension returns the
    generated kernel source.
    """
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

    def mega_moe_m2n_eg(weight_descs, l1_weight_ptrs, sym_buffer, **kwargs):
        compile_only = kwargs.pop("compile_only", False)
        kernel.calls.append((compile_only, _instance_inputs(kwargs)))
        return repr(_instance_inputs(kwargs)) if compile_only else ""

    kernel.mega_moe_m2n_eg = mega_moe_m2n_eg
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    path = Path(minisgl.core.__file__).resolve().parent / "moe" / "megamoe_m2n_afd.py"
    spec = importlib.util.spec_from_file_location("_fastafd_cpu_megamoe_m2n_afd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("MINISGL_MEGAMOE_LOG_LAUNCH", "MINISGL_MEGAMOE_CHECK_EG_OUTPUT",
                 "MINISGL_MEGAMOE_EG_PRECOMPILE"):
        monkeypatch.delenv(name, raising=False)
    return module, kernel


def _instance_inputs(kwargs):
    """The EG call arguments that reach the kernel's template (the JIT instance key).

    Layer descriptors, pointers, the layer count, the rank and the prefetch size are
    runtime arguments; the lane count is how many symmetric buffers are passed.
    """
    lanes = 1 + sum(kwargs.get(f"sym_buffer{i}") is not None for i in (1, 2, 3))
    return (
        kwargs["expected_num_tokens_per_rank"],
        lanes,
        kwargs["num_experts_per_rank"],
        kwargs["num_sms"],
        kwargs["fast_math"],
        kwargs.get("block_scale_residual_ptrs") is not None,
        kwargs.get("debug_timings") is not None,
        kwargs.get("p2p", False),
        kwargs.get("push_dispatch", False),
    )


def _eg_adapter(adapter_cls, *, num_lanes, override, exact_block_scales=False):
    """A registered EG adapter with only the fields the EG launchers read.

    The stacked descriptor and pointer caches are pre-filled on the CPU, since the
    adapter would otherwise build them on CUDA.
    """
    adapter = object.__new__(adapter_cls)
    adapter.is_ag = False
    adapter.is_eg = True
    adapter.rank = 4
    adapter.num_lanes = num_lanes
    adapter.buffers = [object() for _ in range(num_lanes)]
    adapter.bucket = BUCKET
    adapter.eg_expected_tokens_override = override
    adapter.num_experts = 128
    adapter.eg_size = 4
    adapter.eg_num_sms = 0
    adapter.eg_prefetch_bytes = 0
    adapter._eg_l1_nbytes = 1
    adapter._eg_global_to_compact = {layer: layer for layer in range(NUM_LAYERS)}
    adapter._eg_desc_blobs = {layer: None for layer in range(NUM_LAYERS)}
    adapter._desc_blob_all = torch.zeros((NUM_LAYERS, 4, 128), dtype=torch.uint8)
    adapter._l1_ptrs_all = torch.zeros(NUM_LAYERS, dtype=torch.int64)
    adapter.exact_block_scales = exact_block_scales
    adapter._block_scale_residual_ptrs_cache = torch.arange(NUM_LAYERS, dtype=torch.int64)
    return adapter


def _runtime_launches(adapter, num_mb):
    """Every launch the EG step path and the decode-graph capture can make for `num_mb`.

    Dispatch buckets run well past the adapter bucket and its decode limit, plus None.
    """
    buckets = [None, *range(1, 2 * BUCKET * adapter.num_lanes + 3)]
    for bucket in buckets:
        if adapter.use_persistent(num_mb):
            adapter.eg_moe_persistent(NUM_LAYERS, num_mb=num_mb, expected_tokens_per_rank=bucket)
            if adapter.use_dual_lane(num_mb):
                adapter.eg_moe_dual(1, expected_tokens_per_rank=bucket)
        else:
            for lane in range(num_mb):
                adapter.eg_moe(1, lane=lane, expected_tokens_per_rank=bucket)


CASES = [
    pytest.param(1, 1, 0, id="mb1"),
    pytest.param(2, 2, 0, id="mb2"),
    pytest.param(4, 4, 0, id="mb4"),
    pytest.param(1, 1, 300, id="mb1-pinned-300"),
    pytest.param(2, 2, 300, id="mb2-pinned-300"),
    pytest.param(2, 2, 16, id="mb2-remote-shared-16"),
]


@pytest.mark.parametrize(("num_lanes", "num_mb", "override"), CASES)
def test_buckets_reach_every_expected_token_value_once(m2n_afd, num_lanes, num_mb, override):
    module, _ = m2n_afd
    adapter = _eg_adapter(module.MegaMoEM2NAfdAdapter, num_lanes=num_lanes, override=override)
    dual = adapter.use_persistent(num_mb)

    buckets = adapter.eg_precompile_buckets(num_mb)

    values = [adapter._expected_tokens_for_eg(b, dual_lane_decode=dual) for b in buckets]
    reachable = {
        adapter._expected_tokens_for_eg(b, dual_lane_decode=dual)
        for b in [None, *range(1, 4 * BUCKET * num_lanes)]
    }
    assert set(values) == reachable
    assert len(values) == len(set(values)), "one bucket per value, no repeats"


@pytest.mark.parametrize("exact_block_scales", [False, True], ids=["requant", "exact"])
@pytest.mark.parametrize(("num_lanes", "num_mb", "override"), CASES)
def test_precompiled_instances_cover_every_runtime_launch(
        m2n_afd, num_lanes, num_mb, override, exact_block_scales):
    module, kernel = m2n_afd
    adapter = _eg_adapter(module.MegaMoEM2NAfdAdapter, num_lanes=num_lanes, override=override,
                          exact_block_scales=exact_block_scales)

    instances, buckets = adapter.precompile_eg_kernels(num_mb)
    precompile_calls = list(kernel.calls)
    kernel.calls.clear()
    _runtime_launches(adapter, num_mb)
    runtime_calls = list(kernel.calls)

    assert all(compile_only for compile_only, _ in precompile_calls), "precompile launched a kernel"
    assert not any(compile_only for compile_only, _ in runtime_calls)
    assert buckets == len(precompile_calls)
    built = {inputs for _, inputs in precompile_calls}
    assert instances == len(built)
    missing = {inputs for _, inputs in runtime_calls} - built
    assert not missing, f"runtime launches an EG instance never precompiled: {sorted(missing)[:3]}"


def test_precompile_refuses_before_weights_are_registered(m2n_afd):
    module, kernel = m2n_afd
    adapter = _eg_adapter(module.MegaMoEM2NAfdAdapter, num_lanes=2, override=0)
    adapter._eg_global_to_compact = {}

    with pytest.raises(RuntimeError, match="registered first"):
        adapter.precompile_eg_kernels(2)
    assert kernel.calls == []


@pytest.mark.parametrize(("value", "enabled"), [(None, True), ("1", True), ("0", False)])
def test_precompile_is_on_unless_disabled(m2n_afd, monkeypatch, value, enabled):
    module, _ = m2n_afd
    if value is not None:
        monkeypatch.setenv("MINISGL_MEGAMOE_EG_PRECOMPILE", value)
    assert module.eg_precompile_enabled() is enabled


def test_precompile_switch_rejects_other_values(m2n_afd, monkeypatch):
    module, _ = m2n_afd
    monkeypatch.setenv("MINISGL_MEGAMOE_EG_PRECOMPILE", "off")
    with pytest.raises(RuntimeError, match="must be 0"):
        module.eg_precompile_enabled()


def test_precompile_switch_reaches_the_eg_workers():
    assert "MINISGL_MEGAMOE_EG_PRECOMPILE" in ray_env_keys.RAY_ACTOR_ENV_KEYS
    assert "MINISGL_MEGAMOE_EG_PRECOMPILE" in ray_env_keys.AFD_WORKER_ENV_KEYS


@pytest.mark.parametrize("num_lanes", [1, 2])
def test_expected_token_hint_as_a_floor(m2n_afd, num_lanes):
    """MINISGL_MEGAMOE_EG_EXPECTED_FLOOR=1: the hint lifts small decode steps and leaves larger ones on their own bucket."""
    module, _ = m2n_afd
    adapter = _eg_adapter(module.MegaMoEM2NAfdAdapter, num_lanes=num_lanes, override=24)
    pinned = [adapter._expected_tokens_for_eg(b, dual_lane_decode=False) for b in (1, 10, 24, 161, BUCKET)]
    assert pinned == [24] * 5
    adapter.eg_expected_floor = True
    floor = [adapter._expected_tokens_for_eg(b, dual_lane_decode=False) for b in (1, 10, 24, 161, BUCKET)]
    assert floor == [24, 24, 24, 161, BUCKET]
    # precompile still covers every value a step can select
    dual = adapter.use_persistent(num_lanes)
    reachable = {adapter._expected_tokens_for_eg(b, dual_lane_decode=dual) for b in [None, *range(1, 4 * BUCKET * num_lanes)]}
    assert {adapter._expected_tokens_for_eg(b, dual_lane_decode=dual) for b in adapter.eg_precompile_buckets(num_lanes)} == reachable


def test_expected_floor_key_reaches_ray_workers():
    key = "MINISGL_MEGAMOE_EG_EXPECTED_FLOOR"
    assert key in ray_env_keys.AFD_WORKER_ENV_KEYS and key in ray_env_keys.RAY_ACTOR_ENV_KEYS
