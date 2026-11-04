"""MINISGL_MEGAMOE_EXACT_FP8_SCALES: FP8 experts keep their 128x128 block scales.

Without it, the EG adapter requantizes each layer's FP8 experts to UE8M0 block scales,
which rounds every weight a second time. With it, the weights stay as loaded and the
adapter keeps each layer's block scale residuals ([E_local, L1 blocks + L2 blocks]) for
the kernel, which applies them per K block. The kernel module is replaced by stubs that
record what the adapter hands it; the kernel's own arithmetic is tested on a GPU
(tests/kernel/test_m2n_single_rank.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import minisgl.core
from minisgl import ray_env_keys

E, INTER, HIDDEN = 2, 256, 384  # L1: 4 x 3 scale blocks per expert, L2: 3 x 2


@pytest.fixture
def m2n_afd(monkeypatch):
    """moe/megamoe_m2n_afd.py alone, with its Triton-importing dependencies stubbed."""
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

    def fp8_weights_with_block_scale_residuals(w_fp8, block_scales):
        kernel.calls.append(("exact", w_fp8))
        return w_fp8, ("ue8m0 sf", block_scales), block_scales * 0.75

    def requant_qwen_fp8_weights_per32(w_fp8, block_scales, *, inplace=False):
        kernel.calls.append(("requant", w_fp8))
        return w_fp8, ("ue8m0 sf", block_scales)

    kernel.fp8_weights_with_block_scale_residuals = fp8_weights_with_block_scale_residuals
    kernel.requant_qwen_fp8_weights_per32 = requant_qwen_fp8_weights_per32
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    path = Path(minisgl.core.__file__).resolve().parent / "moe" / "megamoe_m2n_afd.py"
    spec = importlib.util.spec_from_file_location("_fastafd_cpu_megamoe_exact_scales", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, kernel


def _adapter(adapter_cls, *, exact):
    """An EG adapter with the fields weight registration reads; storing is recorded."""
    adapter = object.__new__(adapter_cls)
    adapter.is_eg = True
    adapter.exact_block_scales = exact
    adapter.shapes_are_padded = False
    adapter.shared_experts_per_rank = 0
    adapter._eg_block_scale_residuals = {}
    adapter._block_scale_residual_ptrs_cache = torch.zeros(1, dtype=torch.int64)  # stale
    adapter.stored = []
    adapter._store_eg_layer_weights = lambda layer_id, l1, l2: adapter.stored.append(
        (layer_id, l1, l2)
    )
    return adapter


def _experts():
    blocks = lambda n, k: torch.rand(E, n // 128, k // 128) + 0.1  # noqa: E731
    return SimpleNamespace(
        quant=SimpleNamespace(method="fp8", weight_block_size=(128, 128)),
        gate_up_proj=torch.zeros(E, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn),
        gate_up_proj_scale=blocks(2 * INTER, HIDDEN),
        down_proj=torch.zeros(E, HIDDEN, INTER, dtype=torch.float8_e4m3fn),
        down_proj_scale=blocks(HIDDEN, INTER),
    )


def test_exact_registration_keeps_the_weights_and_their_residuals(m2n_afd):
    module, kernel = m2n_afd
    adapter = _adapter(module.MegaMoEM2NAfdAdapter, exact=True)
    experts = _experts()

    adapter.register_eg_layer_weights(7, experts)

    assert [kind for kind, _ in kernel.calls] == ["exact", "exact"]
    ((layer_id, l1, l2),) = adapter.stored
    assert layer_id == 7
    assert l1[0] is experts.gate_up_proj and l2[0] is experts.down_proj
    residuals = adapter._eg_block_scale_residuals[7]
    # per expert: L1's (4 x 3) then L2's (3 x 2) residuals, row-block major
    assert residuals.shape == (E, 4 * 3 + 3 * 2)
    torch.testing.assert_close(residuals[:, :12], experts.gate_up_proj_scale.flatten(1) * 0.75)
    torch.testing.assert_close(residuals[:, 12:], experts.down_proj_scale.flatten(1) * 0.75)
    assert adapter._block_scale_residual_ptrs_cache is None, "the pointer table must be rebuilt"


def test_without_the_flag_the_experts_are_requantized(m2n_afd):
    module, kernel = m2n_afd
    adapter = _adapter(module.MegaMoEM2NAfdAdapter, exact=False)

    adapter.register_eg_layer_weights(7, _experts())

    assert [kind for kind, _ in kernel.calls] == ["requant", "requant"]
    assert adapter._eg_block_scale_residuals == {}
    assert adapter._block_scale_residual_ptrs() is None
    assert adapter._block_scale_residual_ptrs_for_layer(0) is None


def test_a_layer_without_residuals_is_refused(m2n_afd):
    module, _ = m2n_afd
    adapter = _adapter(module.MegaMoEM2NAfdAdapter, exact=True)
    adapter._block_scale_residual_ptrs_cache = None
    adapter._eg_global_to_compact = {5: 0, 9: 1}
    adapter._eg_block_scale_residuals = {5: torch.zeros(E, 18)}

    with pytest.raises(RuntimeError, match=r"layers \[9\]"):
        adapter._block_scale_residual_ptrs()


def test_a_single_layer_launch_gets_its_layers_pointer(m2n_afd):
    module, _ = m2n_afd
    adapter = _adapter(module.MegaMoEM2NAfdAdapter, exact=True)
    adapter._block_scale_residual_ptrs_cache = torch.tensor([11, 22, 33])

    assert adapter._block_scale_residual_ptrs_for_layer(1).tolist() == [22]
    assert adapter._block_scale_residual_ptrs().tolist() == [11, 22, 33]


def test_the_flag_reaches_the_workers():
    assert "MINISGL_MEGAMOE_EXACT_FP8_SCALES" in ray_env_keys.RAY_ACTOR_ENV_KEYS
    assert "MINISGL_MEGAMOE_EXACT_FP8_SCALES" in ray_env_keys.AFD_WORKER_ENV_KEYS
