"""Load the Qwen3.5 kernel modules in the CPU lane, which has no Triton.

The modules need Triton only for their ``@triton.jit`` decorators (and ``triton.cdiv`` /
``next_power_of_2`` inside launch wrappers the CPU lane never calls); a stand-in scoped to each
import provides those and is removed from ``sys.modules`` again, as in tests/cpu/qwen35/conftest.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

import minisgl.distributed.info as dist_info

_KERNEL_DIR = Path(dist_info.__file__).resolve().parents[1] / "kernel"


def load_with_triton_stub(path: Path, name: str) -> ModuleType:
    stub, language = ModuleType("triton"), ModuleType("triton.language")
    stub.jit = lambda fn: fn
    stub.cdiv = lambda a, b: -(-a // b)
    stub.next_power_of_2 = lambda n: 1 << max(n - 1, 0).bit_length()
    language.constexpr = object
    stub.language = language
    saved = {key: sys.modules.get(key) for key in ("triton", "triton.language")}
    sys.modules.update({"triton": stub, "triton.language": language})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


@pytest.fixture(scope="module")
def router():
    return load_with_triton_stub(_KERNEL_DIR / "qwen35_router_topk.py", "_qwen35_router_topk_cpu_lane")


@pytest.fixture(scope="module")
def gated_norm():
    return load_with_triton_stub(_KERNEL_DIR / "qwen35_gated_norm.py", "_qwen35_gated_norm_cpu_lane")


@pytest.fixture(scope="module")
def gdn():
    """kernel/qwen35_gdn.py (its eager ``_qwen35_rmsnorm_gated`` is the current CPU path)."""
    # qwen35_gdn calls torch.compile at import; build one wrapper first so torch's own
    # optional-Triton probes run against the real environment, not the stand-in.
    torch.compile(lambda t: t, dynamic=True, fullgraph=True)
    return load_with_triton_stub(_KERNEL_DIR / "qwen35_gdn.py", "_qwen35_gdn_cpu_lane")
