"""Cross-head Q/K RMSNorm weights under tensor parallelism (MiniMax M2 / M2.5).

MiniMax M2's q_norm / k_norm normalize the whole (heads x head_dim) vector, so the
weight holds one value per head element and must follow the TP head split (its
module, RMSNormCrossHead, allocates full_dim // tp_size and all-reduces the
variance). A per-head-shared norm (Qwen3: head_dim long) stays whole.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

import minisgl


def _import_alone(name: str):
    """Import one minisgl module without running its package's __init__: the
    `minisgl.models` and `minisgl.layers` packages import Triton kernels at package
    scope, which the CPU lane does not install."""
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


BaseOP = _import_alone("minisgl.layers.base").BaseOP
_shard_tensor = _import_alone("minisgl.models.weight")._shard_tensor

HEAD_DIM = 128


def _norm(heads: int) -> torch.Tensor:
    return torch.arange(heads * HEAD_DIM, dtype=torch.float32)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_q_norm_follows_the_head_split(n: int) -> None:
    full = _norm(48)  # MiniMax M2.5: 48 query heads
    shards = [_shard_tensor("model.layers.0.self_attn.q_norm.weight", full, r, n, 8) for r in range(n)]
    assert all(s.numel() == full.numel() // n for s in shards)  # RMSNormCrossHead's size
    assert torch.equal(torch.cat(shards), full)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_k_norm_follows_the_kv_head_split(n: int) -> None:
    full = _norm(8)  # 8 KV heads, n <= 8: each rank owns 8 / n whole heads
    shards = [_shard_tensor("model.layers.3.self_attn.k_norm.weight", full, r, n, 8) for r in range(n)]
    assert all(s.numel() == full.numel() // n for s in shards)
    assert torch.equal(torch.cat(shards), full)


def test_k_norm_with_fewer_kv_heads_than_ranks_takes_its_head() -> None:
    # 8 KV heads over 16 ranks: rank r keeps KV head r * 8 // 16 (the head its
    # attention shard reads), as the KV head replication does
    full = _norm(8)
    for r in range(16):
        got = _shard_tensor("model.layers.0.self_attn.k_norm.weight", full, r, 16, 8)
        head = r * 8 // 16
        assert torch.equal(got, full[head * HEAD_DIM : (head + 1) * HEAD_DIM])


def test_per_head_shared_norm_stays_whole() -> None:
    full = torch.arange(HEAD_DIM, dtype=torch.float32)  # Qwen3 q_norm / k_norm
    for key in ("model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"):
        assert torch.equal(_shard_tensor(key, full, 1, 4, 4), full)


def test_tp1_keeps_the_full_weight() -> None:
    full = _norm(48)
    assert torch.equal(_shard_tensor("model.layers.0.self_attn.q_norm.weight", full, 0, 1, 8), full)


class _OneParam(BaseOP):
    def __init__(self) -> None:
        self.weight = torch.empty(4)


def test_load_state_dict_mismatch_names_the_parameter() -> None:
    op = _OneParam()
    with pytest.raises(AssertionError, match=r"weight: expected \(4,\).*loaded \(8,\)"):
        op.load_state_dict({"weight": torch.empty(8)})
