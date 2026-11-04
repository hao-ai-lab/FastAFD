"""MINISGL_AFD_DRAFT_ATTN_SMS: the SM count the EAGLE3 draft's attention launch is sized by."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

import minisgl
from minisgl import ray_env_keys


@pytest.fixture(scope="module")
def eagle3():
    """minisgl.models.eagle3 alone: a path-only package stands in for minisgl.models, whose __init__
    imports every model (and the GPU kernels with them)."""
    pkg = ModuleType("minisgl.models")
    pkg.__path__ = [str(Path(minisgl.__file__).parent / "models")]
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "minisgl.models", pkg)
        yield importlib.import_module("minisgl.models.eagle3")
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.models"):
            del sys.modules[name]


def test_key_reaches_ray_workers():
    key = "MINISGL_AFD_DRAFT_ATTN_SMS"
    assert key in ray_env_keys.AFD_WORKER_ENV_KEYS and key in ray_env_keys.RAY_ACTOR_ENV_KEYS


@pytest.mark.parametrize("value,expected", [(None, 0), ("", 0), ("0", 0), ("456", 456)])
def test_parse(eagle3, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MINISGL_AFD_DRAFT_ATTN_SMS", raising=False)
    else:
        monkeypatch.setenv("MINISGL_AFD_DRAFT_ATTN_SMS", value)
    assert eagle3._draft_attn_sms() == expected


def test_negative_is_rejected(eagle3, monkeypatch):
    monkeypatch.setenv("MINISGL_AFD_DRAFT_ATTN_SMS", "-1")
    with pytest.raises(ValueError):
        eagle3._draft_attn_sms()


def test_auto(eagle3, monkeypatch):
    monkeypatch.setenv("MINISGL_AFD_DRAFT_ATTN_SMS", "auto")
    auto = eagle3._draft_attn_sms()
    assert auto == eagle3.DRAFT_ATTN_AUTO
    pick = eagle3.draft_attn_launch_sms
    assert pick(auto, 20 * 8, 152) == 608        # past one wave: split
    assert pick(auto, 10 * 8, 152) == 608        # half the SMs idle: split
    assert pick(auto, 17 * 8, 152) == 0          # a near-full single wave: FlashInfer's own
    assert pick(456, 17 * 8, 152) == 456 and pick(0, 160, 152) == 0
