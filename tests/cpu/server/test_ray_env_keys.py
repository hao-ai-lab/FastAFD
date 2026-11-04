"""Both Ray hops forward env keys from one registry."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from minisgl import ray_env_keys
from minisgl.server import supervisor_ray

HOP_LISTS = {
    "RAY_ACTOR_ENV_KEYS": ray_env_keys.RAY_ACTOR_ENV_KEYS,
    "AFD_WORKER_ENV_KEYS": ray_env_keys.AFD_WORKER_ENV_KEYS,
    "RAY_ACTOR_NODE_SCOPED_CACHE_KEYS": ray_env_keys.RAY_ACTOR_NODE_SCOPED_CACHE_KEYS,
    "AFD_WORKER_NODE_SCOPED_CACHE_KEYS": ray_env_keys.AFD_WORKER_NODE_SCOPED_CACHE_KEYS,
}


@pytest.fixture
def afd_support(monkeypatch):
    """minisgl.afd_support, imported with a stub Ray (it imports Ray at module scope)."""
    before = set(sys.modules)
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        from minisgl import afd_support
    yield afd_support
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


@pytest.mark.parametrize("name", HOP_LISTS)
def test_each_list_names_a_key_once(name):
    keys = HOP_LISTS[name]
    assert len(set(keys)) == len(keys)


def test_both_hops_read_the_registry(afd_support):
    # One definition: the modules hold the registry's tuples, not copies of them.
    assert supervisor_ray._RAY_ENV_KEYS is ray_env_keys.RAY_ACTOR_ENV_KEYS
    assert supervisor_ray._NODE_SCOPED_CACHE_ENV_KEYS is ray_env_keys.RAY_ACTOR_NODE_SCOPED_CACHE_KEYS
    assert afd_support._RUNTIME_ENV_KEYS is ray_env_keys.AFD_WORKER_ENV_KEYS
    assert afd_support._NODE_SCOPED_CACHE_ENV_KEYS is ray_env_keys.AFD_WORKER_NODE_SCOPED_CACHE_KEYS


@pytest.mark.parametrize(
    ("cache_keys", "forwarded"),
    [
        (ray_env_keys.RAY_ACTOR_NODE_SCOPED_CACHE_KEYS, ray_env_keys.RAY_ACTOR_ENV_KEYS),
        (ray_env_keys.AFD_WORKER_NODE_SCOPED_CACHE_KEYS, ray_env_keys.AFD_WORKER_ENV_KEYS),
    ],
    ids=["ray_actor", "afd_worker"],
)
def test_node_scoped_cache_keys_are_forwarded_by_the_same_hop(cache_keys, forwarded):
    assert set(cache_keys) <= set(forwarded)


@pytest.mark.parametrize(
    "key",
    [
        "MINISGL_AFD_PAGE_FAST",
        "FLASHINFER_USE_CUDA_NORM",
    ],
)
def test_worker_read_switches_are_in_both_hops(key):
    # A key only in the driver hop reaches AFD workers solely through Ray's
    # runtime_env inheritance from the coordinator.
    assert key in ray_env_keys.RAY_ACTOR_ENV_KEYS
    assert key in ray_env_keys.AFD_WORKER_ENV_KEYS


def test_the_worker_env_carries_a_registered_key(monkeypatch, afd_support):
    monkeypatch.setenv("MINISGL_AFD_PAGE_FAST", "1")
    monkeypatch.setenv("FASTAFD_TEST_NOT_A_RAY_KEY", "1")
    env_vars = afd_support.runtime_env()["env_vars"]
    assert env_vars["MINISGL_AFD_PAGE_FAST"] == "1"
    assert "FASTAFD_TEST_NOT_A_RAY_KEY" not in env_vars
    assert supervisor_ray._build_ray_runtime_env()["env_vars"]["MINISGL_AFD_PAGE_FAST"] == "1"


def test_unset_or_empty_keys_are_not_forwarded(monkeypatch, afd_support):
    monkeypatch.setenv("MINISGL_AFD_PAGE_FAST", "")
    assert "MINISGL_AFD_PAGE_FAST" not in afd_support.runtime_env()["env_vars"]
    monkeypatch.delenv("MINISGL_AFD_PAGE_FAST")
    assert "MINISGL_AFD_PAGE_FAST" not in afd_support.runtime_env()["env_vars"]
