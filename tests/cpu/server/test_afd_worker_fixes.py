"""AFD worker plumbing: the distributed timeout, GPU counting per host, Triton's cache."""

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

from minisgl import ray_env_keys
import minisgl.server.args as server_args


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


@pytest.mark.parametrize("order", [0, 1])
def test_gpu_count_takes_the_raylet_with_gpus(afd_support, monkeypatch, order) -> None:
    # a host running a GPU-less raylet next to its GPU one, under the same IP
    raylets = [
        {"NodeManagerAddress": "10.0.0.5", "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeManagerAddress": "10.0.0.5", "Alive": True, "Resources": {"CPU": 64.0, "GPU": 4.0}},
        {"NodeManagerAddress": "10.0.0.6", "Alive": True, "Resources": {"GPU": 4.0}},
    ]
    if order:
        raylets[0], raylets[1] = raylets[1], raylets[0]
    monkeypatch.setattr(afd_support, "_live_nodes", lambda: raylets)
    assert afd_support._node_gpu_count("10.0.0.5") == 4


def test_triton_cache_reaches_the_workers_scoped_per_node(afd_support) -> None:
    for keys in (ray_env_keys.RAY_ACTOR_ENV_KEYS, ray_env_keys.AFD_WORKER_ENV_KEYS):
        assert "TRITON_CACHE_DIR" in keys
    for keys in (ray_env_keys.RAY_ACTOR_NODE_SCOPED_CACHE_KEYS,
                 ray_env_keys.AFD_WORKER_NODE_SCOPED_CACHE_KEYS):
        assert "TRITON_CACHE_DIR" in keys
    env = {"TRITON_CACHE_DIR": "/shared/triton"}
    afd_support._scope_cache_env(env, "10.0.0.5", rank_tag="rank3")
    assert env["TRITON_CACHE_DIR"] == "/shared/triton/ray-node-10_0_0_5/rank3"


def test_afd_serve_forwards_the_distributed_timeout(monkeypatch) -> None:
    monkeypatch.delenv("MINISGL_AFD_DIST_TIMEOUT", raising=False)
    monkeypatch.setattr(server_args, "_detect_local_ip", lambda: "127.0.0.1")
    server_args.parse_args([
        "--mode", "afd-serve", "--model", "/unused-checkpoint", "--dtype", "bfloat16",
        "--cache-type", "naive", "--attention-backend", "trtllm",
        "--afd-moe-runner-backend", "deep_gemm", "--cuda-graph-max-bs", "8",
        "--afd-attn-dp-size", "4", "--afd-attn-tp-size", "1", "--afd-mlp-dp-size", "4",
        "--afd-mlp-tp-size", "1", "--afd-mlp-ep-size", "4", "--distributed-timeout", "900",
    ])
    assert os.environ["MINISGL_AFD_DIST_TIMEOUT"] == "900.0"
    assert "MINISGL_AFD_DIST_TIMEOUT" in ray_env_keys.AFD_WORKER_ENV_KEYS


@pytest.fixture
def worker_base(monkeypatch):
    """minisgl.afd_worker_base with a stub Ray, and a stub minisgl.layers.linear whose
    dense GEMM warmup records its calls (minisgl.layers imports Triton kernels)."""
    before = set(sys.modules)
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    layers = ModuleType("minisgl.layers")
    layers.__path__ = []
    linear = ModuleType("minisgl.layers.linear")
    linear.calls = []
    linear.warmup_fp8_dense_gemms = lambda model, max_tokens: (linear.calls.append((model, max_tokens)), (2, 9))[1]
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        patch.setitem(sys.modules, "minisgl.layers", layers)
        patch.setitem(sys.modules, "minisgl.layers.linear", linear)
        from minisgl import afd_worker_base

        yield afd_worker_base, linear
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


def _worker(module, model, tmp_path):
    from types import SimpleNamespace

    worker = object.__new__(module.BaseAfdWorker)
    worker._afd_ag_model = model
    worker.runtime = SimpleNamespace(config=SimpleNamespace(max_extend_tokens=512))
    worker.runtime_state = SimpleNamespace(stream=None)
    worker.log_path, worker.role, worker.tp_rank = str(tmp_path / "worker.log"), "attention", 0
    return worker


@pytest.mark.parametrize("mode, warms", [(None, True), ("auto", True), ("1", True), ("0", False)])
def test_attention_workers_warm_up_dense_gemms_unless_disabled(worker_base, monkeypatch, tmp_path, mode, warms):
    module, linear = worker_base
    if mode is None:
        monkeypatch.delenv("MINISGL_DG_GEMM_WARMUP", raising=False)
    else:
        monkeypatch.setenv("MINISGL_DG_GEMM_WARMUP", mode)
    model = object()
    _worker(module, model, tmp_path)._warmup_ag_dense_gemms()
    # the sweep covers every prefill size up to the step's token budget
    assert linear.calls == ([(model, 512)] if warms else [])


def test_expert_workers_have_no_dense_gemms_to_warm_up(worker_base, monkeypatch, tmp_path):
    module, linear = worker_base
    monkeypatch.setenv("MINISGL_DG_GEMM_WARMUP", "1")
    _worker(module, None, tmp_path)._warmup_ag_dense_gemms()
    assert linear.calls == []
