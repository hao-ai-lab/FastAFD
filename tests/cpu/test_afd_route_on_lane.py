"""MINISGL_AFD_ROUTE_ON_LANE: the AFD MB pipeline runs a rank-local router on the lane stream."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import minisgl
from minisgl import ray_env_keys


@pytest.fixture(scope="module")
def attention_worker():
    """afd_attention_worker imported with a stub Ray and engine module."""
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    engine = ModuleType("minisgl.engine.engine")
    engine._adjust_config = lambda config: None
    engine._init_tp_communication = lambda config, dtype: None
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        patch.setitem(sys.modules, "minisgl.engine.engine", engine)
        yield importlib.import_module("minisgl.afd_attention_worker")
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


def test_key_reaches_ray_workers():
    key = "MINISGL_AFD_ROUTE_ON_LANE"
    assert key in ray_env_keys.AFD_WORKER_ENV_KEYS and key in ray_env_keys.RAY_ACTOR_ENV_KEYS


@pytest.mark.parametrize("module,cls", [("minimax_m2_afd", "MiniMaxM2AfdDenseRouterForCausalLM"),
                                        ("qwen3_moe_afd", "Qwen3AfdDenseRouterForCausalLM")])
def test_models_declare_a_local_router(module, cls):
    # read from the source: the model modules import the Triton kernels
    tree = ast.parse((Path(minisgl.__file__).parent / "models" / f"{module}.py").read_text())
    body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    values = [ast.literal_eval(stmt.value) for stmt in body if isinstance(stmt, ast.Assign)
              and any(getattr(target, "id", None) == "afd_route_is_local" for target in stmt.targets)]
    assert values == [True]


class _Model:
    pass


@pytest.mark.parametrize("local,use_megamoe", [(False, True), (True, False)])
def test_unsupported_setups_are_refused(attention_worker, monkeypatch, local, use_megamoe):
    monkeypatch.setenv("MINISGL_AFD_ROUTE_ON_LANE", "1")
    model = _Model()
    if local:
        model.afd_route_is_local = True
    with pytest.raises(ValueError, match="MINISGL_AFD_ROUTE_ON_LANE"):
        attention_worker._run_ag_pipeline_body(
            state=SimpleNamespace(use_megamoe=use_megamoe), num_layers=1, model=model, adapters=[], ctx=None,
            engine_stream=None, phase="decode", retain_graph_objects=False, retire_async_objects=False)
