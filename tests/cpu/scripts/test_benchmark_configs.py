"""The benchmark configurations and their measurement script (scripts/benchmark), without GPUs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "scripts" / "benchmark"


def _load(name: str):
    sys.path.insert(0, str(BENCH))
    try:
        spec = importlib.util.spec_from_file_location(f"_benchmark_{name}", BENCH / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCH))


CONFIGS = json.loads((BENCH / "configs.json").read_text())["configs"]
MODELS = json.loads((BENCH / "models.json").read_text())
POOLS = json.loads((BENCH / "prompt_pools.json").read_text())


def test_configs_fit_eight_nodes_and_reference_shipped_inputs():
    ids = [c["id"] for c in CONFIGS]
    assert len(ids) == len(set(ids))
    for c in CONFIGS:
        assert c["nodes"] == c["ratio"] + 1 <= 8, c["id"]
        assert (ROOT / c["runner"]).is_file(), c["runner"]
        assert c["model"] in MODELS["models"] and (c["draft"] is None or c["draft"] in MODELS["drafts"])
        pool = POOLS[c["workload"]["pool"]]
        assert pool["model"] == c["model"] and c["workload"]["rows"] <= pool["rows"]
        assert all(isinstance(v, str) for v in c["env"].values())
        assert not any(v.startswith("/") or "/home/" in v for v in c["env"].values()), c["id"]
        target = c["reference"].get("step_ms") or c["reference"].get("period_ms")
        assert target and 0 < c["tolerance"] < 0.1
        assert c["metric"] in ("step_mean", "spec_step_mean", "wall_period")
        assert (c["metric"] == "spec_step_mean") == (c["draft"] is not None) == ("MINISGL_AFD_SPEC" in c["env"])


def _write_logs(run: Path, ratio: int, steps: int, block: int = 1, accept: int = 1) -> None:
    """A coordinator log with a prefill step, then decode steps of 10 ms at 4 x ratio x 2 requests."""
    (run / "ray_logs").mkdir(parents=True)
    ranks, per_rank = 4 * ratio, 2
    ctx = [100] * ranks
    lines = ["[afd-step-time] step=0 phase=prefill bucket=0 num_mb=2 ms=50.0 real=[2] ctx=[100]"]
    stamps = []
    for step in range(1, steps + 1):
        ctx = [c + per_rank * accept for c in ctx]
        lines.append(f"[afd-step-time] step={step} phase=decode bucket=0 num_mb=2 ms=10.0 "
                     f"real=[{','.join([str(per_rank)] * ranks)}] ctx=[{','.join(map(str, ctx))}]")
        sec = 1_700_000_000 + step // 100   # 100 steps per second: a 10 ms wall period
        from datetime import datetime, timezone
        t = datetime.fromtimestamp(sec, tz=timezone.utc)
        stamps.append(f"[{t:%Y-%m-%d}|{t:%H:%M:%S}|afd-coordinator] [afd_step] step={step} dp=0 phase=decode bs={per_rank}")
    (run / "ray_logs" / "afd_coordinator.log").write_text("\n".join(lines) + "\n")
    (run / "ray_logs" / "afd-coordinator.log").write_text("\n".join(stamps) + "\n")


@pytest.mark.parametrize("ratio", [1, 7])
def test_step_mean_and_wall_period_on_a_synthetic_log(tmp_path, ratio):
    run_stats = _load("run_stats")
    _write_logs(tmp_path, ratio, steps=600)
    out = run_stats.measure(tmp_path, ratio, "step_mean")
    assert out["ok"] and out["mean_ms"] == 10.0 and out["b"] == 2
    assert out["tok_s_gpu"] == pytest.approx(2 / 10.0 * 1e3 * ratio / (ratio + 1), rel=1e-3)
    wall = run_stats.measure(tmp_path, ratio, "wall_period")
    assert wall["ok"] and wall["period_ms"] == pytest.approx(10.0, rel=1e-6)


def test_spec_steps_count_accepted_tokens(tmp_path):
    run_stats = _load("run_stats")
    _write_logs(tmp_path, 2, steps=200, accept=3)
    assert not run_stats.measure(tmp_path, 2, "step_mean")["ok"]   # context grew by 3x the batch: not a plain decode step
    out = run_stats.measure(tmp_path, 2, "spec_step_mean", block=4)
    assert out["ok"] and out["accept_len"] == 3.0
    assert out["tok_s_gpu"] == pytest.approx(2 * 3 / 10.0 * 1e3 * 2 / 3, rel=1e-3)


def test_env_and_show_name_the_runner_model_and_prompt_rows(tmp_path, monkeypatch, capsys):
    benchmark = _load("benchmark")
    c = benchmark.CONFIGS["qwen3-235b-4k-1to1-eagle3-draft3"]
    pools, out = tmp_path / "pools", tmp_path / "out"
    pools.mkdir()
    (pools / f"{c['workload']['pool']}.txt").write_text("".join(f"prompt {i}\n" for i in range(c["workload"]["rows"] + 3)))
    monkeypatch.setenv("FASTAFD_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("FASTAFD_POOL_DIR", str(pools))
    monkeypatch.setenv("FASTAFD_OUT", str(out))
    assert benchmark.cmd_env(type("A", (), {"config": c["id"]})()) == 0
    env = dict(line.removeprefix("export ").split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert env["PRESET"] == c["runner"] and env["FASTAFD_CONFIG_NODES"] == str(c["nodes"])
    assert env["MODEL_PATH"].endswith(MODELS["models"][c["model"]]["dir"])
    assert env["MINISGL_AFD_SPEC_DRAFT"].endswith(MODELS["drafts"][c["draft"]]["dir"])
    assert len((out / "prompts" / f"{c['workload']['pool']}_n{c['workload']['rows']}.txt").read_text().splitlines()) == c["workload"]["rows"]
    assert benchmark.cmd_show(type("A", (), {"config": c["id"], "nodes": "a,b"})()) == 0
    assert json.loads(capsys.readouterr().out)["nodes_used"] == ["a", "b"]
