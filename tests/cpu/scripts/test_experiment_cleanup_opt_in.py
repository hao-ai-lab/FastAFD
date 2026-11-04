"""Experiment presets run the cluster-wide kill_afd_ray_tasks.sh only when CLEAN_FIRST/CLEAN_AFTER=1."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = "scripts/experiments/afd/qwen3_30b/run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh"
NODES = {"AFD_NODE_LIST": "10.0.0.1,10.0.0.2"}
# Scripts that end in LAUNCHER, with the environment each needs to get there without a cluster.
PRESETS = {
    "scripts/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh": {},
    "scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh": {},
    "scripts/experiments/afd/qwen35/run_qwen35_fp8_afd_dynamicnode.sh": {
        **NODES, "MODEL_PATH": "/models/m", "AFD_NUM_MB": "2", "PER_ATTN_GPU_BSZ": "8",
    },
    "scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh": NODES,
    "scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_16k_b48_dynamicnode_mb2_nsys_alignment.sh": NODES,
    # Fixed ports skip the preset's socket probe.
    "scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh": {
        **NODES, "MINISGL_PORT": "18000", "VLLM_PORT": "18020",
    },
    "scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_16k_b36_dynamicnode_mb2_nsys_alignment.sh": {
        **NODES, "MINISGL_PORT": "18000", "VLLM_PORT": "18020",
    },
}
SHELL_SCRIPTS = sorted(
    {"scripts/kill_afd_ray_tasks.sh", *PRESETS}
    | {str(path.relative_to(ROOT)) for path in (ROOT / "scripts/experiments").rglob("*.sh")}
)


def stub_tree(tmp_path: Path) -> Path:
    """Copy scripts/ into a throwaway repository root so its programs can be replaced by recorders."""
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts", repo / "scripts")
    (repo / "python").mkdir()
    (repo / "prompts").mkdir()
    for length in (8192, 16384):
        (repo / f"prompts/prompts_512x{length}_seed20260527.txt").write_text("first prompt\nsecond prompt\n")
    return repo


def write_program(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def run(repo: Path, script: str, tmp_path: Path, **extra_env: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": f"{tmp_path / 'bin'}:{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        # The copy has no .git; stop git from resolving an enclosing checkout as the repository root.
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        "RECORD": str(tmp_path / "record"),
        **extra_env,
    }
    return subprocess.run(
        ["bash", str(repo / script)], env=env, capture_output=True, text=True, timeout=30, check=False
    )


def recorded(tmp_path: Path) -> list[str]:
    record = tmp_path / "record"
    return record.read_text().splitlines() if record.exists() else []


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_script_parses(script):
    result = subprocess.run(["bash", "-n", str(ROOT / script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_no_script_defaults_the_cleanup_on():
    # Catches a new preset that turns the cleanup back on without being added to PRESETS.
    assignment = re.compile(r"^\s*(?:export\s+)?(CLEAN_FIRST|CLEAN_AFTER)=(.*)$", re.MULTILINE)
    found = [
        (str(path.relative_to(ROOT)), name, value)
        for path in sorted((ROOT / "scripts").rglob("*.sh"))
        for name, value in assignment.findall(path.read_text())
    ]
    assert found
    not_off = [(script, name, value) for script, name, value in found if value != f'"${{{name}:-0}}"']
    assert not_off == []


@pytest.mark.parametrize("script", sorted(PRESETS))
@pytest.mark.parametrize("opt_in", [False, True], ids=["default", "opt_in"])
def test_preset_hands_the_launcher_cleanup_off_by_default(tmp_path, script, opt_in):
    repo = stub_tree(tmp_path)
    write_program(repo / LAUNCHER, 'echo "clean_first=$CLEAN_FIRST clean_after=$CLEAN_AFTER" > "$RECORD"\n')
    extra = {"CLEAN_FIRST": "1", "CLEAN_AFTER": "1"} if opt_in else {}
    result = run(repo, script, tmp_path, **PRESETS[script], **extra)
    assert result.returncode == 0, result.stderr
    expected = "1" if opt_in else "0"
    assert recorded(tmp_path) == [f"clean_first={expected} clean_after={expected}"]


@pytest.mark.parametrize("opt_in", [False, True], ids=["default", "opt_in"])
def test_launcher_calls_the_cluster_cleanup_only_when_asked(tmp_path, opt_in):
    repo = stub_tree(tmp_path)
    write_program(repo / "scripts/kill_afd_ray_tasks.sh", 'echo kill >> "$RECORD"\n')
    write_program(repo / "scripts/serve/fastafd_server.sh", 'echo server >> "$RECORD"\nexec sleep 30\n')
    (repo / "scripts/validate/compare_minisgl_vllm.py").write_text("")
    write_program(tmp_path / "bin/curl", "exit 0\n")
    for tool in ("nvcc", "nsys"):
        write_program(tmp_path / f"cuda/bin/{tool}", "exit 0\n")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first prompt\n")
    extra = {"CLEAN_FIRST": "1", "CLEAN_AFTER": "1"} if opt_in else {}
    result = run(
        repo, LAUNCHER, tmp_path,
        CONDA_DEFAULT_ENV="minisgl-cuda130", CUDA_13_HOME=str(tmp_path / "cuda"), MINISGL_PORT="18000",
        PROMPT_KIND="existing", PROMPT_FILE=str(prompts), RUN_DIR=str(tmp_path / "run"),
        POST_SHUTDOWN_SLEEP="0", **extra,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("Done.")
    expected = "1" if opt_in else "0"
    assert f"  clean_first: {expected}\n  clean_after: {expected}\n" in result.stdout
    assert recorded(tmp_path) == (["kill", "server", "kill"] if opt_in else ["server"])
