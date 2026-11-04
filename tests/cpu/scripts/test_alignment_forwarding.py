"""The FastAFD/vLLM aligner and its Qwen3.5 wrappers pass on what their callers set.

The aligner runs in a copy of its script tree whose servers and sampling/scoring
client are stand-ins that record how they were started; only its HTTP readiness
probe is answered without a server. The real check_alignment_report.py gates the
stand-in report. The wrappers run against a stand-in aligner that prints its
arguments, so the assertions see exactly what the real aligner would be given.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VALIDATE = Path("scripts/validate")
ALIGNER = VALIDATE / "fastafd_vllm_alignment.sh"
FP8_397B = VALIDATE / "qwen35_397b_a17b_fp8_fastafd_alignment.sh"

# Both servers record their environment and arguments, then stay up until stopped.
SERVER_STAND_IN = """#!/usr/bin/env bash
name="$(basename "$0" .sh)"
env > "$STAND_IN_RECORD/$name.env"
printf '%s\\n' "$@" > "$STAND_IN_RECORD/$name.argv"
exec sleep 60
"""
# Samples one prompt and scores it with the ranks in STAND_IN_RANKS.
CLIENT_STAND_IN = """import json, os, sys

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

ranks = [int(rank) for rank in os.environ["STAND_IN_RANKS"].split(",")]
ids = list(range(41, 41 + len(ranks)))
sample = {"prompt": "p", "generated_token_ids": ids}
if sys.argv[1] == "sample":
    with open(value("--sample-json"), "w") as out:
        json.dump({"samples": [sample]}, out)
else:
    steps = [
        {"position": i, "token_id": token, "rank_in_vllm": rank}
        for i, (token, rank) in enumerate(zip(ids, ranks))
    ]
    n = len(ranks)
    summary = {
        "num_prompts": 1,
        "total_generated_tokens": n,
        "overall_top1_agreement_rate": sum(rank == 1 for rank in ranks) / n,
        "overall_top10_hit_rate": sum(rank <= 10 for rank in ranks) / n,
        "overall_top100_hit_rate": sum(rank <= 100 for rank in ranks) / n,
    }
    report = {"results": [{**sample, "num_generated_tokens": n, "steps": steps}], "summary": summary}
    with open(value("--output-json"), "w") as out:
        json.dump(report, out)
"""
# The readiness probe (`python - <url>`) is the aligner's only HTTP request.
PYTHON_SHIM = f"""#!/usr/bin/env bash
if [[ "${{1:-}}" == "-" && "${{2:-}}" == http://* ]]; then cat >/dev/null; exit 0; fi
exec {sys.executable} "$@"
"""
ARGV_STAND_IN = """#!/usr/bin/env bash
printf 'ARGV %s\\n' "$@"
"""


def _write(path: Path, text: str, mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def _repo_copy(tmp_path: Path, *files: Path) -> Path:
    root = tmp_path / "repo"
    for rel in (Path("scripts/lib/common.sh"), *files):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, root / rel)
    # Repository-root discovery outside git looks for scripts/ next to python/.
    (root / "python").mkdir()
    return root


def _run_aligner(tmp_path: Path, ranks: str = "1,4", **extra_env: str):
    root = _repo_copy(tmp_path, ALIGNER, VALIDATE / "check_alignment_report.py")
    _write(root / "scripts/serve/fastafd_server.sh", SERVER_STAND_IN)
    _write(root / "scripts/serve/vllm_server.sh", SERVER_STAND_IN)
    _write(root / VALIDATE / "compare_minisgl_vllm.py", CLIENT_STAND_IN, 0o644)
    _write(tmp_path / "bin/python", PYTHON_SHIM)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("p\n")
    record = tmp_path / "record"
    record.mkdir()
    env = {
        "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        "CONDA_DEFAULT_ENV": "minisgl-cuda130",
        "STAND_IN_RECORD": str(record),
        "STAND_IN_RANKS": ranks,
        "TMP_DIR": str(tmp_path / "run"),
        **extra_env,
    }
    result = subprocess.run(
        [
            "bash", str(root / ALIGNER), "--env", "minisgl-cuda130", "--model", "/models/m",
            "--prompt-file", str(prompts), "--afd-port", "20000", "--vllm-port", "20010",
            "--output-json", str(tmp_path / "report.json"), "--keep-artifacts",
        ],
        env=env, capture_output=True, text=True, timeout=50, check=False,
    )
    return result, record


def _recorded_env(record: Path, name: str) -> dict[str, str]:
    lines = (record / f"{name}.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def _gate_output(result: subprocess.CompletedProcess) -> dict:
    """The checker's JSON verdict, printed between the scoring and completion steps."""
    text = result.stdout.split("==> Scoring under vLLM", 1)[1].split("==> Completed", 1)[0]
    return json.loads(text)


def test_engine_ready_timeout_defaults_to_3600_for_the_vllm_process_only(tmp_path):
    result, record = _run_aligner(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "  vllm_engine_ready_timeout_s: 3600" in result.stdout.splitlines()
    assert _recorded_env(record, "vllm_server")["VLLM_ENGINE_READY_TIMEOUT_S"] == "3600"
    # The AFD server starts first and must not inherit a vLLM-only setting.
    assert "VLLM_ENGINE_READY_TIMEOUT_S" not in _recorded_env(record, "fastafd_server")


def test_engine_ready_timeout_from_the_caller_reaches_vllm(tmp_path):
    result, record = _run_aligner(tmp_path, VLLM_ENGINE_READY_TIMEOUT_S="1200")
    assert result.returncode == 0, result.stderr
    assert _recorded_env(record, "vllm_server")["VLLM_ENGINE_READY_TIMEOUT_S"] == "1200"


@pytest.mark.parametrize("value", ["10m", "0", "-5", "1.5"])
def test_invalid_engine_ready_timeout_fails_before_afd_starts(tmp_path, value):
    result, record = _run_aligner(tmp_path, VLLM_ENGINE_READY_TIMEOUT_S=value)
    assert result.returncode == 1
    assert "VLLM_ENGINE_READY_TIMEOUT_S must be a positive whole number of seconds" in result.stderr
    assert not (record / "fastafd_server.env").exists()


def test_top100_gate_defaults_to_one_and_follows_align_min_top100_rate(tmp_path):
    # Rank 150 misses the top 100: top-10 and top-100 are both 1/2.
    failed, _ = _run_aligner(tmp_path / "default", ranks="1,150", ALIGN_MIN_TOP10_RATE="0.5")
    assert failed.returncode == 1
    verdict = _gate_output(failed)
    assert verdict["failed_metrics"] == ["overall_top100_hit_rate"]
    assert verdict["thresholds"]["overall_top100_hit_rate"] == 1.0

    passed, _ = _run_aligner(
        tmp_path / "relaxed", ranks="1,150", ALIGN_MIN_TOP10_RATE="0.5", ALIGN_MIN_TOP100_RATE="0.5"
    )
    assert passed.returncode == 0, passed.stderr
    assert _gate_output(passed)["status"] == "PASS"


def _run_wrapper(tmp_path: Path, preset: Path, **extra_env: str):
    root = _repo_copy(
        tmp_path, preset, VALIDATE / "qwen35_fastafd_alignment.sh",
    )
    _write(root / ALIGNER, ARGV_STAND_IN)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("p\n")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        # Given explicitly so neither the hub nor the model-view builder is needed.
        "MODEL_PATH": "/models/m",
        "VLLM_MODEL_PATH": "/models/m-view",
        "PROMPT_FILE": str(prompts),
        **extra_env,
    }
    result = subprocess.run(
        ["bash", str(root / preset)], env=env, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    argv = [line.removeprefix("ARGV ") for line in lines if line.startswith("ARGV ")]
    return lines, dict(zip(argv, argv[1:]))


def test_397b_display_name_can_be_overridden(tmp_path):
    lines, _ = _run_wrapper(tmp_path / "default", FP8_397B)
    assert "Running Qwen3.5-397B-A17B-FP8 mb1 AFD alignment" in lines

    lines, _ = _run_wrapper(tmp_path / "named", FP8_397B, MODEL_DISPLAY_NAME="Qwen3.5-397B-FP8-local")
    assert "Running Qwen3.5-397B-FP8-local mb1 AFD alignment" in lines


DP4_TOPOLOGY = {
    "ATTN_DP_SIZE": "4", "ATTN_TP_SIZE": "1", "MLP_DP_SIZE": "4", "MLP_TP_SIZE": "1", "AFD_MLP_EP_SIZE": "4",
}


def test_topology_echo_shows_dp_tp_and_ep_as_forwarded(tmp_path):
    lines, _ = _run_wrapper(tmp_path / "default", FP8_397B)
    assert "  afd_topology: AG-DP1/TP4 + EG-DP1/TP4/EP4" in lines

    lines, args = _run_wrapper(tmp_path / "dp4", FP8_397B, **DP4_TOPOLOGY)
    assert "  afd_topology: AG-DP4/TP1 + EG-DP4/TP1/EP4" in lines
    assert [args[flag] for flag in ("--afd-attn-dp-size", "--attn-tp-size")] == ["4", "1"]
    assert [args[flag] for flag in ("--afd-mlp-dp-size", "--mlp-tp-size", "--afd-mlp-ep-size")] == ["4", "1", "4"]
