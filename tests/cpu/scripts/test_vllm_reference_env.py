"""Select the vLLM reference env: a conda name keeps `conda run`, a path runs its own bin/vllm."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / "scripts/lib/common.sh"
VLLM_SERVER = ROOT / "scripts/serve/vllm_server.sh"
# Records how the fake `vllm` was started: its own path, argv, and environment.
FAKE_VLLM = """#!/bin/bash
printf '%s\\n' "$0" "$@" > "$FAKE_RECORD.argv"
env > "$FAKE_RECORD.env"
"""


def run_common(function: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    script = f'source "{COMMON}"; {function} "$@"'
    return subprocess.run(
        ["bash", "-c", script, "_", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
    )


def make_env(prefix: Path, *, with_vllm: bool = True) -> Path:
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin/python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    if with_vllm:
        vllm = prefix / "bin/vllm"
        vllm.write_text(FAKE_VLLM)
        vllm.chmod(0o755)
    return prefix


def serve(tmp_path: Path, env_arg: str, **extra_env: str) -> tuple[subprocess.CompletedProcess, Path]:
    record = tmp_path / "record"
    for stale in (Path(f"{record}.argv"), Path(f"{record}.env")):
        stale.unlink(missing_ok=True)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "FAKE_RECORD": str(record), **extra_env}
    result = subprocess.run(
        [
            "bash", str(VLLM_SERVER), "--env", env_arg, "--model", "/models/m", "--gpus", "0,1",
            "--port", "18000", "--tp-size", "2", "--extra-args", "--enforce-eager --max-num-seqs 8",
        ],
        env=env, capture_output=True, text=True, timeout=20, check=False,
    )
    return result, record


def recorded_env(record: Path) -> dict[str, str]:
    lines = Path(f"{record}.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


SERVE_ARGS = [
    "serve", "/models/m", "--host", "127.0.0.1", "--port", "18000",
    "--tensor-parallel-size", "2", "--enforce-eager", "--max-num-seqs", "8",
]


def test_bare_name_is_a_conda_env(tmp_path):
    result = run_common("fastafd_python_env_prefix", "vllm-cuda130", cwd=tmp_path)
    assert (result.returncode, result.stdout) == (0, "")


def test_path_resolves_to_an_absolute_prefix(tmp_path):
    prefix = make_env(tmp_path / "ref")
    for value in (str(prefix), "./ref"):
        result = run_common("fastafd_python_env_prefix", value, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(prefix)


@pytest.mark.parametrize("layout", ["missing", "no_python", "not_executable"])
def test_path_that_is_not_a_python_env_is_rejected(tmp_path, layout):
    prefix = tmp_path / "ref"
    if layout != "missing":
        (prefix / "bin").mkdir(parents=True)
    if layout == "not_executable":
        (prefix / "bin/python").write_text("")
    result = run_common("fastafd_python_env_prefix", str(prefix))
    assert result.returncode == 1
    assert "Not a Python environment" in result.stderr


@pytest.mark.parametrize(
    "vllm_env,expected",
    [
        ("minisgl-cuda130", []),
        ("vllm-cuda130", ["conda", "run", "--no-capture-output", "-n", "vllm-cuda130"]),
        ("/opt/vllm-reference", []),
    ],
)
def test_only_a_different_conda_env_goes_through_conda_run(vllm_env, expected):
    result = run_common("fastafd_vllm_env_launcher", "minisgl-cuda130", vllm_env)
    assert result.returncode == 0
    assert result.stdout.splitlines() == expected


def test_path_env_runs_its_own_vllm_without_conda(tmp_path):
    prefix = make_env(tmp_path / "ref")
    fake_conda = tmp_path / "conda"
    (fake_conda / "targets/sbsa-linux/include").mkdir(parents=True)
    (fake_conda / "targets/sbsa-linux/include/cuda_bf16.h").write_text("")
    result, record = serve(
        tmp_path, str(prefix), CONDA_DEFAULT_ENV="minisgl-cuda130", CONDA_PREFIX=str(fake_conda),
        CUDA_HOME="/caller/cuda",
    )
    assert result.returncode == 0, result.stderr
    argv = Path(f"{record}.argv").read_text().splitlines()
    assert argv == [str(prefix / "bin/vllm"), *SERVE_ARGS]
    env = recorded_env(record)
    assert env["PATH"].startswith(f"{prefix}/bin:")
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    # The calling conda env's toolchain block must not re-point a path env.
    assert env["CUDA_HOME"] == "/caller/cuda"


def test_setup_file_configures_only_a_path_env(tmp_path):
    prefix = make_env(tmp_path / "ref")
    setup = tmp_path / "reference.env"
    setup.write_text('export CUDA_HOME=/setup/cuda\nexport REFERENCE_MARK="$FAKE_RECORD"\n')
    result, record = serve(tmp_path, str(prefix), VLLM_ENV_SETUP=str(setup), CUDA_HOME="/caller/cuda")
    assert result.returncode == 0, result.stderr
    env = recorded_env(record)
    assert env["CUDA_HOME"] == "/setup/cuda"
    assert env["REFERENCE_MARK"] == str(record)
    assert "VLLM_ENV_SETUP" not in env

    result, record = serve(tmp_path, "vllm-cuda130", CONDA_DEFAULT_ENV="vllm-cuda130", VLLM_ENV_SETUP=str(setup))
    assert result.returncode == 1
    assert "applies only to a path-type --env" in result.stderr
    assert not Path(f"{record}.argv").exists()


def test_conda_env_still_runs_vllm_from_path(tmp_path):
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    (fake_bin / "vllm").write_text(FAKE_VLLM)
    (fake_bin / "vllm").chmod(0o755)
    path = f"{fake_bin}:/usr/bin:/bin"
    result, record = serve(tmp_path, "vllm-cuda130", CONDA_DEFAULT_ENV="vllm-cuda130", PATH=path)
    assert result.returncode == 0, result.stderr
    assert Path(f"{record}.argv").read_text().splitlines() == [str(fake_bin / "vllm"), *SERVE_ARGS]

    result, _ = serve(tmp_path, "vllm-cuda130", CONDA_DEFAULT_ENV="minisgl-cuda130", PATH=path)
    assert result.returncode == 1
    assert "Expected active conda env 'vllm-cuda130', got 'minisgl-cuda130'." in result.stderr


def test_path_env_without_vllm_is_rejected(tmp_path):
    prefix = make_env(tmp_path / "ref", with_vllm=False)
    result, record = serve(tmp_path, str(prefix))
    assert result.returncode == 1
    assert "has no executable bin/vllm" in result.stderr
    assert not os.path.exists(f"{record}.argv")
