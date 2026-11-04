"""Alignment presets default to the README's env and to the reproducible vLLM reference.

Each preset runs in a copy of its script tree whose aligner is a stand-in that
prints its arguments and VLLM_ENV_SETUP, so the assertions see the envs the real
aligner would get.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VALIDATE = Path("scripts/validate")
FP8_397B = VALIDATE / "qwen35_397b_a17b_fp8_fastafd_alignment.sh"
PRESETS = [FP8_397B]
WARNING = (
    "WARNING: VLLM_ENV_NAME and FASTAFD_VLLM_REFERENCE are unset; scoring in conda env "
    "'vllm-cuda130', which is not the reproducible vLLM reference (README.md, Reproducible "
    "vLLM Reference)."
)
NOTE = "NOTE: no vllm-reference.env in FASTAFD_VLLM_REFERENCE="


def _run(tmp_path: Path, preset: Path, **extra_env: str) -> subprocess.CompletedProcess:
    root = tmp_path / "repo"
    for rel in (Path("scripts/lib/common.sh"), preset, VALIDATE / "qwen35_fastafd_alignment.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, root / rel)
    aligner = root / VALIDATE / "fastafd_vllm_alignment.sh"
    aligner.write_text(
        "#!/usr/bin/env bash\nprintf 'ARGV %s\\n' \"$@\"\n"
        'echo "SETUP ${VLLM_ENV_SETUP-<unset>}"\n'
        'echo "AFD_BACKEND ${MINISGL_AFD_MOE_BACKEND-<unset>}"\n'
        'echo "AFD_EXTRA ${AFD_SERVER_EXTRA_ARGS-<unset>}"\n'
    )
    aligner.chmod(0o755)
    # Repository-root discovery outside git looks for scripts/ next to python/.
    (root / "python").mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("p\n")
    tiktoken = tmp_path / "tiktoken"
    tiktoken.mkdir()
    (tiktoken / "o200k_base.tiktoken").write_text("stub")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        # Given explicitly so neither the hub nor the model-view builder is needed.
        "MODEL_PATH": str(model),
        "VLLM_MODEL_PATH": str(model),
        "PROMPT_FILE": str(prompts),
        "TIKTOKEN_ENCODINGS_BASE": str(tiktoken),
        **extra_env,
    }
    return subprocess.run(
        ["bash", str(root / preset)], env=env, capture_output=True, text=True, timeout=30, check=False
    )


def _envs(result: subprocess.CompletedProcess) -> tuple[str, str, str]:
    """The aligner's --env and --vllm-env, and the VLLM_ENV_SETUP it inherits."""
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    argv = [line.removeprefix("ARGV ") for line in lines if line.startswith("ARGV ")]
    args = dict(zip(argv, argv[1:]))
    [setup] = [line.removeprefix("SETUP ") for line in lines if line.startswith("SETUP ")]
    return args["--env"], args["--vllm-env"], setup


def _reference(tmp_path: Path, *, with_setup: bool) -> Path:
    """An install prefix as build_vllm_reference.sh leaves it, with or without its setup file."""
    prefix = tmp_path / "vllm-reference"
    prefix.mkdir()
    if with_setup:
        (prefix / "vllm-reference.env").write_text("export CUDA_HOME=/cuda\n")
    return prefix


@pytest.mark.parametrize("preset", PRESETS)
def test_defaults_are_the_readme_env_and_a_warned_fallback_reference(tmp_path, preset):
    result = _run(tmp_path, preset)
    assert _envs(result) == ("minisgl-cuda130", "vllm-cuda130", "<unset>")
    assert result.stderr.splitlines().count(WARNING) == 1


@pytest.mark.parametrize("preset", PRESETS)
def test_reference_prefix_brings_its_setup_file(tmp_path, preset):
    prefix = _reference(tmp_path, with_setup=True)
    result = _run(tmp_path, preset, FASTAFD_VLLM_REFERENCE=str(prefix))
    assert _envs(result) == ("minisgl-cuda130", str(prefix), str(prefix / "vllm-reference.env"))
    assert "WARNING" not in result.stderr
    assert NOTE not in result.stderr


@pytest.mark.parametrize("preset", PRESETS)
def test_reference_prefix_without_setup_file_leaves_it_unset_with_a_note(tmp_path, preset):
    prefix = _reference(tmp_path, with_setup=False)
    result = _run(tmp_path, preset, FASTAFD_VLLM_REFERENCE=str(prefix))
    assert _envs(result) == ("minisgl-cuda130", str(prefix), "<unset>")
    assert [line for line in result.stderr.splitlines() if line.startswith(NOTE)] == [
        f"{NOTE}{prefix}; VLLM_ENV_SETUP stays unset and the vLLM reference runs with this "
        "shell's CUDA toolkit."
    ]


@pytest.mark.parametrize("preset", PRESETS)
def test_explicit_setup_file_wins_over_the_prefix(tmp_path, preset):
    prefix = _reference(tmp_path, with_setup=True)
    result = _run(tmp_path, preset, FASTAFD_VLLM_REFERENCE=str(prefix), VLLM_ENV_SETUP="/custom/setup.env")
    assert _envs(result) == ("minisgl-cuda130", str(prefix), "/custom/setup.env")
    assert NOTE not in result.stderr


@pytest.mark.parametrize("preset", PRESETS)
def test_explicit_env_names_win_unchanged(tmp_path, preset):
    # A conda env name in VLLM_ENV_NAME keeps its meaning and outranks the prefix,
    # whose setup file must not reach a conda env (vllm_server.sh rejects that).
    prefix = _reference(tmp_path, with_setup=True)
    result = _run(
        tmp_path, preset, ENV_NAME="my-afd-env", VLLM_ENV_NAME="my-vllm-env",
        FASTAFD_VLLM_REFERENCE=str(prefix),
    )
    assert _envs(result) == ("my-afd-env", "my-vllm-env", "<unset>")
    assert "WARNING" not in result.stderr
    assert NOTE not in result.stderr


@pytest.mark.parametrize("preset", PRESETS)
def test_reference_prefix_must_be_a_path(tmp_path, preset):
    result = _run(tmp_path, preset, FASTAFD_VLLM_REFERENCE="vllm-reference")
    assert result.returncode == 1
    assert "FASTAFD_VLLM_REFERENCE must be the install prefix of the vLLM reference, a path" in result.stderr
    assert "ARGV" not in result.stdout


def test_no_alignment_preset_defaults_to_a_private_env():
    # Static guard for presets added later: the README names one AFD env.
    for script in sorted((ROOT / VALIDATE).glob("*.sh")):
        text = script.read_text()
        assert "minisgl-cuda130-" not in text, script
