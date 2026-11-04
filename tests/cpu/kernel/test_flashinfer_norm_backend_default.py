"""Importing minisgl pins FlashInfer's norm ops to its CUDA kernels unless the environment chose already."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[3] / "python"
KEY = "FLASHINFER_USE_CUDA_NORM"


def _value_after_import(module: str, preset: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != KEY}
    env["PYTHONPATH"] = str(PYTHON_DIR)
    if preset is not None:
        env[KEY] = preset
    code = f"import os, {module}; print(repr(os.environ.get({KEY!r})))"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


# minisgl.ray_env_keys imports nothing else: the default must come from the package __init__, which
# Python runs before any submodule, so it precedes every lazy `import flashinfer` in minisgl.
@pytest.mark.parametrize("module", ["minisgl", "minisgl.ray_env_keys"])
def test_unset_defaults_to_the_cuda_norm_kernels(module):
    assert _value_after_import(module, None) == "'1'"


@pytest.mark.parametrize("preset", ["0", "1"])
def test_an_explicit_choice_is_kept(preset):
    assert _value_after_import("minisgl", preset) == repr(preset)


def test_the_package_is_the_source_under_test():
    # A regular package (with __init__.py) from this checkout, not a namespace spread over checkouts.
    import minisgl

    assert Path(minisgl.__file__).resolve() == PYTHON_DIR / "minisgl" / "__init__.py"


# A stand-in flashinfer that, like flashinfer/norm/__init__.py:57 in 0.6.11, reads the switch once at import.
_STUB = 'import os\n_USE_CUDA_NORM = os.environ.get("FLASHINFER_USE_CUDA_NORM", "0") == "1"\n'


def _use_cuda_norm(tmp_path: Path, code: str) -> str:
    (tmp_path / "flashinfer").mkdir(exist_ok=True)
    (tmp_path / "flashinfer" / "__init__.py").write_text(_STUB)
    env = {k: v for k, v in os.environ.items() if k != KEY}
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(PYTHON_DIR)))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_the_default_only_counts_when_minisgl_is_imported_before_flashinfer(tmp_path):
    report = "import flashinfer; print(flashinfer._USE_CUDA_NORM)"
    assert _use_cuda_norm(tmp_path, f"import flashinfer, minisgl; {report}") == "False"
    assert _use_cuda_norm(tmp_path, f"import minisgl, flashinfer; {report}") == "True"
