"""scripts/enter_clean.sh must make the libcuda link stub of a CUDA 13 Conda env visible."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _library_path(tmp_path: Path, *, stub: bool) -> list[str]:
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").symlink_to(sys.executable)
    if stub:
        (prefix / "lib" / "stubs").mkdir(parents=True)
        (prefix / "lib" / "stubs" / "libcuda.so").write_text("")
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text(
        "conda() {\n"
        '  [ "$1" = activate ] || return 1\n'
        f'  export CONDA_PREFIX="{prefix}" CONDA_DEFAULT_ENV="$2" PATH="{prefix}/bin:$PATH"\n'
        "}\n"
    )
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "CONDA_SH": str(conda_sh),
           "WORKDIR": str(tmp_path), "CACHE_ROOT": str(tmp_path / "cache")}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "enter_clean.sh"), "fake-env"],
        input='echo "LIBRARY_PATH=$LIBRARY_PATH"\nexit\n',
        env=env, capture_output=True, text=True, timeout=30, check=True,
    )
    line = next(x for x in result.stdout.splitlines() if x.startswith("LIBRARY_PATH="))
    return line.removeprefix("LIBRARY_PATH=").split(os.pathsep)


def test_the_stub_directory_is_searched_last(tmp_path):
    paths = _library_path(tmp_path, stub=True)
    prefix = tmp_path / "env"
    assert paths[:2] == [f"{prefix}/lib", f"{prefix}/targets/sbsa-linux/lib"]
    assert paths[-1] == f"{prefix}/lib/stubs"


def test_envs_without_a_stub_keep_the_previous_path(tmp_path):
    paths = _library_path(tmp_path, stub=False)
    assert not any(path.endswith("/stubs") for path in paths)
