"""Keep the vLLM reference recipe's vendored inputs, build script and README in agreement."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "scripts/reference"
BUILD = RECIPE / "build_vllm_reference.sh"
# sha256 of the reference checkout's uncommitted change as one diff (see the patch headers).
RECORDED_DIFF_SHA256 = "899ee3acce35eee4da5c3ae57e5479782e3faa1173981786f1eafe27e5e303e3"


def script_constant(name: str) -> str:
    match = re.search(rf'^{name}="([^"]*)"$', BUILD.read_text(), re.MULTILINE)
    assert match, name
    return match.group(1)


def test_patches_are_the_recorded_reference_diff_with_attribution():
    patches = sorted((RECIPE / "patches").glob("*.patch"))
    assert [patch.name[:5] for patch in patches] == ["0001-", "0002-"]
    bodies = []
    for patch in patches:
        data = patch.read_bytes()
        start = re.search(rb"(?m)^diff --git ", data).start()
        header = data[:start].decode()
        assert "Apache License 2.0" in header
        assert script_constant("VLLM_COMMIT") in header
        bodies.append(data[start:])
    assert hashlib.sha256(b"".join(bodies)).hexdigest() == RECORDED_DIFF_SHA256


def test_precommit_fixers_leave_the_patches_alone():
    # A blank context line of a unified diff is a single space, which trailing-whitespace strips.
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    patches = [str(patch.relative_to(ROOT)) for patch in sorted((RECIPE / "patches").glob("*.patch"))]
    assert patches
    for hook_id in ("trailing-whitespace", "end-of-file-fixer"):
        # pre-commit applies `exclude` with re.search; its default "^$" excludes nothing.
        exclude = re.compile(hooks[hook_id].get("exclude", "^$"))
        assert all(exclude.search(patch) for patch in patches), hook_id
        assert not exclude.search("scripts/reference/build_vllm_reference.sh"), hook_id


def test_lock_pins_every_wheel_by_hash_and_matches_script_and_readme():
    lines = [line for line in (RECIPE / "requirements.lock").read_text().splitlines() if line and line[0] != "#"]
    assert len(lines) == 195
    pins = {}
    for line in lines:
        match = re.fullmatch(r"([A-Za-z0-9._-]+)==(\S+) --hash=sha256:[0-9a-f]{64}", line)
        assert match, line
        pins[match.group(1).lower()] = match.group(2)
    torch, flashinfer = script_constant("TORCH_VERSION"), script_constant("FLASHINFER_VERSION")
    assert pins["torch"] == torch
    assert pins["flashinfer-python"] == pins["flashinfer-cubin"] == flashinfer
    readme = (ROOT / "README.md").read_text()
    for pinned in (torch, flashinfer, script_constant("VLLM_COMMIT"), script_constant("VLLM_WHEEL_SHA256")[:16]):
        assert pinned in readme


def run_build(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(BUILD), *args], capture_output=True, text=True, timeout=20, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
    )


def test_build_needs_a_new_prefix_before_touching_anything(tmp_path):
    missing = run_build()
    assert missing.returncode == 1
    assert "--prefix is required" in missing.stderr
    existing = run_build("--prefix", str(tmp_path))
    assert existing.returncode == 1
    assert "refusing to reuse existing" in existing.stderr
    assert list(tmp_path.iterdir()) == []
    assert run_build("--help").returncode == 0
