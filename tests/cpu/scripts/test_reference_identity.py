"""Record which vLLM reference scored a report, and surface it without changing the verdict."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "scripts/validate/reference_identity.py"
CHECKER = ROOT / "scripts/validate/check_alignment_report.py"
VLLM_SERVER = ROOT / "scripts/serve/vllm_server.sh"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


ri = load("reference_identity", PROBE)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_package(root: Path) -> Path:
    package = root / "vllm"
    (package / "sub").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    (package / "__init__.py").write_text('__version__ = "9.9.9+test"\n')
    (package / "sub/x.py").write_text("".join(f"X{index} = {index}\n" for index in range(20)))
    (package / "_C.abi3.so").write_bytes(b"\x7fELF fake extension")
    (package / "__pycache__/x.cpython-312.pyc").write_bytes(b"bytecode")
    (package / "__pycache__/x.cpython-312.pyc.140230").write_bytes(b"interrupted bytecode write")
    (package / "stray.pyc").write_bytes(b"bytecode outside __pycache__")
    return package


def hermetic_git(monkeypatch, tmp_path: Path) -> dict[str, str]:
    """Isolate git from the user's config and from repositories above tmp_path."""
    config = tmp_path / "empty.gitconfig"
    config.write_text("")
    values = {"GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return dict(os.environ, **values)


def git(cwd: Path, env: dict[str, str], *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd, env=env, capture_output=True, check=True, timeout=30,
    )
    return result.stdout.decode().strip()


def test_tree_digest_is_sha256sum_of_sources_without_bytecode_or_extensions(tmp_path):
    package = make_package(tmp_path)
    files = ri.package_files(package)
    lines = "".join(
        f"{sha256((package / relative).read_bytes())}  {relative}\n" for relative in ("__init__.py", "sub/x.py")
    )
    assert files["tree_sha256"] == sha256(lines.encode())
    assert files["tree_files"] == 2
    extension = (package / "_C.abi3.so").read_bytes()
    assert files["extensions"] == [{"path": "_C.abi3.so", "sha256": sha256(extension), "bytes": len(extension)}]
    (package / "sub/x.py").write_text("X = 2\n")
    assert ri.package_files(package)["tree_sha256"] != files["tree_sha256"]


def test_git_identity_records_head_and_a_config_independent_diff(tmp_path, monkeypatch):
    env = hermetic_git(monkeypatch, tmp_path)
    repo = tmp_path / "src"
    package = make_package(repo)
    git(repo, env, "init", "-q")
    git(repo, env, "add", "vllm/__init__.py", "vllm/sub/x.py")
    git(repo, env, "commit", "-q", "-m", "base")
    head = git(repo, env, "rev-parse", "HEAD")

    clean = ri.git_source(package)
    assert (clean["head"], clean["dirty"], clean["diff_sha256"], clean["diff_files"]) == (head, False, None, [])

    (package / "sub/x.py").write_text((package / "sub/x.py").read_text().replace("X10 = 10", "X10 = -10"))
    dirty = ri.git_source(package)
    plain_diff = subprocess.run(
        ["git", "diff", "--full-index", "--binary", "HEAD", "--"], cwd=repo, env=env, capture_output=True, check=True
    ).stdout
    assert (dirty["head"], dirty["dirty"], dirty["diff_files"]) == (head, True, ["vllm/sub/x.py"])
    assert dirty["diff_sha256"] == sha256(plain_diff)

    # A user's diff settings must not change the recorded digest.
    hostile = tmp_path / "hostile.gitconfig"
    hostile.write_text(
        "[diff]\n\tnoprefix = true\n\tcontext = 7\n\texternal = false\n[color]\n\tui = always\n[core]\n\tabbrev = 5\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile))
    assert ri.git_source(package)["diff_sha256"] == dirty["diff_sha256"]


def test_source_outside_a_checkout_of_the_package_has_no_git_identity(tmp_path, monkeypatch):
    env = hermetic_git(monkeypatch, tmp_path)
    assert ri.git_source(make_package(tmp_path / "extracted")) is None
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "README").write_text("unrelated\n")
    git(outer, env, "init", "-q")
    git(outer, env, "add", "README")
    git(outer, env, "commit", "-q", "-m", "unrelated")
    assert ri.git_source(make_package(outer)) is None


def fake_site(root: Path) -> Path:
    package = make_package(root)
    (root / "torch").mkdir()
    (root / "torch/__init__.py").write_text('__version__ = "2.11.0+cu130"\n\nclass version:\n    cuda = "13.0"\n')
    return package


def test_probe_records_imported_modules_and_missing_ones_without_failing(tmp_path):
    package = fake_site(tmp_path / "site")
    output = tmp_path / "identity.json"
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "site"), CUDA_HOME="/opt/cuda-13.0",
               GIT_CEILING_DIRECTORIES=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-B", str(PROBE), "--output", str(output)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    identity = json.loads(output.read_text())
    vllm = identity["vllm"]
    assert (vllm["version"], vllm["git"]) == ("9.9.9+test", None)
    assert Path(vllm["file"]).resolve() == (package / "__init__.py").resolve()
    assert vllm["tree_sha256"] == ri.package_files(package)["tree_sha256"]
    assert [entry["path"] for entry in vllm["extensions"]] == ["_C.abi3.so"]
    assert (identity["torch"]["version"], identity["torch"]["cuda"]) == ("2.11.0+cu130", "13.0")
    # flashinfer is either absent or fails to import against the stand-in torch; both are recorded, not raised
    assert identity["flashinfer"]["error"].split(":")[0] in ("ModuleNotFoundError", "ImportError", "AttributeError")
    assert identity["cuda_home"] == "/opt/cuda-13.0"
    assert "vllm=9.9.9+test" in result.stdout


def test_package_dir_mode_digests_a_tree_without_importing_it(tmp_path):
    package = make_package(tmp_path / "tree")
    (package / "__init__.py").write_text("raise SystemExit('imported')\n")
    output = tmp_path / "source.json"
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-B", str(PROBE), "--package-dir", str(package), "--output", str(output)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    source = json.loads(output.read_text())
    assert (source["git"], source["tree_sha256"]) == (None, ri.package_files(package)["tree_sha256"])


def serve_with_probe(tmp_path: Path, python: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    prefix = tmp_path / "ref"
    (prefix / "bin").mkdir(parents=True)
    if python == "real":
        (prefix / "bin/python").symlink_to(sys.executable)
    else:
        (prefix / "bin/python").write_text("#!/bin/sh\nexit 3\n")
        (prefix / "bin/python").chmod(0o755)
    (prefix / "bin/vllm").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$FAKE_RECORD"\n')
    (prefix / "bin/vllm").chmod(0o755)
    fake_site(tmp_path / "site")
    record, output = tmp_path / "vllm.argv", tmp_path / "identity.json"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "FAKE_RECORD": str(record),
           "PYTHONPATH": str(tmp_path / "site"), "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    result = subprocess.run(
        ["bash", str(VLLM_SERVER), "--env", str(prefix), "--model", "/models/m", "--tp-size", "1",
         "--identity-json", str(output)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    return result, record, output


def test_server_records_identity_with_the_env_python_before_serving(tmp_path):
    result, record, output = serve_with_probe(tmp_path, "real")
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["vllm"]["version"] == "9.9.9+test"
    assert record.read_text().splitlines()[:2] == ["serve", "/models/m"]


def test_failed_identity_probe_does_not_block_serving(tmp_path):
    result, record, output = serve_with_probe(tmp_path, "failing")
    assert result.returncode == 0, result.stderr
    assert "WARNING: reference identity probe failed" in result.stderr
    assert not output.exists()
    assert record.read_text().splitlines()[:2] == ["serve", "/models/m"]


def report_pair(identity=None):
    sample = {"prompt": "A short prompt", "generated_token_ids": [41, 42]}
    result = {
        **sample,
        "num_generated_tokens": 2,
        "steps": [
            {"position": 0, "token_id": 41, "rank_in_vllm": 1},
            {"position": 1, "token_id": 42, "rank_in_vllm": 12},
        ],
    }
    report = {
        "results": [result],
        "summary": {
            "num_prompts": 1,
            "total_generated_tokens": 2,
            "overall_top1_agreement_rate": 0.5,
            "overall_top10_hit_rate": 0.5,
            "overall_top100_hit_rate": 1.0,
        },
    }
    if identity is not None:
        report["reference_identity"] = identity
    return report, {"samples": [copy.deepcopy(sample)]}


FULL_IDENTITY = {
    "schema": 1,
    "vllm": {
        "version": "0.21.1rc1.dev278+gd4004455d",
        "file": "/ref/src/vllm/vllm/__init__.py",
        "git": {"head": "d4004455d2357985830af10e432709b42c820455", "dirty": True, "diff_sha256": "ab" * 32},
        "tree_sha256": "cd" * 32,
        "extensions": [{"path": "_C.abi3.so", "sha256": "ef" * 32, "bytes": 1}],
    },
    "torch": {"version": "2.11.0+cu130"},
    "flashinfer": {"version": "0.6.11.post2"},
}


def check(tmp_path: Path, identity, min_top10: str) -> tuple[subprocess.CompletedProcess, dict]:
    report, samples = report_pair(identity)
    report_path, sample_path = tmp_path / "report.json", tmp_path / "sample.json"
    report_path.write_text(json.dumps(report))
    sample_path.write_text(json.dumps(samples))
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--report", str(report_path), "--sample-json", str(sample_path),
         "--min-top10", min_top10],
        capture_output=True, text=True, timeout=10, check=False,
    )
    return result, json.loads(result.stdout)


def test_report_without_identity_still_passes_with_a_warning(tmp_path):
    result, output = check(tmp_path, None, "0.5")
    assert result.returncode == 0
    assert (output["status"], output["reference_identity"]) == ("PASS", None)
    assert output["warnings"] == ["report has no reference_identity; the vLLM reference that scored it is unidentified"]
    assert "WARNING: report has no reference_identity" in result.stderr


def test_identity_is_printed_and_gaps_are_warnings_only(tmp_path):
    result, output = check(tmp_path, FULL_IDENTITY, "0.5")
    assert (result.returncode, output["warnings"], output["reference_identity"]) == (0, [], FULL_IDENTITY)

    partial = copy.deepcopy(FULL_IDENTITY)
    partial["vllm"].update(git=None, tree_sha256=None, extensions=[])
    partial["flashinfer"] = {"error": "ModuleNotFoundError: No module named 'flashinfer'"}
    result, output = check(tmp_path, partial, "0.5")
    assert result.returncode == 0
    assert output["warnings"] == [
        "reference_identity names no vLLM source (neither git.head nor tree_sha256)",
        "reference_identity lists no compiled vLLM extensions",
        "reference_identity lacks flashinfer.version",
    ]


def test_identity_never_rescues_a_failing_report(tmp_path):
    result, output = check(tmp_path, FULL_IDENTITY, "1.0")
    assert (result.returncode, output["status"], output["warnings"]) == (1, "FAIL", [])


# Importing the scorer pulls in urllib3, whose import-time IPv6 probe creates (and never connects) a socket.
@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket:UserWarning")
def test_score_copies_a_recorded_identity_and_warns_when_unusable(tmp_path, capsys):
    compare = load("compare_minisgl_vllm", ROOT / "scripts/validate/compare_minisgl_vllm.py")
    recorded = tmp_path / "identity.json"
    recorded.write_text(json.dumps(FULL_IDENTITY))
    not_object = tmp_path / "list.json"
    not_object.write_text("[]")
    assert compare.load_reference_identity(recorded) == FULL_IDENTITY
    assert compare.load_reference_identity(None) is None
    assert compare.load_reference_identity(tmp_path / "missing.json") is None
    assert compare.load_reference_identity(not_object) is None
    errors = capsys.readouterr().err
    assert "WARNING: no reference identity recorded" in errors
    assert "is not a JSON object" in errors
