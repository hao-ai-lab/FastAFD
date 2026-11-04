"""The Ray nsys preflight must see the raylet PATH, report every node, and name the fix."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from minisgl.server import supervisor_ray

NSYS_BIN = "/opt/cuda/bin/nsys"
PYTHON_DIR = str(Path(supervisor_ray.__file__).resolve().parents[2])


def _ok(host: str) -> dict:
    return {"host": host, "which_nsys": NSYS_BIN, "version_rc": 0, "version": "NVIDIA Nsight Systems version 2025.3.2"}


def _missing(host: str) -> dict:
    return {"host": host, "which_nsys": None, "version_rc": None, "version": ""}


def test_all_nodes_ok_is_not_an_error():
    results = {"10.0.0.1": _ok("node-a"), "10.0.0.2": _ok("node-b")}
    assert supervisor_ray._nsys_preflight_error(results, NSYS_BIN) is None


def test_missing_nsys_names_the_raylet_path_fix_and_lists_every_node():
    results = {"10.0.0.1": _ok("node-a"), "10.0.0.2": _missing("node-b")}
    message = supervisor_ray._nsys_preflight_error(results, NSYS_BIN)
    assert message is not None
    assert "failed on 1 of 2 node(s)" in message
    assert "Ray daemon (raylet) PATH" in message
    assert f"MINISGL_RAY_NSYS_BIN={NSYS_BIN} only reaches this driver" in message
    assert "append the nsys directory (for example /opt/cuda/bin) to PATH" in message
    assert "restart those Ray daemons" in message
    assert "  OK 10.0.0.1 (node-a): which nsys=/opt/cuda/bin/nsys, nsys --version rc=0: NVIDIA" in message
    assert "  FAIL 10.0.0.2 (node-b): which nsys=None" in message


@pytest.mark.parametrize(
    ("result", "line"),
    [
        (
            {"host": "node-c", "which_nsys": NSYS_BIN, "version_rc": 1, "version": "no license"},
            "  FAIL 10.0.0.3 (node-c): which nsys=/opt/cuda/bin/nsys, nsys --version rc=1: no license",
        ),
        ({"error": "GetTimeoutError: timed out"}, "  FAIL 10.0.0.3 (?): probe task failed: GetTimeoutError: timed out"),
    ],
)
def test_failed_version_or_failed_probe_task_is_an_error(result, line):
    message = supervisor_ray._nsys_preflight_error({"10.0.0.3": result}, NSYS_BIN)
    assert message is not None
    assert line in message.splitlines()


def _write_fake_nsys(directory: Path, exit_code: int) -> Path:
    path = directory / "nsys"
    path.write_text(f"#!/bin/sh\necho 'NVIDIA Nsight Systems version test'\nexit {exit_code}\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("exit_code", [0, 3])
def test_probe_runs_the_bare_nsys_found_on_path(tmp_path, monkeypatch, exit_code):
    fake = _write_fake_nsys(tmp_path, exit_code)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}/usr/bin{os.pathsep}/bin")
    result = supervisor_ray._probe_nsys_on_node()
    assert result["which_nsys"] == str(fake)
    assert result["version_rc"] == exit_code
    assert result["version"] == "NVIDIA Nsight Systems version test"


def test_probe_reports_nsys_absent_from_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    result = supervisor_ray._probe_nsys_on_node()
    assert result["which_nsys"] is None
    assert result["version_rc"] is None


class _FakeRay:
    """Records task submissions; `get` returns (or raises) the canned outcome for the task's node."""

    class exceptions:
        class RayError(Exception):
            pass

    def __init__(self, outcomes: dict):
        self.outcomes = outcomes
        self.remote_kwargs = None
        self.submissions = []

    def nodes(self):
        return [
            {"Alive": True, "NodeManagerAddress": ip, "Resources": {"CPU": 8.0, f"node:{ip}": 1.0}}
            for ip in self.outcomes
        ]

    def remote(self, **kwargs):
        self.remote_kwargs = kwargs
        fake = self

        class _RemoteFunction:
            def __init__(self, fn):
                assert fn is supervisor_ray._probe_nsys_on_node

            def options(self, **options):
                (resource,) = options["resources"]
                ip = resource.removeprefix("node:")
                fake.submissions.append((ip, options))
                return SimpleNamespace(remote=lambda: ip)

        return _RemoteFunction

    def get(self, ref, timeout):
        outcome = self.outcomes[ref]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_preflight_probes_each_node_once_with_pythonpath_only():
    fake = _FakeRay({"10.0.0.1": _ok("node-a"), "10.0.0.2": _ok("node-b")})
    results = supervisor_ray._preflight_ray_nsys(fake, ["10.0.0.2", "10.0.0.1", "10.0.0.2"], NSYS_BIN)
    assert results == {"10.0.0.1": _ok("node-a"), "10.0.0.2": _ok("node-b")}
    assert fake.remote_kwargs == {"num_cpus": 0, "max_retries": 0}
    assert [ip for ip, _ in fake.submissions] == ["10.0.0.1", "10.0.0.2"]
    for ip, options in fake.submissions:
        assert options["resources"] == {f"node:{ip}": 0.001}
        # Forwarding PATH would show the driver's PATH instead of the raylet PATH the nsight plugin uses.
        assert options["runtime_env"] == {"env_vars": {"PYTHONPATH": PYTHON_DIR}}


def test_preflight_raises_with_every_node_result():
    fake = _FakeRay(
        {
            "10.0.0.1": _ok("node-a"),
            "10.0.0.2": _missing("node-b"),
            "10.0.0.3": _FakeRay.exceptions.RayError("worker died"),
        }
    )
    with pytest.raises(RuntimeError) as excinfo:
        supervisor_ray._preflight_ray_nsys(fake, ["10.0.0.1", "10.0.0.2", "10.0.0.3"], NSYS_BIN)
    lines = str(excinfo.value).splitlines()
    assert "failed on 2 of 3 node(s)" in lines[0]
    assert any(line.startswith("  OK 10.0.0.1 (node-a)") for line in lines)
    assert "  FAIL 10.0.0.2 (node-b): which nsys=None" in lines
    assert "  FAIL 10.0.0.3 (?): probe task failed: RayError: worker died" in lines
