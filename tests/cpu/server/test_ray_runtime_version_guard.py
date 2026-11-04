"""The AFD driver refuses Ray daemons whose package versions differ from its own, and names both envs and the fix."""

import importlib.metadata
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from minisgl.server import supervisor_ray

PYTHON_DIR = str(Path(supervisor_ray.__file__).resolve().parents[2])
NEW = {"python": "3.12.13", "torch": "2.10.0+cu130", "flashinfer-python": "0.6.11.post2",
       "flashinfer-cubin": "0.6.11.post2", "triton": "3.6.0"}
OLD = dict(NEW, **{"flashinfer-python": "0.6.6", "flashinfer-cubin": "0.6.6"})
DRIVER = {"host": "login", "prefix": "/envs/fi0611", "versions": NEW}


def _node(host, prefix="/envs/fi0611", versions=NEW):
    return {"host": host, "prefix": prefix, "versions": dict(versions)}


def test_same_env_everywhere_is_silent():
    assert supervisor_ray._runtime_version_check(DRIVER, {"10.0.0.1": _node("a"), "10.0.0.2": _node("b")}) == (
        None,
        None,
    )


def test_a_private_clone_with_the_same_versions_only_warns():
    error, warning = supervisor_ray._runtime_version_check(
        DRIVER, {"10.0.0.1": _node("a", prefix="/envs/daemon"), "10.0.0.2": _node("b")}
    )
    assert error is None
    assert warning == (
        "FastAFD driver env /envs/fi0611 is not the Ray daemons' env on 10.0.0.1 (/envs/daemon); package "
        "versions match (python=3.12.13 torch=2.10.0+cu130 flashinfer-python=0.6.11.post2 "
        "flashinfer-cubin=0.6.11.post2 triton=3.6.0), continuing."
    )


def test_other_versions_name_both_envs_every_node_and_the_fix():
    error, warning = supervisor_ray._runtime_version_check(
        DRIVER,
        {
            "10.0.0.1": _node("a"),
            "10.0.0.2": _node("b", prefix="/envs/old", versions=OLD),
            "10.0.0.3": {"error": "RayTaskError: boom"},
        },
    )
    assert warning is None
    lines = error.splitlines()
    assert lines[0].startswith("FastAFD driver and Ray worker environments differ on 2 of 3 node(s).")
    assert "raylet" in lines[0] and "PATH, LD_LIBRARY_PATH, CUDA_HOME and CONDA_PREFIX" in lines[0]
    assert lines[1] == (
        "Driver: /envs/fi0611 python=3.12.13 torch=2.10.0+cu130 flashinfer-python=0.6.11.post2 "
        "flashinfer-cubin=0.6.11.post2 triton=3.6.0"
    )
    assert lines[2].startswith("Fix: run the driver from the conda env the Ray daemons were started from")
    assert "CONDA_ENV" in lines[2]
    assert lines[4].startswith("  OK 10.0.0.1 (a): /envs/fi0611 ")
    assert lines[5].startswith("  MISMATCH 10.0.0.2 (b): /envs/old ")
    assert lines[5].endswith(
        "differs: flashinfer-python 0.6.6 (driver 0.6.11.post2), flashinfer-cubin 0.6.6 (driver 0.6.11.post2)"
    )
    assert lines[6] == "  FAIL 10.0.0.3 (?): probe task failed: RayTaskError: boom"


@pytest.mark.parametrize("field", ["python", "torch", "triton"])
def test_any_checked_version_counts(field):
    other = dict(NEW, **{field: "0"})
    error, _ = supervisor_ray._runtime_version_check(DRIVER, {"10.0.0.1": _node("a", versions=other)})
    assert error is not None and f"differs: {field} 0 (driver {NEW[field]})" in error


def _installed(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None  # e.g. flashinfer in the CPU venv: reported as None, not raised


def test_probe_reports_this_interpreter_from_package_metadata():
    result = supervisor_ray._probe_runtime_versions()
    assert result["prefix"] == sys.prefix
    assert result["versions"] == {
        "python": platform.python_version(),
        **{dist: _installed(dist) for dist in supervisor_ray._RUNTIME_VERSION_DISTS},
    }


def test_the_afd_driver_checks_before_it_creates_the_coordinator():
    # AfdBackendSupervisor.start needs a live cluster; pin the order of the two calls in its source instead.
    import inspect

    from minisgl.server import supervisor_afd

    source = inspect.getsource(supervisor_afd.AfdBackendSupervisor.start)
    assert 0 <= source.index("_preflight_runtime_versions(ray, actor_nodes)") < source.index("actor_cls.options(")


class _FakeRay:
    class exceptions:
        class RayError(Exception):
            pass

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.remote_kwargs = None
        self.submissions = []

    def nodes(self):
        return [{"Alive": True, "NodeManagerAddress": ip, "Resources": {f"node:{ip}": 1.0}} for ip in self.outcomes]

    def remote(self, **kwargs):
        self.remote_kwargs = kwargs
        fake = self

        class _RemoteFunction:
            def __init__(self, fn):
                assert fn is supervisor_ray._probe_runtime_versions

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


@pytest.fixture
def driver_is(monkeypatch):
    def set_driver(probe):
        monkeypatch.setattr(supervisor_ray, "_probe_runtime_versions", lambda: probe)

    monkeypatch.delenv("MINISGL_RAY_PY_EXECUTABLE", raising=False)
    return set_driver


def test_preflight_probes_each_node_once_with_the_raylet_interpreter(driver_is):
    # driver_is swaps the module's probe; the fake checks that this same probe is what runs on the nodes.
    driver_is(DRIVER)
    fake = _FakeRay({"10.0.0.1": _node("a"), "10.0.0.2": _node("b")})
    assert supervisor_ray._preflight_runtime_versions(fake, ["10.0.0.2", "10.0.0.1", "10.0.0.2"]) is None
    assert fake.remote_kwargs == {"num_cpus": 0, "max_retries": 0}
    assert [ip for ip, _ in fake.submissions] == ["10.0.0.1", "10.0.0.2"]
    for ip, options in fake.submissions:
        assert options["resources"] == {f"node:{ip}": 0.001}
        # No py_executable, no forwarded PATH: the task runs the interpreter the raylet gives an AFD worker.
        assert options["runtime_env"] == {"env_vars": {"PYTHONPATH": PYTHON_DIR}}


def test_preflight_returns_the_prefix_warning(driver_is):
    driver_is(DRIVER)
    fake = _FakeRay({"10.0.0.1": _node("a", prefix="/envs/daemon")})
    assert "is not the Ray daemons' env on 10.0.0.1" in supervisor_ray._preflight_runtime_versions(fake, ["10.0.0.1"])


def test_preflight_raises_on_other_versions_or_a_failed_probe(driver_is):
    driver_is(DRIVER)
    fake = _FakeRay(
        {
            "10.0.0.1": _node("a"),
            "10.0.0.2": _node("b", prefix="/envs/old", versions=OLD),
            "10.0.0.3": _FakeRay.exceptions.RayError("worker died"),
        }
    )
    with pytest.raises(RuntimeError) as excinfo:
        supervisor_ray._preflight_runtime_versions(fake, ["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    lines = str(excinfo.value).splitlines()
    assert "differ on 2 of 3 node(s)" in lines[0]
    assert "  FAIL 10.0.0.3 (?): probe task failed: RayError: worker died" in lines


def test_preflight_is_skipped_when_the_worker_interpreter_is_replaced(monkeypatch):
    monkeypatch.setenv("MINISGL_RAY_PY_EXECUTABLE", "compute-sanitizer python")

    class _NoRay:
        def remote(self, **kwargs):
            raise AssertionError("must not probe")

    warning = supervisor_ray._preflight_runtime_versions(_NoRay(), ["10.0.0.1"])
    assert warning.startswith("FastAFD driver/worker version check skipped: MINISGL_RAY_PY_EXECUTABLE")
