"""Keep the CPU lane source-bound and honest about skipped coverage."""

import os
import sys
from pathlib import Path

import pytest

_collection_skips = []


def pytest_sessionstart(session):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise pytest.UsageError("Run CPU tests through make test-cpu (CUDA must be hidden).")
    import minisgl.core
    import torch

    root = Path(__file__).resolve().parents[2]
    actual = Path(minisgl.core.__file__).resolve()
    expected = root / "python/minisgl/core.py"
    if actual != expected:
        raise pytest.UsageError(f"Wrong source tree: {actual}; expected {expected}")
    torch.set_num_threads(1)
    _collection_skips.clear()
    session.config._fastafd_cpu_skips = []
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"CPU source: {actual}; interpreter: {sys.executable}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped or hasattr(report, "wasxfail"):
        item.config._fastafd_cpu_skips.append(item.nodeid)


def pytest_collectreport(report):
    if report.skipped:
        _collection_skips.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    skips = getattr(session.config, "_fastafd_cpu_skips", []) + _collection_skips
    if skips:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("CPU tests may not silently skip/xfail: " + ", ".join(skips))
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
