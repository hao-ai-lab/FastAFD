"""Exercise the real stop helper with no Ray/GPU/socket dependencies."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest


def _coordinator():
    path = Path(__file__).resolve().parents[3] / "python/minisgl/afd_coordinator.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "AfdCoordinator")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_stop_workers_once")
    scope = {"AfdStopCmd": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    coord = SimpleNamespace(_worker_stop_lock=Lock(), _worker_stop_sent=False, sends=[])
    coord._broadcast_cmd_to_workers = coord.sends.append
    return coord, scope["_stop_workers_once"]


def test_exit_message_then_actor_shutdown_sends_only_one_stop():
    coord, stop = _coordinator()
    stop(coord)
    stop(coord)
    assert len(coord.sends) == 1
    assert coord._worker_stop_sent


def test_concurrent_shutdown_callers_send_only_once():
    coord, stop = _coordinator()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: stop(coord), range(32)))
    assert len(coord.sends) == 1


def test_failed_send_is_not_marked_delivered():
    coord, stop = _coordinator()
    def fail(_):
        raise RuntimeError("no worker")
    coord._broadcast_cmd_to_workers = fail
    with pytest.raises(RuntimeError, match="no worker"):
        stop(coord)
    assert not coord._worker_stop_sent
