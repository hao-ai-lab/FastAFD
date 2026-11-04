"""An AFD first wave that cannot admit every request decodes on and admits the rest later (vLLM semantics)."""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.message import ExitMsg, UserMsg

PAGE_SIZE = 4
PROMPT_TOKENS = 8
MAX_TOKENS = 4  # each request needs (8 + 4) / 4 = 3 pages
REQS_PER_LANE = 3
# 7 pages admit two requests per lane; the third waits for pages that only a
# finished request could free. 10 pages admit all three. 2 pages (8 tokens) can
# never hold one 12-token request.
STALL_POOL_PAGES = 7
FIT_POOL_PAGES = 10
NEVER_FIT_POOL_PAGES = 2


class _Spinning(Exception):
    """The loop kept polling for messages without scheduling anything."""


@pytest.fixture
def afd(monkeypatch):
    """AfdCoordinator and its lane scheduler, imported with stub Ray and worker-launcher modules."""
    before = set(sys.modules)
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    launcher = ModuleType("minisgl.afd_worker_launcher")
    launcher.startup_afd_workers = None
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        patch.setitem(sys.modules, "minisgl.afd_worker_launcher", launcher)
        import minisgl.afd_coordinator as coordinator_module
    # CPU torch has no NVTX; the ranges only annotate traces.
    monkeypatch.setattr(coordinator_module, "nvtx_range", contextmanager(lambda name: (yield)))
    yield SimpleNamespace(
        Coordinator=coordinator_module.AfdCoordinator,
        Scheduler=coordinator_module.CentralizedAfdDpScheduler,
    )
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


def _lanes(afd, num_pages: int, num_lanes: int = 2) -> list:
    lanes = [
        afd.Scheduler(
            dp_rank=dp,
            max_running_req=REQS_PER_LANE,
            max_seq_len=64,
            num_pages=num_pages,
            page_size=PAGE_SIZE,
            prefill_budget=64,
            decode_graph_bs=(),
            cuda_graph_max_bs=0,
            afd_num_mb=1,
            mlp_fanout=1,
            eos_token_id=0,
        )
        for dp in range(num_lanes)
    ]
    for uid in range(REQS_PER_LANE * num_lanes):
        lanes[uid % num_lanes].add_reqs([
            UserMsg(
                uid=uid,
                input_ids=torch.zeros(PROMPT_TOKENS, dtype=torch.int32),
                sampling_params=SamplingParams(ignore_eos=True, max_tokens=MAX_TOKENS),
            )
        ])
    return lanes


def _coordinator(afd, lanes, tmp_path, *, disable_overlap: bool):
    """The real scheduling loops, with the Ray worker queues replaced by the ledger updates they cause."""
    coord = afd.Coordinator.__new__(afd.Coordinator)
    coord._central_schedulers = lanes
    coord.attn_tp_size = 1
    coord.async_pending_steps = 2
    coord.log_path = str(tmp_path / "afd_coordinator.log")
    coord.server_args = SimpleNamespace(afd_disable_overlap=disable_overlap, afd_num_mb=1)
    coord._control_profile_start_step = 0
    coord._control_profile_stop_step = 0
    coord._nsys_start_step = 0
    coord._nsys_stop_step = 0
    coord._shutdown_requested = False
    coord._worker_stop_lock = threading.Lock()
    coord._worker_stop_sent = False
    coord._afd_admission_blocked = False
    coord.finished_uids = []
    coord.decode_sizes = [[] for _ in lanes]  # real requests per decode step, per lane
    coord.polls = 0

    def receive_msg(blocking: bool = False):
        coord.polls += 1
        if blocking:  # idle: every request has finished
            return [ExitMsg()]
        if coord.polls > 200:
            raise _Spinning(f"{coord.polls} polls without progress")
        return []

    def launch_step(*, step_id, dp_batches, scheds, phase, **_):
        # What _afd_launch_step does to the ledger once the AG/EG commands are sent.
        coord.polls = 0
        items = []
        for item in dp_batches:
            if phase == "decode" and not item.is_dummy:
                coord.decode_sizes[item.dp_rank].append(len(item.batch.reqs))
            if not item.is_dummy:
                for req in item.batch.reqs:
                    req.complete_one()
                scheds[item.dp_rank].decode_manager.filter_reqs(item.batch.reqs)
            items.append((item.dp_rank, item.batch, ()))
        return items

    def complete_global(pending_item, scheds, forward_remains, attn_tp):
        _step_id, items = pending_item
        freed = {}
        for dp, batch, _gens in items:
            done = scheds[dp].complete_results(batch, [0] * len(batch.reqs))
            if done.reply_uids is not None:  # decode fast path
                coord.finished_uids += [
                    uid for uid, fin in zip(done.reply_uids, done.reply_finished) if fin
                ]
            else:
                coord.finished_uids += [reply.uid for reply in done.replies if reply.finished]
            freed[dp] = done.freed_table_indices
        return freed

    coord.receive_msg = receive_msg
    coord._afd_launch_step = launch_step
    coord._afd_complete_global = complete_global
    coord._broadcast_cmd_to_workers = lambda cmd: None
    return coord


def _prefill_until_blocked(lanes) -> None:
    """Run lockstep prefill steps (no decode) until no lane admits anything."""
    for step_id in range(1, 100):
        results = [lane.schedule_step(step_id, phase="prefill") for lane in lanes]
        if all(result is None for result in results):
            return
        for lane, result in zip(lanes, results):
            if result is not None:
                batch = result[1]
                for req in batch.reqs:
                    req.complete_one()
                lane.decode_manager.filter_reqs(batch.reqs)
                lane.complete_results(batch, [0] * len(batch.reqs))
    raise AssertionError("prefill never stopped admitting")


def test_page_blocked_waiting_request_keeps_the_global_phase_on_prefill(afd):
    """The stall itself, on the real scheduler: prefill keeps priority but admits nothing."""
    lanes = _lanes(afd, STALL_POOL_PAGES)
    _prefill_until_blocked(lanes)

    for lane in lanes:
        assert len(lane.decode_manager.running_reqs) == 2  # could decode and free pages
        assert len(lane.prefill_manager.pending_list) == 1
        assert lane.table_manager.available_size == 1
    # The same state every time the loop asks: prefill is chosen, nothing is admitted.
    for step_id in (100, 101):
        assert afd.Coordinator._afd_next_global_phase(lanes) == "prefill"
        assert all(lane.schedule_step(step_id, phase="prefill") is None for lane in lanes)


def test_describe_waiting_admission_reports_the_admission_numbers(afd):
    lanes = _lanes(afd, STALL_POOL_PAGES, num_lanes=1)
    assert lanes[0].describe_waiting_admission() == (
        "waiting=3 running=0 free_tables=3 head_request_tokens=12 "
        "reserved_running_tokens=0 free_tokens=28 pool_pages=7 page_size=4"
    )
    lanes[0].prefill_manager.pending_list.clear()
    assert lanes[0].describe_waiting_admission() is None


def _dedup(sizes: list[int]) -> list[int]:
    return [n for i, n in enumerate(sizes) if i == 0 or sizes[i - 1] != n]


@pytest.mark.parametrize("disable_overlap", [False, True], ids=["overlap_loop", "normal_loop"])
def test_loop_decodes_the_first_wave_then_admits_the_waiting_request(afd, tmp_path, disable_overlap):
    lanes = _lanes(afd, STALL_POOL_PAGES)
    coord = _coordinator(afd, lanes, tmp_path, disable_overlap=disable_overlap)

    coord._afd_run_loop()  # returns on the ExitMsg delivered once the loop goes idle

    assert sorted(coord.finished_uids) == list(range(REQS_PER_LANE * len(lanes)))
    assert all(len(lane.cache_manager.free_slots) == STALL_POOL_PAGES for lane in lanes)
    # Two requests decode together, then the waiting one runs as a second wave.
    assert all(_dedup(sizes) == [2, 1] for sizes in coord.decode_sizes)
    blocked = [
        line for line in (tmp_path / "afd_coordinator.log").read_text().splitlines()
        if "admission_blocked" in line
    ]
    assert len(blocked) == 1  # one line per blocked episode, not per decode step
    for dp in (0, 1):
        assert f"dp{dp}: waiting=1 running=2 free_tables=1 head_request_tokens=12" in blocked[0]


@pytest.mark.parametrize("disable_overlap", [False, True], ids=["overlap_loop", "normal_loop"])
def test_loop_raises_when_a_request_can_never_fit_the_pool(afd, tmp_path, disable_overlap):
    lanes = _lanes(afd, NEVER_FIT_POOL_PAGES)
    coord = _coordinator(afd, lanes, tmp_path, disable_overlap=disable_overlap)

    with pytest.raises(RuntimeError, match="AFD capacity shortfall") as raised:
        coord._afd_run_loop()

    message = str(raised.value)
    assert "On lanes [0, 1]" in message
    for dp in (0, 1):
        assert f"dp{dp}: waiting=3 running=0 free_tables=3 head_request_tokens=12" in message
    assert "pool_pages=2 page_size=4" in message
    assert "AFD capacity shortfall" in (tmp_path / "afd_coordinator.log").read_text()


def test_loop_raises_on_a_page_ledger_that_leaked_its_pages(afd, tmp_path):
    lanes = _lanes(afd, FIT_POOL_PAGES)
    for lane in lanes:  # pages no request owns: the pool fits, the free list does not
        while len(lane.cache_manager.free_slots) > 2:  # a list or, under MINISGL_AFD_PAGE_FAST, a deque
            lane.cache_manager.free_slots.pop()
    coord = _coordinator(afd, lanes, tmp_path, disable_overlap=False)

    with pytest.raises(RuntimeError, match="cannot make progress"):
        coord._afd_run_loop()


@pytest.mark.parametrize("disable_overlap", [False, True], ids=["overlap_loop", "normal_loop"])
def test_loop_serves_every_request_when_the_first_wave_fits(afd, tmp_path, disable_overlap):
    lanes = _lanes(afd, FIT_POOL_PAGES)
    coord = _coordinator(afd, lanes, tmp_path, disable_overlap=disable_overlap)

    coord._afd_run_loop()  # returns on the ExitMsg delivered once the loop goes idle

    assert sorted(coord.finished_uids) == list(range(REQS_PER_LANE * len(lanes)))
    assert all(len(lane.cache_manager.free_slots) == FIT_POOL_PAGES for lane in lanes)
    assert all(_dedup(sizes) == [3] for sizes in coord.decode_sizes)
    assert "admission_blocked" not in (tmp_path / "afd_coordinator.log").read_text()


@pytest.mark.parametrize("disable_overlap", [False, True], ids=["overlap_loop", "normal_loop"])
def test_step_times_log_one_line_per_timed_step(afd, tmp_path, disable_overlap):
    """MINISGL_AFD_STEP_TIMES: the synchronous loop times every step; the overlap loop
    times the interval between completions, so its first step after idling is not timed."""
    import re

    lanes = _lanes(afd, FIT_POOL_PAGES)
    coord = _coordinator(afd, lanes, tmp_path, disable_overlap=disable_overlap)
    coord._log_step_times = True
    coord.page_size = PAGE_SIZE
    launched = []
    launch = coord._afd_launch_step

    def counting_launch(**kw):
        launched.append(kw["step_id"])
        return launch(**kw)

    coord._afd_launch_step = counting_launch
    coord._afd_run_loop()

    lines = [ln for ln in open(coord.log_path) if "[afd-step-time]" in ln]
    pattern = re.compile(r"step=(\d+) phase=(prefill|decode) bucket=\d+ num_mb=1 ms=([\d.]+) "
                         r"real=\[(\d+), (\d+)\] ctx=\[\d+, \d+\] group_pages=\[\d+, \d+\]")
    parsed = [pattern.search(ln) for ln in lines]
    assert all(parsed), lines
    expected = len(launched) - (0 if disable_overlap else 1)
    assert len(lines) == expected
    assert all(float(m.group(3)) >= 0 for m in parsed)
    # every lane runs all REQS_PER_LANE requests together: the decode lines show them
    assert any(m.group(2) == "decode" and m.group(4) == str(REQS_PER_LANE) for m in parsed)


def test_step_times_are_off_by_default(afd):
    assert afd.Coordinator._log_step_times is False
