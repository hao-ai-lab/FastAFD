"""CPU regressions for AFD admission and coordinator progress.

The scheduler, KV ledger, plans, and completion handling are real. Only the
unused GPU worker launcher, NVTX instrumentation, and worker transport are
replaced; these tests do not execute distributed collectives or model kernels.
"""

import sys
from collections import deque
from contextlib import nullcontext
from queue import SimpleQueue
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.afd_protocol import (
    AfdAGStepReply,
    AfdEGStepReply,
    AfdFlushStepCmd,
    AfdRunAGStepCmd,
    AfdRunEGStepCmd,
    AfdTopology,
)
from minisgl.afd_scheduler import CentralizedAfdDpScheduler
from minisgl.core import SamplingParams
from minisgl.message import ExitMsg, UserMsg
from minisgl.scheduler.prefill import ChunkedReq

# The coordinator imports GPU workers eagerly, although these tests never start
# workers. Restore only this module key, preserving other imports' identities.
with pytest.MonkeyPatch.context() as import_patch:
    launcher = ModuleType("minisgl.afd_worker_launcher")
    launcher.startup_afd_workers = None
    import_patch.setitem(sys.modules, launcher.__name__, launcher)
    from minisgl import afd_coordinator


def make_scheduler(dp=0, *, pages=32, tables=4, budget=16):
    return CentralizedAfdDpScheduler(
        dp_rank=dp,
        max_running_req=tables,
        max_seq_len=64,
        num_pages=pages,
        page_size=1,
        prefill_budget=budget,
        decode_graph_bs=(1, 2, 4),
        cuda_graph_max_bs=4,
        afd_num_mb=2,
        mlp_fanout=1,
        eos_token_id=2,
    )


def enqueue(sched, uid, prompt=16, output=4):
    sched.add_reqs(
        [
            UserMsg(
                uid,
                torch.arange(prompt, dtype=torch.int32),
                SamplingParams(max_tokens=output, ignore_eos=True),
            )
        ]
    )


def complete(sched, batch):
    """Apply the same launch and fixed-token completion transitions as AFD."""
    for req in batch.reqs:
        req.complete_one()
    sched.decode_manager.filter_reqs(batch.reqs)
    return sched.complete_results(batch, [7] * len(batch.reqs))


def pressured_scheduler(dp=0):
    sched = make_scheduler(dp)
    enqueue(sched, 10 + dp)
    _, batch = sched.schedule_step(0, phase="prefill")
    complete(sched, batch)
    # 16 KV tokens remain free, but admission needs 24 plus decode reservations.
    enqueue(sched, 20 + dp, output=8)
    assert sched.cache_manager.available_size == 16
    assert sched.table_manager.available_size > 0
    return sched


@pytest.fixture
def coordinator(monkeypatch, tmp_path):
    monkeypatch.setattr(afd_coordinator, "nvtx_range", lambda *args: nullcontext())

    def make(scheds, *, overlap=False):
        coord = afd_coordinator.AfdCoordinator.__new__(afd_coordinator.AfdCoordinator)
        coord.server_args = SimpleNamespace(afd_num_mb=2, afd_disable_overlap=not overlap)
        coord._central_schedulers = scheds
        coord.attn_tp_size = 1
        coord._layout = AfdTopology(len(scheds), 1, 1, 1)
        coord._control_profile_start_step = coord._control_profile_stop_step = 0
        coord._nsys_start_step = coord._nsys_stop_step = 0
        coord._log_afd_steps = False
        coord.log_path = str(tmp_path / "coordinator.log")
        coord.enable_attention_decode_graph = coord.enable_model_decode_graph = True
        coord.decode_graph_bs = (1, 2, 4)
        coord._send_into_detokenizer = SimpleQueue()
        return coord

    return make


def prepare(coord, step=1):
    scheds = coord._central_schedulers
    return coord._afd_prepare_step_launch(step, scheds, coord._afd_next_global_phase(scheds))


def test_kv_rejected_prefill_falls_back_until_decode_releases_kv(coordinator):
    sched = pressured_scheduler()
    coord = coordinator([sched])
    waiting = sched.prefill_manager.pending_list[0]
    assert coord._afd_next_global_phase([sched]) == "prefill"

    for step in range(1, 4):
        prepared = prepare(coord, step)
        assert prepared is not None, "KV-blocked prefill must not starve runnable decode"
        _, lanes, phase, _ = prepared
        assert phase == "decode"
        assert [req.uid for req in lanes[0].batch.reqs] == [10]
        assert sched.prefill_manager.pending_list == [waiting]
        complete(sched, lanes[0].batch)

    assert sched.cache_manager.available_size == 32
    assert sched.table_manager.available_size == 4
    _, lanes, phase, _ = prepare(coord, 4)
    assert phase == "prefill"
    assert [req.uid for req in lanes[0].batch.reqs] == [20]


def test_admissible_prefill_keeps_priority(coordinator):
    sched = make_scheduler(pages=64)
    enqueue(sched, 10)
    _, batch = sched.schedule_step(0, phase="prefill")
    complete(sched, batch)
    decoding = sched.decode_manager.running_reqs[10]
    enqueue(sched, 20)
    _, lanes, phase, _ = prepare(coordinator([sched]))
    assert phase == "prefill"
    assert [req.uid for req in lanes[0].batch.reqs] == [20]
    assert decoding.cached_len == 16


def test_partial_dp_admission_keeps_global_prefill(coordinator):
    blocked, admitted = pressured_scheduler(), make_scheduler(1)
    enqueue(admitted, 30, prompt=4)
    _, lanes, phase, _ = prepare(coordinator([blocked, admitted]))
    assert phase == "prefill"
    assert [lane.is_dummy for lane in lanes] == [True, False]
    assert [req.uid for req in lanes[1].batch.reqs] == [30]
    assert blocked.decode_manager.running_reqs[10].cached_len == 16
    assert blocked.cache_manager.available_size == 16


def test_all_dp_rejected_use_uniform_decode_with_idle_dummy(coordinator):
    scheds = [pressured_scheduler(0), pressured_scheduler(1), make_scheduler(2)]
    bucket, lanes, phase, num_mb = prepare(coordinator(scheds))
    assert phase == "decode"
    assert [lane.is_dummy for lane in lanes] == [False, False, True]
    assert all(lane.model_plan.phase == "decode" for lane in lanes)
    assert all(lane.model_plan.padded_size == bucket for lane in lanes)
    assert all(lane.model_plan.num_mb == num_mb == 2 for lane in lanes)
    assert lanes[2].batch.reqs == []


@pytest.mark.parametrize("pending", [False, True])
def test_no_decode_returns_none_without_retrying_forever(coordinator, pending):
    sched = make_scheduler()
    if pending:
        enqueue(sched, 20, prompt=24, output=12)
    assert prepare(coordinator([sched])) is None
    assert bool(sched.prefill_manager.pending_list) == pending
    assert sched.cache_manager.available_size == 32


@pytest.mark.parametrize("tables", [1, 2])
def test_chunked_prefill_continues_on_its_existing_table(coordinator, tables):
    sched = make_scheduler(tables=tables, budget=4)
    enqueue(sched, 20, prompt=8)
    _, batch = sched.schedule_step(0, phase="prefill")
    assert isinstance(batch.reqs[0], ChunkedReq)
    table = batch.reqs[0].table_idx
    complete(sched, batch)
    _, lanes, phase, _ = prepare(coordinator([sched]))
    assert phase == "prefill"
    req = lanes[0].batch.reqs[0]
    assert not isinstance(req, ChunkedReq)
    assert (req.table_idx, req.cached_len, req.extend_len) == (table, 4, 4)


def test_full_table_with_decode_preserves_existing_phase_policy(coordinator):
    sched = make_scheduler(tables=2, budget=4)
    enqueue(sched, 10, prompt=4)
    _, batch = sched.schedule_step(0, phase="prefill")
    complete(sched, batch)
    enqueue(sched, 20, prompt=8)
    _, batch = sched.schedule_step(1, phase="prefill")
    complete(sched, batch)
    assert sched.table_manager.available_size == 0
    chunk = sched.prefill_manager.pending_list[0].chunked_req
    _, lanes, phase, _ = prepare(coordinator([sched]), 2)
    assert phase == "decode"
    assert [req.uid for req in lanes[0].batch.reqs] == [10]
    assert chunk.cached_len == 4


@pytest.mark.parametrize("overlap", [False, True], ids=["normal", "overlap"])
def test_control_loops_finish_blocking_decode_and_queued_prefill(coordinator, overlap):
    sched = pressured_scheduler()
    coord = coordinator([sched], overlap=overlap)
    replies, commands, broadcasts = deque(), [], []
    receive_calls = 0

    def send(ranks, cmd):
        commands.append(cmd)
        for rank in ranks:
            if isinstance(cmd, AfdRunAGStepCmd):
                replies.append(
                    AfdAGStepReply(rank, cmd.plan.step_id, next_tokens=[7] * cmd.plan.real_size)
                )
            else:
                assert isinstance(cmd, AfdRunEGStepCmd)
                replies.append(AfdEGStepReply(rank, cmd.plan.step_id))

    def receive_workers():
        assert replies, "coordinator is waiting for a step that was never launched"
        result = list(replies)
        replies.clear()
        return result

    def receive_users(blocking=False):
        nonlocal receive_calls
        receive_calls += 1
        assert receive_calls <= 30, "coordinator made no forward progress"
        return [ExitMsg()] if not sched.has_runnable_reqs() and not replies else []

    coord._send_cmd_to_workers = send
    coord._recv_worker_inbox = receive_workers
    coord._broadcast_cmd_to_workers = broadcasts.append
    coord.receive_msg = receive_users
    coord._afd_run_loop()

    ag_plans = [cmd.plan for cmd in commands if isinstance(cmd, AfdRunAGStepCmd)]
    eg_plans = [cmd.plan for cmd in commands if isinstance(cmd, AfdRunEGStepCmd)]
    assert [plan.phase for plan in ag_plans] == ["decode"] * 3 + ["prefill"] + ["decode"] * 7
    assert [(p.step_id, p.phase, p.dispatch_bucket) for p in ag_plans] == [
        (p.step_id, p.phase, p.dispatch_bucket) for p in eg_plans
    ]
    assert ag_plans[3].free_table_indices == ag_plans[0].table_indices
    emitted = []
    while not coord._send_into_detokenizer.empty():
        emitted.append(coord._send_into_detokenizer.get_nowait())
    assert [(msg.uid, msg.next_token) for msg in emitted] == [(10, 7)] * 3 + [(20, 7)] * 8
    assert [msg.uid for msg in emitted if msg.finished] == [10, 20]
    assert sched.cache_manager.available_size == 32
    assert sched.table_manager.available_size == 4
    assert not sched.has_runnable_reqs()
    if overlap:
        assert any(isinstance(cmd, AfdFlushStepCmd) for cmd in broadcasts)
