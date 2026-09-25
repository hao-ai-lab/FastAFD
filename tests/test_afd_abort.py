"""CPU coverage of abort propagation and attention-side table reuse.

Scheduling, generation filtering, plans, and runtime page/request ownership are
real. Worker transport returns fixed tokens; CUDA contexts and GPU metadata
staging are replaced. This does not exercise model kernels or collectives.
"""

import sys
from collections import deque
from contextlib import nullcontext
from queue import SimpleQueue
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl import afd_attention_runtime
from minisgl.afd_protocol import (
    AfdAGStepReply,
    AfdEGStepReply,
    AfdRunAGStepCmd,
    AfdRunEGStepCmd,
    AfdTopology,
)
from minisgl.afd_scheduler import CentralizedAfdDpScheduler
from minisgl.core import SamplingParams
from minisgl.message import AbortBackendMsg, BatchBackendMsg, ExitMsg, UserMsg
from minisgl.scheduler.prefill import ChunkedReq

with pytest.MonkeyPatch.context() as import_patch:
    launcher = ModuleType("minisgl.afd_worker_launcher")
    launcher.startup_afd_workers = None
    import_patch.setitem(sys.modules, launcher.__name__, launcher)
    from minisgl import afd_coordinator


def user(uid, prompt=8, output=8):
    return UserMsg(
        uid,
        torch.arange(prompt, dtype=torch.int32),
        SamplingParams(max_tokens=output, ignore_eos=True),
    )


def cpu_runtime(sched):
    runtime = afd_attention_runtime.AfdAttentionRuntime.__new__(
        afd_attention_runtime.AfdAttentionRuntime
    )
    runtime.config = SimpleNamespace(page_size=1)
    runtime.state = SimpleNamespace(device=torch.device("cpu"), page_table=sched.page_table.clone())
    runtime.stream = None
    runtime._free_pages = list(range(32))
    runtime._table_cached_len, runtime._table_pages, runtime._table_reqs = {}, {}, {}
    runtime._req_input_ids = torch.zeros(1, dtype=torch.int32)
    runtime._dummy_req = sched._dummy_req
    runtime.token_pool = torch.zeros_like(sched.page_table)
    runtime._page_to_token_tensor = lambda pages: torch.tensor(pages, dtype=torch.int32)

    def input_mapping(batch, plan):
        tables, positions = [], []
        for req in batch.padded_reqs:
            tables.extend([req.table_idx] * req.extend_len)
            positions.extend(range(req.cached_len, req.device_len))
        return 0, torch.tensor(tables), torch.tensor(positions)

    # Attention/sampling metadata normally requires CUDA staging and a backend.
    # The actual materialize entrypoint still frees tables, writes token blocks,
    # allocates pages, and rebuilds Req objects before reaching these adapters.
    runtime._attach_input_mapping = input_mapping
    runtime._attach_attention_metadata = lambda batch, plan: None
    runtime._record_input_mapping_stage = lambda stage: None
    runtime._attach_sampling_metadata = lambda batch, plan: None
    return runtime


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.setattr(afd_coordinator, "nvtx_range", lambda *args: nullcontext())
    monkeypatch.setattr(afd_attention_runtime, "nvtx_range", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())

    def make(n_dp=1, *, budget=32, overlap=False):
        scheds = [
            CentralizedAfdDpScheduler(
                dp_rank=dp,
                max_running_req=1,
                max_seq_len=64,
                num_pages=32,
                page_size=1,
                prefill_budget=budget,
                decode_graph_bs=(1,),
                cuda_graph_max_bs=1,
                afd_num_mb=1,
                mlp_fanout=1,
                eos_token_id=2,
            )
            for dp in range(n_dp)
        ]
        coord = afd_coordinator.AfdCoordinator.__new__(afd_coordinator.AfdCoordinator)
        coord.server_args = SimpleNamespace(afd_num_mb=1, afd_disable_overlap=not overlap)
        coord._central_schedulers = scheds
        coord.attn_tp_size, coord.max_seq_len = 1, 64
        coord._layout = AfdTopology(n_dp, 1, 1, 1)
        coord._control_profile_start_step = coord._control_profile_stop_step = 0
        coord._nsys_start_step = coord._nsys_stop_step = 0
        coord._log_afd_steps = False
        coord.log_path = str(tmp_path / "coordinator.log")
        coord.enable_attention_decode_graph = coord.enable_model_decode_graph = False
        coord.decode_graph_bs = ()
        coord._uid_to_attn_dp, coord._next_attn_dp_rr = {}, 0
        coord._afd_reply_seen, coord._afd_reply_tokens_by_dp = {}, {}
        coord._send_into_detokenizer = SimpleQueue()
        state = SimpleNamespace(
            coord=coord,
            scheds=scheds,
            runtimes=[cpu_runtime(s) for s in scheds],
            generations=[afd_coordinator._AfdForwardRemain() for _ in scheds],
            pending_free=dict.fromkeys(range(n_dp), ()),
            commands=[],
            replies=deque(),
            materialized=[],
            broadcasts=[],
        )

        def send(ranks, cmd):
            state.commands.append(cmd)
            for rank in ranks:
                if isinstance(cmd, AfdRunAGStepCmd):
                    batch = state.runtimes[rank].materialize_ag_plan(cmd.plan)
                    state.materialized.append((rank, cmd.plan.step_id, batch))
                    state.replies.append(
                        AfdAGStepReply(rank, cmd.plan.step_id, next_tokens=[7] * cmd.plan.real_size)
                    )
                else:
                    assert isinstance(cmd, AfdRunEGStepCmd)
                    state.replies.append(AfdEGStepReply(rank, cmd.plan.step_id))

        def receive():
            assert state.replies, "waiting for a command that was never sent"
            replies = list(state.replies)
            state.replies.clear()
            return replies

        coord._send_cmd_to_workers = send
        coord._recv_worker_inbox = receive
        coord._broadcast_cmd_to_workers = state.broadcasts.append
        return state

    return make


def process(state, msg):
    return state.coord._process_one_msg(msg, state.generations, state.pending_free)


def launch(state, step):
    coord = state.coord
    bucket, lanes, phase, num_mb = coord._afd_prepare_step_launch(
        step, state.scheds, coord._afd_next_global_phase(state.scheds)
    )
    return coord._afd_launch_step(
        step_id=step,
        attn_tp=1,
        scheds=state.scheds,
        forward_remains=state.generations,
        pending_free=state.pending_free,
        bucket=bucket,
        dp_batches=lanes,
        phase=phase,
        num_mb=num_mb,
    )


def complete(state, step, items):
    return state.coord._afd_complete_global((step, items), state.scheds, state.generations, 1)


@pytest.mark.parametrize("old_phase", ["prefill", "decode"])
def test_abort_reuse_frees_worker_state_before_new_materialization(rig, old_phase):
    state = rig()
    process(state, user(10))
    old_step, old_items = 1, launch(state, 1)
    if old_phase == "decode":
        complete(state, 1, old_items)
        old_step, old_items = 2, launch(state, 2)
    sched, runtime = state.scheds[0], state.runtimes[0]
    old_req = old_items[0][1].reqs[0]
    table = old_req.table_idx
    old_worker_req = runtime._table_reqs[table]
    assert runtime._table_cached_len[table] >= 8

    process(state, AbortBackendMsg(10))
    assert state.pending_free[0] == (table,)
    assert state.generations[0].generation(table) == 1
    assert sched.cache_manager.available_size == 32
    process(state, user(20))
    new_items = launch(state, old_step + 1)
    plan = [cmd.plan for cmd in state.commands if isinstance(cmd, AfdRunAGStepCmd)][-1]
    assert plan.table_indices == plan.free_table_indices == (table,)
    assert state.pending_free[0] == ()
    new_worker_req = runtime._table_reqs[table]
    assert new_worker_req is not old_worker_req
    assert (new_worker_req.cached_len, new_worker_req.device_len) == (0, 8)
    assert runtime._table_cached_len[table] == 8
    assert len(runtime._table_pages[table]) == 8
    assert len(runtime._free_pages) == len(set(runtime._free_pages)) == 24
    assert state.materialized[-1][2].input_ids.tolist() == list(range(8))

    # The old reply was already in flight when the table was reused. It must
    # neither emit an aborted token nor free the new request's central state.
    assert complete(state, old_step, old_items) == {0: ()}
    assert old_items[0][1].reqs == []
    assert sched.decode_manager.running_reqs[20] is new_items[0][1].reqs[0]
    assert sched.table_manager.available_size == 0
    assert sched.cache_manager.available_size == 24


@pytest.mark.parametrize("mapped", [True, False])
def test_abort_is_queued_only_on_owning_dp_and_preserves_earlier_frees(rig, mapped):
    state = rig(2)
    process(state, BatchBackendMsg([user(10), user(20)]))
    items = launch(state, 1)
    table = items[1][1].reqs[0].table_idx
    state.pending_free[1] = (99,)
    if not mapped:
        state.coord._uid_to_attn_dp.clear()
    process(state, AbortBackendMsg(20))
    assert state.pending_free == {0: (), 1: (99, table)}
    assert state.generations[0].generation(table) == 0
    assert state.generations[1].generation(table) == 1
    assert state.scheds[0].decode_manager.runnable
    process(state, AbortBackendMsg(20))
    assert state.pending_free[1] == (99, table)
    assert state.generations[1].generation(table) == 1


def test_unadmitted_and_unknown_aborts_do_not_queue_worker_frees(rig):
    state = rig()
    process(state, user(10))
    process(state, AbortBackendMsg(10))
    process(state, AbortBackendMsg(999))
    assert state.pending_free == {0: ()}
    assert state.generations[0].empty()
    assert not state.scheds[0].has_runnable_reqs()


def test_chunked_abort_releases_its_existing_table_once(rig):
    state = rig(budget=4)
    process(state, user(10))
    items = launch(state, 1)
    req = items[0][1].reqs[0]
    assert isinstance(req, ChunkedReq)
    process(state, AbortBackendMsg(10))
    process(state, AbortBackendMsg(10))
    assert state.pending_free[0] == (req.table_idx,)
    assert not state.scheds[0].has_runnable_reqs()
    assert state.scheds[0].cache_manager.available_size == 32
    assert complete(state, 1, items) == {0: ()}


def test_nested_messages_share_abort_queue_and_keep_user_order(rig):
    state = rig()
    process(state, user(10))
    old = launch(state, 1)
    table = old[0][1].reqs[0].table_idx
    process(
        state,
        BatchBackendMsg(
            [
                BatchBackendMsg([AbortBackendMsg(10), user(20)]),
                BatchBackendMsg([user(30), BatchBackendMsg([AbortBackendMsg(30)])]),
            ]
        ),
    )
    assert state.pending_free[0] == (table,)
    assert [req.uid for req in state.scheds[0].prefill_manager.pending_list] == [20]
    launch(state, 2)
    plan = [cmd.plan for cmd in state.commands if isinstance(cmd, AfdRunAGStepCmd)][-1]
    assert plan.free_table_indices == (table,)


@pytest.mark.parametrize("overlap", [False, True], ids=["normal", "overlap"])
def test_control_loop_passes_abort_queue_through_to_next_plan(rig, overlap):
    state = rig(overlap=overlap)
    calls = 0

    def receive(blocking=False):
        nonlocal calls
        calls += 1
        assert calls <= 30, "coordinator did not finish the replacement request"
        if calls == 1:
            return [user(10, output=16)]
        if calls == 2:
            return [BatchBackendMsg([AbortBackendMsg(10), user(20, output=3)])]
        if not state.scheds[0].has_runnable_reqs() and not state.replies:
            return [ExitMsg()]
        return []

    state.coord.receive_msg = receive
    state.coord._afd_run_loop()
    plans = [cmd.plan for cmd in state.commands if isinstance(cmd, AfdRunAGStepCmd)]
    reused = [plan for plan in plans if plan.free_table_indices]
    assert len(reused) == 1
    assert reused[0].phase == "prefill"
    assert reused[0].free_table_indices == reused[0].table_indices
    replacement = next(batch for _, step, batch in state.materialized if step == reused[0].step_id)
    # Req objects advance on later decode plans; the first token span still
    # records whether the replacement prompt read positions zero through seven.
    assert replacement.input_ids.tolist() == list(range(8))
    emitted = []
    while not state.coord._send_into_detokenizer.empty():
        emitted.append(state.coord._send_into_detokenizer.get_nowait())
    assert [(msg.next_token, msg.finished) for msg in emitted if msg.uid == 20] == [
        (7, False),
        (7, False),
        (7, True),
    ]
    assert sum(msg.uid == 10 for msg in emitted) == 1
    assert state.scheds[0].cache_manager.available_size == 32
    assert state.scheds[0].table_manager.available_size == 1
