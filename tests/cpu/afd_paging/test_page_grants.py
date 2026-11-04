"""Explicit AFD paging: the coordinator grants page ids in every step plan, and an
attention worker that applies only the grants ends up with the coordinator's page
table, through chunked prefill, decode page crossings, finishes and slot reuse."""

from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.message import UserMsg

PAGE = 4


def _sched(afd, num_pages=64, budget=10):
    return afd.Scheduler(
        dp_rank=0, max_running_req=4, max_seq_len=128, num_pages=num_pages, page_size=PAGE,
        prefill_budget=budget, decode_graph_bs=(), cuda_graph_max_bs=0, afd_num_mb=1,
        mlp_fanout=1, eos_token_id=-1,
    )


def _msg(uid, n, out):
    return UserMsg(uid=uid, input_ids=torch.arange(1, n + 1, dtype=torch.int32),
                   sampling_params=SamplingParams(max_tokens=out, ignore_eos=True))


def _step(afd, sched, worker, step, phase, pending_free):
    out = sched.schedule_step(step, phase=phase)
    if out is None:
        return None, pending_free
    model_plan, batch = out
    plan = afd.build_plan(step, model_plan, batch, dispatch_bucket=0, free_table_indices=pending_free,
                          grants=sched.page_grants(batch.reqs))
    worker.apply(plan)
    for req in batch.reqs:  # the coordinator advances its requests once a step is out
        req.complete_one()
    sched.decode_manager.filter_reqs(batch.reqs)
    completion = sched.complete_results(batch, [7] * len(batch.reqs))
    return batch, tuple(completion.freed_table_indices)


def _assert_rows_match(sched, worker):
    for req in list(sched.decode_manager.running_reqs.values()):
        n = req.device_len
        assert torch.equal(worker.rt.state.page_table[req.table_idx, :n], sched.page_table[req.table_idx, :n])


def test_worker_follows_grants_through_prefill_decode_and_reuse(afd, cpu_worker):
    sched = _sched(afd)
    worker = cpu_worker(sched)
    sched.add_reqs([_msg(1, 23, 9), _msg(2, 5, 30)])
    pending, step = (), 0
    for _ in range(80):
        step += 1
        phase = "prefill" if sched.prefill_manager.runnable else "decode"
        batch, freed = _step(afd, sched, worker, step, phase, pending)
        pending = freed if batch is not None else pending
        _assert_rows_match(sched, worker)
        if step == 20:  # a newcomer reuses a finished request's table slot
            sched.add_reqs([_msg(3, 17, 6)])
        if not sched.has_runnable_reqs():
            break
    assert not sched.has_runnable_reqs()
    # every granted page came from the coordinator: the worker's own list is untouched
    assert len(worker.rt._free_pages) == sched.cache_manager.num_pages


def test_grants_cover_each_page_once_and_restart_after_free(afd):
    sched = _sched(afd, budget=8)
    sched.add_reqs([_msg(5, 13, 2)])
    seen = []
    out = sched.schedule_step(1, phase="prefill")
    _, batch = out
    cached, counts, pages = sched.page_grants(batch.reqs)
    assert cached == (0,) and counts == (2,)  # 8 tokens -> 2 pages
    seen += pages
    assert sched.page_grants(batch.reqs) == ((0,), (0,), ())  # nothing twice
    for req in batch.reqs:
        req.complete_one()
    sched.decode_manager.filter_reqs(batch.reqs)
    sched.complete_results(batch, [7])
    _, batch = sched.schedule_step(2, phase="prefill")
    cached, counts, pages = sched.page_grants(batch.reqs)
    assert cached == (8,) and counts == (2,)  # tokens 8..13 -> pages 2, 3
    seen += pages
    assert len(set(seen)) == 4
    t = int(batch.reqs[0].table_idx)
    sched.forget_granted(t)
    assert sched._granted_pages.get(t) is None
