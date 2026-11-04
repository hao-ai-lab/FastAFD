"""Coordinator-side radix prefix caching for AFD (MINISGL_AFD_PREFIX_CACHE=radix): shared
prompt prefixes are matched page-aligned and granted to the workers as the same page
ids, finished requests are cached with their generated tokens, unreferenced prefixes
are evicted LRU-first when pages run out, probing a group does not touch its tree, and
admission follows a prompt's cached prefix (round-robin otherwise)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from minisgl.core import SamplingParams
from minisgl.message import UserMsg

PAGE = 4
SYS = list(range(100, 124))  # a 24-token shared system prompt: 6 pages


def _sched(afd, num_pages=64, budget=64):
    return afd.Scheduler(
        dp_rank=0, max_running_req=4, max_seq_len=128, num_pages=num_pages, page_size=PAGE,
        prefill_budget=budget, decode_graph_bs=(), cuda_graph_max_bs=0, afd_num_mb=1,
        mlp_fanout=1, eos_token_id=-1, prefix_cache="radix",
    )


def _msg(uid, ids, out):
    return UserMsg(uid=uid, input_ids=torch.tensor(ids, dtype=torch.int32),
                   sampling_params=SamplingParams(max_tokens=out, ignore_eos=True))


def _run(afd, sched, worker, steps=200, tokens=lambda step, i: 1000 + step):
    # table frees go out with the next plan, which may be the next _run's
    pending, plans = worker.pending_free, []
    for step in range(1, steps):
        if not sched.has_runnable_reqs():
            break
        phase = "prefill" if sched.prefill_manager.runnable else "decode"
        out = sched.schedule_step(step, phase=phase)
        if out is None:
            continue
        model_plan, batch = out
        plan = afd.build_plan(step, model_plan, batch, dispatch_bucket=0, free_table_indices=pending,
                              grants=sched.page_grants(batch.reqs))
        plans.append(plan)
        worker.apply(plan)
        for r in batch.reqs:
            assert torch.equal(worker.rt.state.page_table[r.table_idx, : r.device_len],
                               sched.page_table[r.table_idx, : r.device_len])
            r.complete_one()
        sched.decode_manager.filter_reqs(batch.reqs)
        done = sched.complete_results(batch, [tokens(step, i) for i in range(len(batch.reqs))])
        pending = tuple(done.freed_table_indices)
        worker.pending_free = pending
    return plans


def test_second_request_reuses_the_first_ones_prefix_pages(afd, cpu_worker):
    sched = _sched(afd)
    worker = cpu_worker(sched)
    sched.add_reqs([_msg(1, SYS + [1, 2, 3], 3)])
    plans = _run(afd, sched, worker, steps=3)  # prefill of request 1 (and a decode)
    first_pages = plans[0].grant_pages[:6]
    sched.add_reqs([_msg(2, SYS + [7, 8, 9, 10, 11], 3)])
    plans = _run(afd, sched, worker)
    pre = next(p for p in plans if p.phase == "prefill")
    assert pre.cached_lens == (24,) and pre.extend_lens == (5,)  # the 6 prefix pages are hits
    assert pre.grant_pages[:6] == first_pages  # granted as the same page ids


def test_finished_request_is_cached_with_its_output(afd, cpu_worker):
    sched = _sched(afd)
    worker = cpu_worker(sched)
    sched.add_reqs([_msg(1, SYS[:16], 9)])
    out = []
    _run(afd, sched, worker, tokens=lambda step, i: out.append(500 + step) or 500 + step)
    turn2 = SYS[:16] + out[:8] + [42]  # the next turn repeats the first turn's 8 generated tokens
    hit = sched.prefix_match_len(torch.tensor(turn2, dtype=torch.int32))
    assert hit == 24  # prompt (4 pages) + output (2 pages), page-aligned


def test_eviction_frees_unreferenced_prefixes_lru_first(afd, cpu_worker):
    sched = _sched(afd, num_pages=12)
    worker = cpu_worker(sched)
    for uid, base in ((1, 0), (2, 1000)):
        sched.add_reqs([_msg(uid, [base + i for i in range(17)], 2)])
        _run(afd, sched, worker)
    cm = sched.cache_manager
    assert cm.prefix_cache.size_info.evictable_size == 2 * 16  # both prompts cached, idle
    sched.add_reqs([_msg(3, [5000 + i for i in range(40)], 2)])  # needs 11 of 12 pages
    _run(afd, sched, worker)
    assert sched.prefix_match_len(torch.tensor([i for i in range(17)], dtype=torch.int32)) == 0
    cm.check_integrity()


def test_probe_does_not_touch_the_tree(afd, cpu_worker):
    sched = _sched(afd)
    worker = cpu_worker(sched)
    sched.add_reqs([_msg(1, SYS + [1], 2)])
    _run(afd, sched, worker)
    root = sched.cache_manager.prefix_cache.root_node
    before = [(id(n), n.timestamp, n.length) for n in root.children.values()]
    assert sched.prefix_match_len(torch.tensor(SYS[:10] + [99] * 9, dtype=torch.int32)) == 8
    assert [(id(n), n.timestamp, n.length) for n in root.children.values()] == before


def test_admission_sends_a_prompt_to_the_group_holding_its_prefix(afd, cpu_worker):
    scheds = [_sched(afd) for _ in range(3)]
    worker = cpu_worker(scheds[2])
    scheds[2].add_reqs([_msg(1, SYS + [1], 2)])
    _run(afd, scheds[2], worker)  # group 2 now caches SYS
    coord = afd.Coordinator.__new__(afd.Coordinator)
    coord._layout = SimpleNamespace(attn_dp_size=3)
    coord._central_schedulers = scheds
    coord._next_attn_dp_rr = 0
    coord._prefix_cache = "radix"
    assert coord._admit_dp(_msg(2, SYS + [5, 6], 2)) == 2  # its prefix lives there
    assert [coord._admit_dp(_msg(u, [9000 + u] * 9, 2)) for u in (3, 4, 5)] == [0, 1, 2]  # no hit: round-robin
