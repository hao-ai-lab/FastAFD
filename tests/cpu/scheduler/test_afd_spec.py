"""AFD speculative decoding: greedy acceptance, the reply encoding, page lookahead, verify scheduling."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from minisgl.spec.config import greedy_accept, pack_spec_reply, unpack_spec_reply


@pytest.fixture
def afd_scheduler(monkeypatch):
    """minisgl.afd_scheduler, imported with a stub Ray (the CPU lane has none)."""
    before = set(sys.modules)
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        import minisgl.afd_scheduler as module
    yield module
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


def test_greedy_accept_takes_the_matching_draft_prefix_and_the_target_bonus():
    # block = [bonus, d1, d2, d3]; posterior[j] is the target's token after input j.
    block_ids = torch.tensor([[5, 7, 8, 9], [5, 1, 2, 3], [5, 7, 8, 9]])
    posterior = torch.tensor([[7, 8, 9, 4], [6, 2, 3, 4], [7, 0, 9, 4]])
    n, emitted = greedy_accept(block_ids, posterior)
    assert n.tolist() == [4, 1, 2]
    assert emitted[0].tolist() == [7, 8, 9, 4]  # all drafts, then the bonus after d3
    assert emitted[1, :1].tolist() == [6]  # first draft rejected: the target's own token
    assert emitted[2, :2].tolist() == [7, 0]  # d1 accepted, d2 rejected -> posterior[1]


def test_simulated_acceptance_has_the_requested_mean_and_keeps_the_bonus_last():
    gen = torch.Generator().manual_seed(0)
    block_ids = torch.arange(8).repeat(20000, 1)
    posterior = torch.full((20000, 8), -1)
    n, emitted = greedy_accept(block_ids, posterior, sim_accept=3.4, generator=gen)
    assert abs(n.float().mean().item() - 3.4) < 0.02
    assert set(n.unique().tolist()) == {3, 4}
    rows = torch.arange(20000)
    assert (emitted[rows, n - 1] == -1).all()


def test_reply_round_trips_only_the_committed_tokens():
    n = torch.tensor([1, 3])
    emitted = torch.tensor([[11, 0, 0], [21, 22, 23]])
    assert unpack_spec_reply(pack_spec_reply(n, emitted).tolist(), 3) == [[11], [21, 22, 23]]
    with pytest.raises(ValueError):
        unpack_spec_reply([0, 1, 2, 3], 3)


def test_ledger_allocates_the_lookahead_and_frees_every_held_page_after_rollback(afd_scheduler):
    ledger = afd_scheduler._CpuNaiveCacheManager(16, 4, torch.zeros((2, 64), dtype=torch.int32))
    req = SimpleNamespace(table_idx=0, cached_len=0, device_len=6)
    ledger.allocate_paged([req], lookahead=4)  # positions [0, 10) -> 3 pages
    assert ledger.held_pages(0) == 3 and len(ledger.free_slots) == 13
    # A verify step that commits only part of its block rolls cached_len back
    # below the held pages; the next step starts from what is held.
    req.cached_len, req.device_len = 7, 11
    ledger.allocate_paged([req], lookahead=4)  # through position 15 -> 4 pages
    assert ledger.held_pages(0) == 4 and len(ledger.free_slots) == 12
    req.cached_len = 8
    ledger.cache_req(req, finished=True)
    assert len(ledger.free_slots) == 16 and ledger.held_pages(0) == 0


def _spec_scheduler(afd_scheduler, monkeypatch, *, block=4, eos=7, prefill_budget=256):
    monkeypatch.setenv("MINISGL_AFD_SPEC", "eagle3")
    monkeypatch.setenv("MINISGL_AFD_SPEC_BLOCK", str(block))
    monkeypatch.setenv("MINISGL_AFD_SPEC_DRAFT", "unused")
    return afd_scheduler.CentralizedAfdDpScheduler(
        dp_rank=0, max_running_req=4, max_seq_len=256, num_pages=64, page_size=4,
        prefill_budget=prefill_budget, decode_graph_bs=(1, 2, 4), cuda_graph_max_bs=4, afd_num_mb=1,
        mlp_fanout=1, eos_token_id=eos,
    )


def _running_req(sched, uid, prompt_len, max_tokens):
    from minisgl.core import SamplingParams
    from minisgl.message import UserMsg

    sched.add_reqs([UserMsg(uid=uid, input_ids=torch.arange(prompt_len, dtype=torch.int32),
                            sampling_params=SamplingParams(max_tokens=max_tokens))])
    plan, batch = sched.schedule_step(1, phase="prefill")
    for req in batch.reqs:
        req.complete_one()
    sched.decode_manager.filter_reqs(batch.reqs)
    sched.complete_results(batch, [11] * len(batch.reqs))
    return batch.reqs[0]


def test_verify_step_pads_to_a_graph_slot_and_allocates_past_inflight_steps(afd_scheduler, monkeypatch):
    sched = _spec_scheduler(afd_scheduler, monkeypatch, block=4)
    req = _running_req(sched, uid=1, prompt_len=6, max_tokens=40)
    req.afd_spec_inflight = 1  # one verify step still in flight (overlap loop)
    plan, batch = sched.schedule_step(2, phase="decode", verify_slot=2)
    assert batch.afd_spec_verify and plan.spec_verify
    assert [r.extend_len for r in batch.padded_reqs] == [4, 4]  # real + block-sized dummy
    # cached 6, block 4, lookahead 4 x (1 + 1 in flight): through position 18 -> 5 pages
    assert sched.cache_manager.held_pages(req.table_idx) == 5


def test_chunked_prefill_keeps_the_lookahead_page_of_a_page_aligned_chunk(afd_scheduler, monkeypatch):
    # Chunks of 8 tokens end on page boundaries (page 4), so each chunk's lookahead
    # page is the next chunk's first page. Every chunk is a new ChunkedReq; the
    # ledger must count what the table slot holds, not what the object saw.
    from minisgl.core import SamplingParams
    from minisgl.message import UserMsg

    sched = _spec_scheduler(afd_scheduler, monkeypatch, block=4, prefill_budget=8)
    sched.add_reqs([UserMsg(uid=1, input_ids=torch.arange(24, dtype=torch.int32),
                            sampling_params=SamplingParams(max_tokens=8))])
    for step in range(1, 4):  # 3 chunks of 8
        plan, batch = sched.schedule_step(step, phase="prefill")
        for req in batch.reqs:
            req.complete_one()
        sched.decode_manager.filter_reqs(batch.reqs)
        sched.complete_results(batch, [11] * len(batch.reqs))
    (req,) = sched.decode_manager.running_reqs.values()
    # prompt 24 + lookahead 4 -> positions [0, 28) -> 7 pages, and no more taken
    assert sched.cache_manager.held_pages(req.table_idx) == 7
    assert len(sched.cache_manager.free_slots) == 64 - 7
    sched.abort_req(1)
    assert len(sched.cache_manager.free_slots) == 64


def test_spec_completion_emits_tokens_until_eos_and_frees_the_request(afd_scheduler, monkeypatch):
    sched = _spec_scheduler(afd_scheduler, monkeypatch, block=4, eos=7)
    req = _running_req(sched, uid=1, prompt_len=6, max_tokens=40)
    _, batch = sched.schedule_step(2, phase="decode")
    free_before = len(sched.cache_manager.free_slots)
    reply = pack_spec_reply(torch.tensor([4]), torch.tensor([[21, 7, 23, 24]])).tolist()
    done = sched.complete_results(batch, reply)
    assert done.reply_next_tokens == [21, 7] and done.reply_finished == [False, True]
    assert done.freed_table_indices == (int(req.table_idx),)
    assert len(sched.cache_manager.free_slots) > free_before


def test_verify_steps_carry_no_page_grants(afd_scheduler, monkeypatch):
    # A verify step rolls lengths back and stages pages a block ahead; the attention
    # workers' allocator mirrors that, so the coordinator sends no explicit page ids.
    sched = _spec_scheduler(afd_scheduler, monkeypatch, block=4)
    _running_req(sched, uid=1, prompt_len=6, max_tokens=40)
    _, batch = sched.schedule_step(2, phase="decode")
    assert sched.page_grants(batch.reqs) == ((), (), ())


def test_spec_rejects_deployments_it_does_not_implement(monkeypatch):
    from minisgl.spec.config import AfdSpecConfig, afd_spec_config, check_spec_supported

    monkeypatch.setenv("MINISGL_AFD_SPEC", "eagle3")
    monkeypatch.setenv("MINISGL_AFD_SPEC_DRAFT", "unused")
    check_spec_supported(afd_spec_config(), prefix_cache="naive")  # supported
    check_spec_supported(AfdSpecConfig(), prefix_cache="radix")  # spec off: anything goes
    with pytest.raises(ValueError, match="MINISGL_AFD_PREFIX_CACHE=radix"):
        check_spec_supported(afd_spec_config(), prefix_cache="radix")
