"""A decode DP lane can be re-padded to the collective step's dispatch bucket.

Each DP scheduler pads decode to its own graph size, but one collective step
dispatches with a single DeepEP bucket (the max over lanes). The coordinator
re-pads every smaller decode-graph lane to that bucket, so no lane replays a
graph captured with a different num_max_dispatch_tokens_per_rank than its peers.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.message import UserMsg

DECODE_GRAPH_BS = (8, 16, 24, 32, 40, 48, 56, 64)


@pytest.fixture
def afd_scheduler(monkeypatch):
    """minisgl.afd_scheduler, imported with a stub Ray (afd_support imports it at module scope)."""
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        for name in ("minisgl.afd_scheduler", "minisgl.afd_support"):
            patch.delitem(sys.modules, name, raising=False)
        from minisgl import afd_scheduler
    return afd_scheduler


def _decode_ready_scheduler(afd_scheduler, num_reqs: int, num_mb: int = 1):
    """A DP scheduler whose num_reqs requests finished prefill and wait for decode."""
    sched = afd_scheduler.CentralizedAfdDpScheduler(
        dp_rank=0,
        max_running_req=64,
        max_seq_len=64,
        num_pages=4096,
        page_size=1,
        prefill_budget=8192,
        decode_graph_bs=DECODE_GRAPH_BS,
        cuda_graph_max_bs=64,
        afd_num_mb=num_mb,
        mlp_fanout=1,
        eos_token_id=2,
    )
    if num_reqs == 0:
        return sched
    sched.add_reqs(
        [
            UserMsg(
                uid=uid,
                input_ids=torch.tensor([5, 6, 7], dtype=torch.int32),
                sampling_params=SamplingParams(max_tokens=16),
            )
            for uid in range(num_reqs)
        ]
    )
    _, batch = sched.schedule_step(0, phase="prefill")
    # Mirror the coordinator's post-launch state update, then the prefill completion.
    for req in batch.reqs:
        req.complete_one()
    sched.decode_manager.filter_reqs(batch.reqs)
    sched.complete_results(batch, [9] * len(batch.reqs))
    return sched


@pytest.mark.parametrize(("num_mb", "mb_offsets"), [(1, (0, 56)), (2, (0, 28, 56))])
def test_repad_decode_step_pads_to_bucket(afd_scheduler, num_mb, mb_offsets):
    # 46 live requests (some finished at EOS) pad to the 48 graph; a peer lane
    # still at 50 requests set the step's bucket to 56.
    sched = _decode_ready_scheduler(afd_scheduler, 46, num_mb)
    plan, batch = sched.schedule_step(1, phase="decode")
    assert plan.padded_size == 48
    real_reqs = list(batch.reqs)

    plan = sched.repad_decode_step(1, batch, padded_size=56)

    assert plan.padded_size == 56
    assert plan.graph_token_count == 56
    assert tuple(plan.microbatch_offsets) == mb_offsets
    assert batch.reqs == real_reqs
    assert [req for req in batch.padded_reqs if req is not sched._dummy_req] == real_reqs
    # Each micro-batch keeps a real prefix followed by a dummy tail.
    for start, end in zip(mb_offsets, mb_offsets[1:]):
        lane = batch.padded_reqs[start:end]
        num_real = sum(req is not sched._dummy_req for req in lane)
        assert num_real == 46 // num_mb
        assert all(req is sched._dummy_req for req in lane[num_real:])


def test_repad_decode_step_rejects_prefill(afd_scheduler):
    sched = _decode_ready_scheduler(afd_scheduler, 0)
    sched.add_reqs(
        [
            UserMsg(
                uid=0,
                input_ids=torch.tensor([5, 6], dtype=torch.int32),
                sampling_params=SamplingParams(),
            )
        ]
    )
    _, batch = sched.schedule_step(0, phase="prefill")
    with pytest.raises(RuntimeError, match="expects a decode batch"):
        sched.repad_decode_step(0, batch, padded_size=8)
