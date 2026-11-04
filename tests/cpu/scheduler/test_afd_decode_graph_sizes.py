"""AFD decode graph sizes follow vLLM's capture-size rule and always cover a lane at capacity."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.message import UserMsg

# ServerArgs of an MB2 run at 250 requests per lane: 1000 active + 4 spare
# requests over 4 attention-DP lanes, explicit sizes for 250 per lane.
LANES250 = dict(afd_decode_graph_bs=(1, 125), cuda_graph_max_bs=125, afd_batch_size=1004,
          afd_max_running_req=1004, afd_num_mb=2)
# ServerArgs of an MB2 run at 164 requests per lane.
LANES164 = dict(afd_decode_graph_bs=(1, 82), cuda_graph_max_bs=82, afd_batch_size=660,
           afd_max_running_req=660, afd_num_mb=2)
# LANES250 without an explicit list: server/args.py then sets
# cuda_graph_max_bs = min(afd_max_running_req, 512).
LANES250_DEFAULT = {**LANES250, "afd_decode_graph_bs": (), "cuda_graph_max_bs": 512}


def _vllm_default_capture_sizes(max_capture_size: int, max_num_tokens: int = 8192) -> list[int]:
    """vLLM's default list, transcribed from VllmConfig._set_cudagraph_sizes
    (vllm/config/vllm.py:1688-1708 in the reference build)."""
    sizes = [i for i in [1, 2, 4] if i <= max_capture_size]
    if max_capture_size >= 8:
        sizes += list(range(8, min(max_capture_size + 1, 256), 8))
    if max_capture_size >= 256:
        sizes += list(range(256, max_capture_size + 1, 16))
    if max_num_tokens <= max_capture_size and max_num_tokens not in sizes:
        sizes.append(max_num_tokens)
    return sorted(set(sizes))


@pytest.fixture
def afd(monkeypatch):
    """The coordinator module, imported with stub Ray and worker-launcher modules; warnings recorded."""
    before = set(sys.modules)
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda fn: fn)
    launcher = ModuleType("minisgl.afd_worker_launcher")
    launcher.startup_afd_workers = None
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ray", ray)
        patch.setitem(sys.modules, "minisgl.afd_worker_launcher", launcher)
        import minisgl.afd_coordinator as coordinator_module
        import minisgl.afd_support as support_module
    # CPU torch has no NVTX; the ranges only annotate traces.
    monkeypatch.setattr(coordinator_module, "nvtx_range", contextmanager(lambda name: (yield)))
    warnings: list[str] = []
    monkeypatch.setattr(coordinator_module.logger, "warning",
                        lambda message, *args: warnings.append(message % args))
    yield SimpleNamespace(module=coordinator_module, support=support_module, warnings=warnings)
    for name in set(sys.modules) - before:
        if name.startswith("minisgl.afd_"):
            del sys.modules[name]


def _coordinator(afd, *, attn_dp: int = 4, max_batched_tokens: int = 2048, graphs: bool = True,
                 **server_args):
    """A coordinator sized by its real _refresh_runtime_sizing, with worker sends recorded."""
    coord = afd.module.AfdCoordinator.__new__(afd.module.AfdCoordinator)
    coord.server_args = SimpleNamespace(cache_type="naive", **server_args)
    coord.attention_backend = "trtllm"
    coord.page_size = 16
    coord.max_batched_tokens = max_batched_tokens
    coord.enable_attention_decode_graph = graphs
    coord.enable_model_decode_graph = graphs
    coord._layout = SimpleNamespace(
        attn_dp_size=attn_dp, attn_fanin_per_mlp_dp=1, mlp_dp_size=attn_dp, mlp_tp_size=1,
        mlp_worker_rank=lambda dp, tp: attn_dp + dp,
    )
    coord._refresh_runtime_sizing()
    coord._control_profile_start_step = 0
    coord._control_profile_stop_step = 0
    coord._log_afd_steps = False
    coord.sent = []
    coord._send_cmd_to_workers = lambda ranks, cmd: coord.sent.append(cmd)
    return coord


def _lane(afd, coord):
    """One lane scheduler built as _init_centralized_schedulers builds it."""
    return afd.module.CentralizedAfdDpScheduler(
        dp_rank=0, max_running_req=coord.attn_max_running_req, max_seq_len=64,
        num_pages=4096, page_size=coord.page_size, prefill_budget=coord.max_batched_tokens,
        decode_graph_bs=tuple(coord.decode_graph_bs), cuda_graph_max_bs=coord.max_graph_bs,
        afd_num_mb=coord.afd_num_mb, mlp_fanout=1, eos_token_id=0,
    )


@pytest.mark.parametrize("pages", [None, 256, 4096])
def test_explicit_num_pages_reaches_afd_runtime(afd, pages):
    coord = _coordinator(afd, **LANES250, num_page_override=pages)
    assert coord.num_page_override == pages


def _prefilled_lanes(afd, coord, prompt_lens_per_lane: list[list[int]]) -> list:
    """Lanes prefilled under the coordinator's phase rule until the first decode step."""
    lanes = [_lane(afd, coord) for _ in prompt_lens_per_lane]
    uid = 0
    for lane, prompt_lens in zip(lanes, prompt_lens_per_lane):
        for prompt_len in prompt_lens:
            lane.add_reqs([UserMsg(
                uid=uid,
                input_ids=torch.zeros(prompt_len, dtype=torch.int32),
                sampling_params=SamplingParams(ignore_eos=True, max_tokens=4),
            )])
            uid += 1
    step_id = 0
    while afd.module.AfdCoordinator._afd_next_global_phase(lanes) == "prefill":
        step_id += 1
        for lane in lanes:
            result = lane.schedule_step(step_id, phase="prefill")
            if result is None:
                continue
            batch = result[1]
            for req in batch.reqs:  # what _afd_launch_step does at launch
                req.complete_one()
            lane.decode_manager.filter_reqs(batch.reqs)
            lane.complete_results(batch, [0] * len(batch.reqs))
    return lanes


def _decode_step(afd, coord, lanes) -> tuple[int, list[int], list[int], list[bool]]:
    """(dispatch bucket, real and padded requests per lane, whether each AG worker replays a graph)."""
    bucket, dp_batches, phase, num_mb = coord._afd_prepare_step_launch(1000, lanes, "decode")
    coord._afd_launch_step(
        step_id=1000, attn_tp=1, scheds=lanes,
        forward_remains=[afd.module._AfdForwardRemain() for _ in lanes],
        pending_free={dp: () for dp in range(len(lanes))},
        bucket=bucket, dp_batches=dp_batches, phase=phase, num_mb=num_mb,
    )
    ag_plans = [cmd.plan for cmd in coord.sent if isinstance(cmd, afd.module.AfdRunAGStepCmd)]
    padded = [len(item.batch.padded_reqs) for item in dp_batches]
    # An AG worker replays only a captured graph; it keys them by per-MB size x num_mb.
    captured = {bs * coord.afd_num_mb for bs in coord.decode_graph_bs}
    use_graph = [plan.use_decode_graph and size in captured for plan, size in zip(ag_plans, padded)]
    return bucket, [len(item.batch.reqs) for item in dp_batches], padded, use_graph


# ---- compute_decode_graph_sizes ---------------------------------------------


@pytest.mark.parametrize("max_per_lane", [8, 125, 126, 251, 256])
def test_default_sizes_are_vllms_list_plus_the_lane_capacity(afd, max_per_lane):
    sizes = afd.support.compute_decode_graph_sizes(max_per_lane)
    assert sizes == tuple(sorted({*_vllm_default_capture_sizes(max_per_lane), max_per_lane}))


def test_default_sizes_worked_values(afd):
    compute = afd.support.compute_decode_graph_sizes
    assert compute(125) == (1, 2, 4, *range(8, 121, 8), 125)
    assert compute(256) == (1, 2, 4, *range(8, 257, 8))
    assert compute(300) == (1, 2, 4, *range(8, 257, 8), 272, 288, 300)


def test_default_sizes_stop_at_the_capture_limit_then_add_the_lane_capacity(afd):
    compute = afd.support.compute_decode_graph_sizes
    assert compute(251, max_capture_size=64) == (*_vllm_default_capture_sizes(64), 251)
    assert compute(251, max_capture_size=512) == compute(251)


def test_explicit_sizes_keep_their_granularity(afd):
    compute = afd.support.compute_decode_graph_sizes
    assert compute(126, (125, 1, 125)) == (1, 125, 126)
    assert compute(126, (1, 128)) == (1, 128)
    assert compute(126, ()) == (126,)


# ---- coordinator sizing ----------------------------------------------------


def test_g4_explicit_sizes_gain_the_lane_capacity_with_a_warning(afd):
    coord = _coordinator(afd, **LANES250)
    assert coord.attn_max_running_req == 251
    assert coord.decode_graph_bs == (1, 125, 126)
    assert coord.max_graph_bs == 126
    assert coord.max_comm_tokens == 2048  # still the prefill budget
    assert len(afd.warnings) == 1 and "[1, 125]" in afd.warnings[0] and "126" in afd.warnings[0]


def test_g2b_explicit_sizes_gain_the_lane_capacity(afd):
    coord = _coordinator(afd, **LANES164)
    assert coord.decode_graph_bs == (1, 82, 83)


def test_explicit_sizes_that_cover_a_lane_are_kept_without_a_warning(afd):
    coord = _coordinator(afd, **{**LANES250, "afd_decode_graph_bs": (1, 64, 128)})
    assert coord.decode_graph_bs == (1, 64, 128)
    assert afd.warnings == []


def test_multi_microbatch_explicit_sizes_above_a_split_batch_are_dropped(afd):
    coord = _coordinator(afd, **{**LANES250, "afd_decode_graph_bs": (1, 600)})  # 600 > ceil(1004 / 2)
    assert coord.decode_graph_bs == (1, 126)


def test_single_microbatch_explicit_sizes_gain_the_lane_capacity(afd):
    """One microbatch at 275 requests per lane: 1100 active + 4 spare requests over 4 lanes."""
    coord = _coordinator(afd, afd_decode_graph_bs=(1, 275), cuda_graph_max_bs=275,
                         afd_batch_size=1104, afd_max_running_req=1104, afd_num_mb=1)
    assert coord.decode_graph_bs == (1, 275, 276)
    assert len(afd.warnings) == 1


@pytest.mark.parametrize(("num_mb", "lane_graph_bs"), [(1, 251), (2, 126)])
def test_default_sizes_come_from_the_lane_not_the_server_batch(afd, num_mb, lane_graph_bs):
    coord = _coordinator(afd, **{**LANES250_DEFAULT, "afd_num_mb": num_mb})
    assert coord.decode_graph_bs == afd.support.compute_decode_graph_sizes(lane_graph_bs)
    assert afd.warnings == []


@pytest.mark.parametrize(
    "overrides",
    [{"afd_max_running_req": 0}, {"afd_max_running_req": 1001}],
    ids=["auto_from_batch_size", "not_divisible_by_lanes"],
)
def test_lane_capacity_is_the_coordinators_rounded_up_share(afd, overrides):
    coord = _coordinator(afd, **{**LANES250, **overrides})
    assert coord.attn_max_running_req == 251
    assert coord.decode_graph_bs == (1, 125, 126)


def test_startup_refuses_sizes_that_leave_a_lane_without_a_graph(afd, monkeypatch):
    short = afd.support.AfdRuntimeSizing(
        requested_graph_bs=(1, 125), decode_graph_bs=(1, 125), max_graph_bs=125,
        max_running_req=1004, num_page_override=None, max_comm_tokens=2048, num_mb=2,
    )
    monkeypatch.setattr(afd.module, "build_runtime_sizing", lambda **kwargs: short)
    with pytest.raises(RuntimeError, match=r"stop at 125 .* up to 251 requests .* 126 per microbatch"):
        _coordinator(afd, **LANES250)
    # With decode graphs off every step is eager anyway.
    _coordinator(afd, graphs=False, **LANES250)


# ---- padding and the step's graph decision ----------------------------------


@pytest.mark.parametrize(
    ("num_mb", "real", "padded"),
    [(2, 241, 252), (2, 200, 208), (2, 3, 4), (1, 249, 251), (1, 241, 248), (1, 1, 1)],
)
def test_decode_batch_pads_to_the_next_default_size(afd, num_mb, real, padded):
    coord = _coordinator(afd, **{**LANES250_DEFAULT, "afd_num_mb": num_mb})
    assert _lane(afd, coord)._select_padded_size(phase="decode", real_size=real) == padded


def test_g4_first_wave_with_a_251_request_lane_replays_on_every_lane(afd):
    """The per-lane first wave measured in the LANES250 AG traces: 250, 251, 250, 250."""
    coord = _coordinator(afd, **LANES250)
    lanes = _prefilled_lanes(afd, coord, [[4] * 250, [4] * 251, [4] * 250, [4] * 250])
    assert _decode_step(afd, coord, lanes) == (
        252, [250, 251, 250, 250], [252, 252, 252, 252], [True] * 4
    )


@pytest.mark.parametrize(("num_mb", "real"), [(1, [9, 8, 8, 8]), (2, [3, 2, 2, 2])])
def test_uneven_dp_lanes_share_bucket_without_changing_real_requests(afd, num_mb, real):
    """Main's common-bucket padding composes with full-lane MB graph sizing."""
    coord = _coordinator(afd, afd_decode_graph_bs=(1, 8, 16), cuda_graph_max_bs=16,
                         afd_batch_size=64, afd_max_running_req=64, afd_num_mb=num_mb)
    lanes = _prefilled_lanes(afd, coord, [[4] * n for n in real])
    assert _decode_step(afd, coord, lanes) == (16, real, [16] * 4, [True] * 4)
    plans = [cmd.plan for cmd in coord.sent if isinstance(cmd, afd.module.AfdRunAGStepCmd)]
    for plan, count in zip(plans, real):
        assert plan.real_size == count
        assert sum(plan.microbatch_real_token_counts) == count
        assert plan.dispatch_bucket == 16


def test_g4_first_wave_of_250_per_lane_keeps_its_exact_size(afd):
    coord = _coordinator(afd, **LANES250)
    lanes = _prefilled_lanes(afd, coord, [[4] * 250] * 4)
    assert _decode_step(afd, coord, lanes) == (250, [250] * 4, [250] * 4, [True] * 4)


def test_g2b_first_wave_of_164_per_lane_keeps_its_exact_size(afd):
    coord = _coordinator(afd, **LANES164)
    lanes = _prefilled_lanes(afd, coord, [[4] * 164] * 4)
    assert _decode_step(afd, coord, lanes) == (164, [164] * 4, [164] * 4, [True] * 4)


def test_spare_that_finishes_prefill_first_joins_its_lane_and_the_step_replays(afd):
    """The driver's shape at toy scale: 4 active + 1 spare per lane, prompts longer than the prefill budget.

    Both lanes must finish already-admitted chunks before decode, even with
    full request tables. Existing slots are reused rather than allocated again.
    """
    coord = _coordinator(afd, attn_dp=2, max_batched_tokens=8, afd_decode_graph_bs=(1, 2),
                         cuda_graph_max_bs=2, afd_batch_size=10, afd_max_running_req=10, afd_num_mb=2)
    lanes = _prefilled_lanes(afd, coord, [[12] * 5, [10] * 5])
    assert _decode_step(afd, coord, lanes) == (6, [5, 5], [6, 6], [True, True])


@pytest.mark.parametrize("num_mb", [1, 2])
@pytest.mark.parametrize("per_rank", [8, 302, 438])
def test_full_table_resumes_admitted_chunks_before_decode(afd, num_mb, per_rank):
    """Real ledger at target request counts; toy tokens, no GPU performance claim."""
    coord = _coordinator(
        afd, attn_dp=4, max_batched_tokens=8,
        afd_decode_graph_bs=(1, per_rank // num_mb),
        cuda_graph_max_bs=per_rank // num_mb,
        afd_batch_size=4 * (per_rank + 1), afd_max_running_req=4 * per_rank,
        afd_num_mb=num_mb,
    )
    lanes = _prefilled_lanes(afd, coord, [[n] * (per_rank + 1) for n in (12, 10, 12, 10)])
    assert _decode_step(afd, coord, lanes) == (
        per_rank, [per_rank] * 4, [per_rank] * 4, [True] * 4
    )
    for lane in lanes:
        assert lane.table_manager.available_size == 0
        # The spare is unadmitted: completing an existing chunk must not
        # oversubscribe the table to admit this new request.
        assert len(lane.prefill_manager.pending_list) == 1
        assert lane.prefill_manager.pending_list[0].chunked_req is None
        reqs = lane.decode_manager.running_reqs.values()
        assert len({r.table_idx for r in reqs}) == per_rank
