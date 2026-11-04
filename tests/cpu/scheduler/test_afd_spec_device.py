"""Device-input verify steps (spec.device_inputs): host length bookkeeping and the device input gather."""

from __future__ import annotations

import torch

from minisgl.spec.device_inputs import SpecLengths, commit_lengths, verify_inputs_from_committed


def test_start_is_an_upper_bound_over_the_steps_in_flight():
    lengths = SpecLengths(block=4)
    # Just prefilled: the host knows the exact start, so it seeds the device copy.
    assert lengths.begin_step(7, fallback_len=100) == (100, True, 0)
    # A second step before the first reply: every in-flight block counted as fully accepted.
    assert lengths.begin_step(7, fallback_len=0) == (104, False, 0)
    assert lengths.finish_step(7, 0, committed=102)       # first step committed 2 tokens
    assert lengths.exact[7] == 102 and lengths.inflight[7] == 1
    assert lengths.begin_step(7, fallback_len=0) == (106, False, 0)   # 102 + one block in flight
    assert lengths.finish_step(7, 0, committed=105)
    assert lengths.finish_step(7, 0, committed=107)
    assert 7 not in lengths.inflight
    assert lengths.begin_step(7, fallback_len=0) == (107, True, 0)    # nothing in flight: exact again


def test_a_late_reply_of_a_freed_row_does_not_reach_the_next_request():
    lengths = SpecLengths(block=4)
    _, _, gen = lengths.begin_step(3, fallback_len=50)
    lengths.free(3)                                         # the request finished; row 3 reused
    start, exact, new_gen = lengths.begin_step(3, fallback_len=20)
    assert (start, exact) == (20, True) and new_gen != gen
    assert not lengths.finish_step(3, gen, committed=53)    # the old request's reply
    assert lengths.exact[3] == 20 and lengths.inflight[3] == 1
    assert lengths.finish_step(3, new_gen, committed=22)


def test_device_inputs_follow_the_committed_lengths():
    rows, max_len, page_size, block = 4, 32, 4, 3
    dummy = rows - 1
    token_table = torch.arange(rows * max_len, dtype=torch.int32).view(rows, max_len) + 1000
    token_table[dummy].fill_(9999 * page_size)              # the dummy row maps to the dummy page
    token_pool = torch.arange(rows * max_len, dtype=torch.int32).view(rows, max_len)
    committed = torch.tensor([5, 9, 0, 0], dtype=torch.int32)
    slots = torch.tensor([1, 0, dummy], dtype=torch.int64)  # two real slots, then padding
    positions, out_loc, input_ids, start, seq_lens, block_table = verify_inputs_from_committed(
        committed, slots, block, token_table, token_pool, page_size, table_cols=3
    )
    assert positions.tolist() == [9, 10, 11, 5, 6, 7, 0, 1, 2]
    assert start.tolist() == [9, 5, 0] and seq_lens.tolist() == [12, 8, 3]
    assert out_loc.tolist() == [int(token_table[r, p]) for r, p in zip([1, 1, 1, 0, 0, 0, dummy, dummy, dummy], positions.tolist())]
    assert input_ids.tolist() == [int(token_pool[r, p]) for r, p in zip([1, 1, 1, 0, 0, 0, dummy, dummy, dummy], positions.tolist())]
    assert block_table[0].tolist() == [int(token_table[1, c * page_size]) // page_size for c in range(3)]
    assert block_table[2].tolist() == [9999] * 3


def test_commit_writes_real_slots_and_keeps_the_dummy_row_at_zero():
    committed = torch.tensor([5, 9, 0, 0], dtype=torch.int32)
    slots = torch.tensor([1, 0, 3, 3], dtype=torch.int64)   # row 3 is the dummy, padding twice
    commit_lengths(committed, slots, torch.tensor([11, 7, 3, 3]), dummy_row=3)
    assert committed.tolist() == [7, 11, 0, 0]
