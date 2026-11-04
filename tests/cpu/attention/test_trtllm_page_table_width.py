"""CPU page arithmetic: capture tables must cover every reachable sequence."""

from __future__ import annotations

import pytest

from minisgl.attention.trtllm import TensorRTLLMBackend


def _backend(page_size: int) -> TensorRTLLMBackend:
    """A backend carrying only what `_num_pages` reads.

    `__init__` needs a global context and a 128 MiB CUDA workspace; the page
    arithmetic depends on nothing but `page_size`.
    """
    backend = TensorRTLLMBackend.__new__(TensorRTLLMBackend)
    backend.page_size = page_size
    return backend


@pytest.mark.parametrize(
    ("max_seq_len", "page_size", "expected"),
    [
        # The two shapes that crashed 397B decode: max_seq_len is not a multiple
        # of the page size, so flooring left the capture buffer one page short.
        (8505, 64, 133),
        (8361, 64, 131),
        # Exact multiples must not gain a spurious page.
        (8448, 64, 132),
        (8192, 64, 128),
        (1, 64, 1),
    ],
)
def test_num_pages_rounds_up(max_seq_len: int, page_size: int, expected: int) -> None:
    assert _backend(page_size)._num_pages(max_seq_len) == expected


@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_capture_width_covers_every_reachable_sequence(page_size: int) -> None:
    """The capture width bounds every sequence the engine may serve.

    This is the invariant `prepare_for_replay` relies on: replay builds its table
    from the batch's longest sequence, which can be anything up to `max_seq_len`.
    """
    backend = _backend(page_size)
    max_seq_len = 8505
    capture_width = backend._num_pages(max_seq_len)

    for seq_len in range(1, max_seq_len + 1):
        assert backend._num_pages(seq_len) <= capture_width


def test_floor_division_would_be_one_page_short() -> None:
    """Pin the defect this guards against, so a regression is unambiguous."""
    backend = _backend(64)
    max_seq_len = 8505

    assert max_seq_len // 64 == 132, "the old capture width"
    assert backend._num_pages(max_seq_len) == 133, "what replay actually needs"
    assert backend._num_pages(max_seq_len) == max_seq_len // 64 + 1
