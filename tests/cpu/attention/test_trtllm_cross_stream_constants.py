"""TRT-LLM decode metadata constants shared across runtime streams.

An attention worker with several runtimes (one per microbatch lane) builds every runtime's
decode metadata with the same attention backends, each runtime on its own stream. The page offsets of the
block-table gather (and the decode cu_seqlens_q) are cached by the first runtime that
needs them; the others' streams read them afterwards and must wait for the stream that
built them. Unordered, a runtime released together with the builder (all of them wait
for the previous step's compute) gathered with offsets not yet written: index out of
bounds at the first decode step with decode graphs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import minisgl.attention.trtllm as trt


def _backend(page_size: int = 64) -> trt.TensorRTLLMBackend:
    """A backend carrying only what the cached constants read (`__init__` needs a global
    context and a CUDA workspace)."""
    backend = trt.TensorRTLLMBackend.__new__(trt.TensorRTLLMBackend)
    backend.page_size = page_size
    backend.kvcache = SimpleNamespace(device=torch.device("cpu"))
    backend._page_offsets_by_len = {}
    backend._decode_cu_seqlens_q = torch.empty((0,), dtype=torch.int32)
    backend._decode_cu_seqlens_q_ready = None
    return backend


@pytest.fixture
def streams(monkeypatch):
    """The current stream by name; built constants record (builder, tensor) events, and
    every stream wait is logged as (waiting stream, event)."""
    s = SimpleNamespace(current="rt0", capturing=False, waits=[])
    monkeypatch.setattr(trt, "_built_on_this_stream", lambda t: ("built", s.current, t.data_ptr()))
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: SimpleNamespace(wait_event=lambda ev: s.waits.append((s.current, ev))))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: s.capturing)
    return s


def test_page_offsets_one_stream_built_are_waited_for_by_the_others(streams):
    backend = _backend()
    offsets = backend._get_page_offsets(8000)  # 125 pages, built on rt0
    assert torch.equal(offsets, torch.arange(125) * 64)
    built = ("built", "rt0", offsets.data_ptr())
    streams.current = "rt1"
    assert backend._get_page_offsets(7990) is offsets  # the same page count, cached
    assert streams.waits == [("rt0", built), ("rt1", built)]
    # a new page count is built and recorded on the stream that needs it first
    streams.current = "rt2"
    wider = backend._get_page_offsets(8100)
    assert streams.waits[-1] == ("rt2", ("built", "rt2", wider.data_ptr()))
    streams.current = "rt0"
    assert backend._get_page_offsets(8100) is wider
    assert streams.waits[-1] == ("rt0", ("built", "rt2", wider.data_ptr()))


def test_decode_cu_seqlens_q_one_stream_grew_is_waited_for_by_the_others(streams):
    backend = _backend()
    first = backend._get_decode_cu_seqlens_q(8)  # grown on rt0
    assert torch.equal(first, torch.arange(9, dtype=torch.int32))
    built = ("built", "rt0", backend._decode_cu_seqlens_q.data_ptr())
    streams.current = "rt3"
    assert torch.equal(backend._get_decode_cu_seqlens_q(4), torch.arange(5, dtype=torch.int32))
    assert streams.waits == [("rt0", built), ("rt3", built)]
    backend._get_decode_cu_seqlens_q(16)  # grown again, on rt3
    assert streams.waits[-1] == ("rt3", ("built", "rt3", backend._decode_cu_seqlens_q.data_ptr()))


def test_no_wait_is_captured(streams):
    backend = _backend()
    backend._get_page_offsets(640)
    streams.capturing = True
    backend._get_page_offsets(640)
    assert len(streams.waits) == 1


def test_host_constants_need_no_event():
    assert trt._built_on_this_stream(torch.arange(4)) is None
    trt._wait_until_built(None)
