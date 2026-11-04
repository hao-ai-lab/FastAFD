"""A prompt given as token ids (input_ids): its checks, and the tokenizer passing it through without
encoding (re-encoding a ~0.8M-token text cost seconds per call before the backend
saw it). The endpoint itself needs FastAPI, which the CPU lane does not install:
tests/core/test_api_input_ids_endpoint.py covers it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.message import BaseTokenizerMsg, TokenizeMsg
from minisgl.server.prompt_ids import prompt_ids_tensor
from minisgl.tokenizer.tokenize import TokenizeManager


def test_ids_become_an_int32_tensor():
    ids = prompt_ids_tensor([5, 0, 200063], 4, vocab_size=200064, max_seq_len=8)
    assert ids.dtype == torch.int32 and ids.tolist() == [5, 0, 200063]


@pytest.mark.parametrize(("ids", "max_tokens", "match"), [
    ([], 1, "empty"),
    ([1, 200064], 1, r"\[0, 200064\)"),
    ([-1, 2], 1, "min -1"),
    ([1, 2**63], 1, "outside int64"),
    ([-(2**63) - 1], 1, "outside int64"),
    ([1] * 6, 3, "exceed max_seq_len 8"),
])
def test_bad_ids_are_refused(ids, max_tokens, match):
    with pytest.raises(ValueError, match=match):
        prompt_ids_tensor(ids, max_tokens, vocab_size=200064, max_seq_len=8)


def test_the_tokenizer_passes_ids_through_and_encodes_text():
    class _Tok:
        def encode(self, prompt, return_tensors):
            assert prompt == "a b"
            return torch.tensor([[11, 12]])

    params = SimpleNamespace()
    out = TokenizeManager(_Tok()).tokenize([
        TokenizeMsg(uid=1, text="", sampling_params=params, input_ids=torch.tensor([9, 8, 7], dtype=torch.int32)),
        TokenizeMsg(uid=2, text="a b", sampling_params=params),
    ])
    assert [t.tolist() for t in out] == [[9, 8, 7], [11, 12]]
    assert all(t.dtype == torch.int32 for t in out)


def test_ids_survive_the_zmq_encoding():
    from minisgl.core import SamplingParams

    msg = TokenizeMsg(uid=3, text="", sampling_params=SamplingParams(max_tokens=2),
                      input_ids=torch.arange(5, dtype=torch.int32))
    back = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(back, TokenizeMsg) and back.input_ids.tolist() == [0, 1, 2, 3, 4]
    plain = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(TokenizeMsg(uid=4, text="t", sampling_params=SamplingParams())))
    assert plain.input_ids is None and plain.text == "t"
