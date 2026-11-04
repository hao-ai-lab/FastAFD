from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class DetokenizeBatchMsg(BaseTokenizerMsg):
    """Compact form of many DetokenizeMsg objects.

    Large AFD decode steps return thousands of one-token replies per step. Sending
    three flat lists avoids constructing and serializing one dataclass per request
    on the coordinator hot path; tokenizer/server.py expands it at the boundary
    where normal detokenization logic still consumes DetokenizeMsg.
    """

    uids: List[int]
    next_tokens: List[int]
    finished: List[bool]


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams
    # The prompt as int32 token ids (the API's input_ids); when set the tokenizer passes them
    # through and `text` is unused.
    input_ids: torch.Tensor | None = None


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
