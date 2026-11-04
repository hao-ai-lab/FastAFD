"""Checks for a prompt given as token ids (the input_ids extension of /v1/chat/completions).

Kept apart from api_server, which imports the web stack (FastAPI, pydantic, uvicorn) that the
CPU test lane does not install.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def prompt_ids_tensor(input_ids: List[int], max_tokens: int, *, vocab_size: int, max_seq_len: int) -> torch.Tensor:
    """The request's input_ids as an int32 tensor, or ValueError: empty, an id outside the
    vocabulary, or a prompt that leaves no room for max_tokens within max_seq_len."""
    if not input_ids:
        raise ValueError("input_ids is empty")
    try:
        ids = np.asarray(input_ids, dtype=np.int64)
    except OverflowError as exc:  # an id beyond int64 is outside every vocabulary too
        raise ValueError(f"input_ids must lie in [0, {vocab_size}); got an id outside int64") from exc
    lo, hi = int(ids.min()), int(ids.max())
    if lo < 0 or hi >= vocab_size:
        raise ValueError(f"input_ids must lie in [0, {vocab_size}); got min {lo}, max {hi}")
    if len(ids) + max_tokens > max_seq_len:
        raise ValueError(f"{len(ids)} prompt tokens + max_tokens {max_tokens} exceed max_seq_len {max_seq_len}")
    return torch.from_numpy(ids.astype(np.int32))
