from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from minisgl.layers import BaseOP


# ---------------------------------------------------------------------------
# AFD inter-node data types
# ---------------------------------------------------------------------------

@dataclass
class ModelStageState:
    hidden_states: torch.Tensor
    residual: torch.Tensor | None


# ---------------------------------------------------------------------------
# Weight extraction utility
# ---------------------------------------------------------------------------

def extract_stage_state_dict(
    stage_model: BaseOP,
    full_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    expected_state = stage_model.state_dict()
    expected_keys = expected_state.keys()
    missing = [key for key in expected_keys if key not in full_state_dict]
    if missing:
        raise RuntimeError(f"Missing keys for AFD model stage: {missing[:8]}")
    incompatible = [
        (
            key,
            tuple(expected_state[key].shape),
            expected_state[key].dtype,
            tuple(full_state_dict[key].shape),
            full_state_dict[key].dtype,
        )
        for key in expected_keys
        if (
            expected_state[key].shape != full_state_dict[key].shape
            or expected_state[key].dtype != full_state_dict[key].dtype
        )
    ]
    if incompatible:
        details = [
            (
                f"{key}: expected shape={expected_shape} dtype={expected_dtype}, "
                f"got shape={actual_shape} dtype={actual_dtype}"
            )
            for key, expected_shape, expected_dtype, actual_shape, actual_dtype in incompatible[:8]
        ]
        raise RuntimeError("Incompatible AFD stage tensors: " + " | ".join(details))
    return {key: full_state_dict[key] for key in expected_keys}


__all__ = [
    "ModelStageState",
    "extract_stage_state_dict",
]
