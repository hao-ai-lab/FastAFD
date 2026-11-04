"""GDN state-slot index dtype under MINISGL_QWEN35_VLLM_KERNELS=1: int32, as vLLM passes its block ids (call side only).

vLLM db9527a46873 hands both GDN decode kernels (causal_conv1d_update's conv_state_indices and the packed recurrent
kernel's ssm_state_indices) int32 block ids: v1/attention/backends/gdn_attn.py:145-147 (the int32 persistent
non_spec_state_indices_tensor) and :301 (block_table_tensor[:, 0]). FastAFD's request-table slots were int64. The index
dtype is part of the Triton signature, so it selects a different specialization of the same vendored kernel source;
the kernels are unchanged (both load the index and widen it to int64 before the address arithmetic,
fused_recurrent_packed.py:63, causal_conv1d_update.py:87-88 / :203-204). Measured per layer
(2 GPUs): int32 indices cut the packed recurrent kernel's per-request slope from 1.254-1.270
to 1.207-1.216 us (vLLM 1.1905); pool stride and base offset had no consistent effect.

Switch off: int64, byte-identical to before. Slot ids are request-table rows (< max_running_req + 1, slot 0 real, the
dummy request's slot pads a bucket), far below 2**31; padding never uses -1 here (the GDN-sentinel policy is refused
with the switch, models/qwen35_decode.py:199), and a negative id still reads as negative in int32 (the recurrent kernel's
`state_idx < 0` skip, the one vendored edit, is unchanged).
"""

from __future__ import annotations

import torch

from .vllm_kernels import enabled

INT32_SLOTS = 2 ** 31


def state_index_dtype() -> torch.dtype:
    """dtype of the decode state-slot index buffers (AG decode graph runners, the eager decode batch)."""
    return torch.int32 if enabled() else torch.int64


def decode_state_indices(table_indices: torch.Tensor) -> torch.Tensor:
    """The eager decode batch's state indices: int32 under the switch, otherwise the tensor itself (no copy)."""
    if not enabled() or table_indices.dtype == torch.int32:
        return table_indices
    return table_indices.to(torch.int32)


def check_slot_range(max_slots: int) -> None:
    """Refuse a GDN state pool whose slot ids would not fit int32 (only under the switch)."""
    if enabled() and int(max_slots) >= INT32_SLOTS:
        raise ValueError(f"MINISGL_QWEN35_VLLM_KERNELS=1 passes int32 GDN slot ids; the pool has {int(max_slots)} slots (>= 2**31)")
