# SPDX-License-Identifier: Apache-2.0
"""The few vLLM helpers the vendored kernel files import, re-created from vLLM db9527a46873.

Each definition cites the vLLM source it reproduces (paths relative to the vLLM repo root):
  * ``triton`` / ``tl``           vllm/triton_utils/__init__.py:11-13 (plain ``import triton``)
  * ``exp`` / ``log``             vllm/third_party/flash_linear_attention/ops/op.py:15-24
                                  (tl.exp / tl.log unless FLA_USE_FAST_OPS=1, then libdevice fast_expf / fast_logf)
  * ``cdiv`` / ``next_power_of_2``  vllm/utils/math_utils.py:10-17
  * ``num_compute_units``         vllm/platforms/cuda.py:722-723 (multi_processor_count)
  * ``current_platform.is_arch_support_pdl``  vllm/platforms/cuda.py:755-761 (device capability major >= 9)
  * ``PAD_SLOT_ID`` / ``NULL_BLOCK_ID``       vllm/v1/attention/backends/utils.py:51-52
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    import triton.language.extra.libdevice as tldevice

    exp = tldevice.fast_expf
    log = tldevice.fast_logf
else:
    exp = tl.exp
    log = tl.log

PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)."""
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def num_compute_units(device_id: int = 0) -> int:
    return torch.cuda.get_device_properties(device_id).multi_processor_count


class _CudaPlatform:
    @staticmethod
    def is_arch_support_pdl() -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9


current_platform = _CudaPlatform()

__all__ = ["NULL_BLOCK_ID", "PAD_SLOT_ID", "cdiv", "current_platform", "exp", "log",
           "next_power_of_2", "num_compute_units", "tl", "triton"]
