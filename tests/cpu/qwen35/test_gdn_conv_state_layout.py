"""The Qwen3.5 GDN conv state is stored tap-major (vLLM "SD") behind the logical [layer, slot, channel, tap] view.

vLLM keeps the conv state as (state_len, dim) with dim innermost (mamba_utils.get_conv_state_layout
defaults to "SD"), so each tap of the decode conv is one contiguous channel run. The pool must keep
that storage while every consumer (decode kernel strides, eager decode, prefill gather/scatter,
slot clearing) keeps seeing the same logical values it saw with the channel-major layout. The
decode conv then launches like vLLM's causal_conv1d_update (256-channel blocks on 4 warps).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import minisgl.distributed.info as dist_info
from minisgl.distributed.info import DistributedInfo
from minisgl.kvcache.hybrid_pool import create_hybrid_gdn_state_pool

CHANNELS = 2 * 2 * 8 + 4 * 6  # 2 key heads x 8, twice, plus 4 value heads x 6


def _pool(monkeypatch, dtype):
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank=0, size=1))
    config = SimpleNamespace(
        layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention"),
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
    )
    pool = create_hybrid_gdn_state_pool(config, max_slots=5, dtype=dtype, device=torch.device("cpu"))
    gen = torch.Generator().manual_seed(0)
    pool.conv_state.copy_(torch.randn(pool.conv_state.shape, generator=gen).to(dtype))
    return pool


def test_conv_state_is_tap_major_behind_the_logical_view(monkeypatch):
    pool = _pool(monkeypatch, torch.bfloat16)
    assert pool.conv_state.shape == (3, 5, CHANNELS, 3)
    # one tap's channels are contiguous; taps are CHANNELS apart (vLLM's SD layout)
    assert pool.conv_state.stride()[-2:] == (1, CHANNELS)
    layer = pool.layer_state(1).conv
    assert layer.shape == (5, CHANNELS, 3)
    assert layer.stride() == (3 * CHANNELS, 1, CHANNELS)
    assert layer.data_ptr() == pool.conv_state[1].data_ptr()
    assert pool.bytes_per_slot == 3 * CHANNELS * 3 * 2 + 3 * 4 * 6 * 8 * 4


def test_eager_decode_state_update_matches_the_channel_major_layout(gdn, monkeypatch):
    pool = _pool(monkeypatch, torch.bfloat16)
    layer = pool.layer_state(0).conv
    dense = layer.contiguous().clone()  # the former [slot, channel, tap] storage, same values
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(3, CHANNELS, generator=gen).to(torch.bfloat16)
    weight = torch.randn(CHANNELS, 1, 4, generator=gen).to(torch.bfloat16)
    slots = torch.tensor([4, 0, 2])
    positions = torch.tensor([5, 0, 9], dtype=torch.int32)  # includes a position-0 reset

    out_view = gdn.qwen35_causal_conv1d_decode_eager(x, weight, layer, slots, positions)
    out_dense = gdn.qwen35_causal_conv1d_decode_eager(x, weight, dense, slots, positions)

    assert torch.equal(out_view, out_dense)
    assert torch.equal(layer, dense)  # touched and untouched slots alike


def test_prefill_gather_and_scatter_match_the_channel_major_layout(gdn, monkeypatch):
    """Qwen3_5LinearAttention.forward's prefill: index_select slots, convolve, index_copy_ back."""
    pool = _pool(monkeypatch, torch.float32)
    layer = pool.layer_state(2).conv
    dense = layer.contiguous().clone()
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(7, CHANNELS, generator=gen)
    weight = torch.randn(CHANNELS, 1, 4, generator=gen)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
    slots = torch.tensor([3, 1])
    first_positions = torch.tensor([0, 6])

    results = []
    for state in (layer, dense):
        initial = state.index_select(0, slots)
        mixed, final = gdn.qwen35_causal_conv1d_prefill(x, weight, initial, cu_seqlens, first_positions)
        state.index_copy_(0, slots, final)
        results.append(mixed)

    assert torch.equal(results[0], results[1])
    assert torch.equal(layer, dense)


def test_clear_slots_zeroes_only_those_slots_through_the_view(monkeypatch):
    pool = _pool(monkeypatch, torch.float32)
    expected = pool.conv_state.contiguous().clone()
    expected[:, 1] = 0

    pool.clear_slots(torch.tensor([1], dtype=torch.int64))

    assert torch.equal(pool.conv_state, expected)


class _CudaLooking(torch.Tensor):
    """A CPU tensor that reports is_cuda, so the decode wrapper takes its kernel branch."""

    @property
    def is_cuda(self):
        return True


class _LaunchRecorder:
    """Stands in for the Triton kernel: `kernel[grid](*args, **kwargs)` records the launch."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((grid, args, kwargs))


def test_decode_launch_matches_vllm_causal_conv1d_update(gdn, monkeypatch):
    """BLOCK 256 channels on 4 warps (vLLM's BLOCK_N), and the pool's tap-major strides reach the kernel."""
    pool = _pool(monkeypatch, torch.bfloat16)
    recorder = _LaunchRecorder()
    monkeypatch.setattr(gdn, "_qwen35_causal_conv1d_decode_kernel", recorder)
    monkeypatch.delenv("MINISGL_QWEN35_CONV1D_EAGER", raising=False)
    channels = 2 * 16 * 128 + 64 * 128  # Qwen3.5-397B at TP1
    x = torch.zeros(3, channels, dtype=torch.bfloat16).as_subclass(_CudaLooking)
    weight = torch.zeros(channels, 1, 4, dtype=torch.bfloat16)
    state = torch.zeros(4, 3, channels, dtype=torch.bfloat16).transpose(-1, -2)

    gdn.qwen35_causal_conv1d_decode(
        x, weight, state, torch.tensor([0, 2, 3]), torch.ones(3, dtype=torch.int32)
    )

    ((grid, args, kwargs),) = recorder.launches
    assert kwargs["BLOCK_C"] == 256
    assert kwargs["num_warps"] == 4
    assert grid == (3, channels // 256)
    assert args[8:11] == (3 * channels, 1, channels)  # slot, channel, tap strides
    # the pool hands the kernel the same channel-contiguous strides
    assert pool.layer_state(0).conv.stride()[1:] == (1, CHANNELS)
