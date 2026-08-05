from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class GDNLayerState:
    conv: torch.Tensor
    recurrent: torch.Tensor


class HybridGDNStatePool:
    """Fixed request-slot state for all Qwen3.5 linear-attention layers."""

    def __init__(
        self,
        *,
        layer_ids: tuple[int, ...],
        max_slots: int,
        conv_channels: int,
        conv_width: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_dtype: torch.dtype,
        state_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.layer_ids = layer_ids
        self._layer_to_index = {layer_id: idx for idx, layer_id in enumerate(layer_ids)}
        self.max_slots = int(max_slots)
        self.conv_state = torch.zeros(
            len(layer_ids),
            max_slots,
            conv_channels,
            conv_width - 1,
            dtype=conv_dtype,
            device=device,
        )
        self.recurrent_state = torch.zeros(
            len(layer_ids),
            max_slots,
            num_value_heads,
            value_head_dim,
            key_head_dim,
            dtype=state_dtype,
            device=device,
        )

    def layer_state(self, layer_id: int) -> GDNLayerState:
        try:
            index = self._layer_to_index[int(layer_id)]
        except KeyError as exc:
            raise KeyError(f"layer {layer_id} is not a GDN layer") from exc
        return GDNLayerState(self.conv_state[index], self.recurrent_state[index])

    def clear_slots(self, slots: torch.Tensor) -> None:
        self.conv_state.index_fill_(1, slots, 0)
        self.recurrent_state.index_fill_(1, slots, 0)

    @property
    def bytes_per_slot(self) -> int:
        return (
            self.conv_state[:, 0].numel() * self.conv_state.element_size()
            + self.recurrent_state[:, 0].numel()
            * self.recurrent_state.element_size()
        )


def create_hybrid_gdn_state_pool(
    model_config: ModelConfig,
    *,
    max_slots: int,
    dtype: torch.dtype,
    device: torch.device,
) -> HybridGDNStatePool | None:
    layer_ids = tuple(
        idx
        for idx, layer_type in enumerate(model_config.layer_types)
        if layer_type == "linear_attention"
    )
    if not layer_ids:
        return None
    tp_size = get_tp_info().size
    local_key_heads = div_even(model_config.linear_num_key_heads, tp_size)
    local_value_heads = div_even(model_config.linear_num_value_heads, tp_size)
    conv_channels = (
        2 * local_key_heads * model_config.linear_key_head_dim
        + local_value_heads * model_config.linear_value_head_dim
    )
    state_dtype = (
        torch.float32
        if model_config.mamba_ssm_dtype == "float32"
        else dtype
    )
    return HybridGDNStatePool(
        layer_ids=layer_ids,
        max_slots=max_slots,
        conv_channels=conv_channels,
        conv_width=model_config.linear_conv_kernel_dim,
        num_value_heads=local_value_heads,
        key_head_dim=model_config.linear_key_head_dim,
        value_head_dim=model_config.linear_value_head_dim,
        conv_dtype=dtype,
        state_dtype=state_dtype,
        device=device,
    )


__all__ = ["GDNLayerState", "HybridGDNStatePool", "create_hybrid_gdn_state_pool"]
