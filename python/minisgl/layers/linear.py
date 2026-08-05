from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, div_even

from .base import BaseOP


@lru_cache(maxsize=1)
def _dense_fp8_use_bf16_cublas() -> bool:
    """Dequant fp8 dense weights to bf16 and use cuBLAS (decode-skinny win)."""
    return False


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        quant: Any | None = None,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self._fp8_block_size: tuple[int, int] | None = None
        self._fp8_scale_format: str | None = None
        self._fp8_weight_scale_for_deepgemm: torch.Tensor | None = None
        self._bf16_weight: torch.Tensor | None = None
        self._use_bf16_cublas_for_fp8 = False
        quant_method = getattr(quant, "method", None) if quant is not None else None
        if quant_method in ("fp8", "fp8_channel"):
            if quant_method == "fp8_channel":
                block_out, block_in = (1, 1)
                self._fp8_scale_format = "channel"
            else:
                block_out, block_in = tuple(getattr(quant, "weight_block_size"))
                self._fp8_scale_format = "block"
            self._fp8_block_size = (int(block_out), int(block_in))
            self.weight = torch.empty(
                local_osize,
                local_isize,
                dtype=torch.float8_e4m3fn,
            )
            if self._fp8_scale_format == "channel":
                self.weight_scale = torch.empty(local_osize, 1, dtype=torch.float32)
            else:
                self.weight_scale = torch.empty(
                    div_ceil(local_osize, int(block_out)),
                    div_ceil(local_isize, int(block_in)),
                    dtype=torch.float32,
                )
        else:
            self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype == torch.float8_e4m3fn:
            if _dense_fp8_use_bf16_cublas() or self._use_bf16_cublas_for_fp8:
                return self._forward_bf16_dequant(x)
            return self._forward_fp8(x)
        return F.linear(x, self.weight, self.bias)

    def _forward_bf16_dequant(self, x: torch.Tensor) -> torch.Tensor:
        # Decode-skinny m: cuBLAS bf16 runs at the weight-stream roofline
        # (9-12us at m=46) while the blockscaled fp8 path takes ~19us PLUS a
        # per-token activation-quant kernel.  Dequanting the fp8 checkpoint
        # weight to bf16 is lossless in the mantissa (e4m3 c bf16) and
        # removes the activation quantization entirely.
        w = self._bf16_weight
        if w is None:
            w = self._dequant_weight_to_bf16()
        return F.linear(x, w, self.bias)

    def _dequant_weight_to_bf16(self) -> torch.Tensor:
        if not hasattr(self, "weight_scale"):
            raise RuntimeError("FP8 dense linear is missing weight_scale")
        if self._fp8_scale_format == "channel":
            with torch.no_grad():
                scale = self.weight_scale.to(torch.float32)
                while scale.ndim < self.weight.ndim:
                    scale = scale.unsqueeze(-1)
                w = (self.weight.to(torch.float32) * scale).to(torch.bfloat16)
            self._bf16_weight = w.contiguous()
            return self._bf16_weight
        block_out, block_in = self._fp8_block_size
        with torch.no_grad():
            scale = self.weight_scale.to(torch.float32)
            scale = torch.repeat_interleave(scale, block_out, dim=0)
            scale = torch.repeat_interleave(scale, block_in, dim=1)
            scale = scale[: self.weight.size(0), : self.weight.size(1)]
            w = (self.weight.to(torch.float32) * scale).to(torch.bfloat16)
        self._bf16_weight = w.contiguous()
        return self._bf16_weight

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        if self.weight.dtype == torch.float8_e4m3fn and self.weight.is_cuda:
            if _dense_fp8_use_bf16_cublas() or self._use_bf16_cublas_for_fp8:
                self._dequant_weight_to_bf16()
                # Free the fp8 copy + scales (the dtype of the empty stub
                # keeps routing forward() to the bf16 branch); net cost is
                # +1 byte/elem vs fp8 instead of +2.
                dev = self._bf16_weight.device
                self.weight = torch.empty((0,), dtype=torch.float8_e4m3fn, device=dev)
                if hasattr(self, "weight_scale"):
                    self.weight_scale = torch.empty((0,), dtype=torch.float32, device=dev)
            else:
                self._fp8_weight_for_deepgemm()

    def _fp8_weight_for_deepgemm(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fp8_weight_scale_for_deepgemm is not None:
            return self.weight, self._fp8_weight_scale_for_deepgemm
        if self._fp8_scale_format == "channel":
            if not hasattr(self, "weight_scale"):
                raise RuntimeError("FP8 dense linear is missing weight_scale")

            from minisgl.kernel import deepgemm as deep_gemm

            with torch.no_grad():
                scale = self.weight_scale.to(torch.float32)
                while scale.ndim < self.weight.ndim:
                    scale = scale.unsqueeze(-1)
                weight_dequant = self.weight.to(torch.float32) * scale
                weight_fp8, weight_scale = deep_gemm.per_block_cast_to_fp8(
                    weight_dequant,
                    use_ue8m0=True,
                    gran_k=128,
                )
                self.weight.copy_(weight_fp8)
                self.weight_scale = weight_scale.contiguous()
                self._fp8_block_size = (128, 128)
                self._fp8_scale_format = "block"
                self._fp8_weight_scale_for_deepgemm = (
                    deep_gemm.transform_sf_into_required_layout(
                        self.weight_scale.unsqueeze(0),
                        self.weight.size(0),
                        self.weight.size(1),
                        (1, 128, 128),
                        1,
                        False,
                        False,
                    ).squeeze(0)
                )
                del weight_dequant
            return self.weight, self._fp8_weight_scale_for_deepgemm
        if self._fp8_block_size != (128, 128):
            raise RuntimeError(
                f"DeepGEMM FP8 dense linear requires 128x128 block scales, "
                f"got {self._fp8_block_size}"
            )
        if not hasattr(self, "weight_scale"):
            raise RuntimeError("FP8 dense linear is missing weight_scale")

        from minisgl.kernel import deepgemm as deep_gemm

        block_out, block_in = self._fp8_block_size
        with torch.no_grad():
            scale_float = self.weight_scale.to(torch.float32)
            scale_expanded = torch.repeat_interleave(scale_float, block_out, dim=0)
            scale_expanded = torch.repeat_interleave(scale_expanded, block_in, dim=1)
            scale_expanded = scale_expanded[: self.weight.size(0), : self.weight.size(1)]
            weight_bf16 = self.weight.to(torch.float32) * scale_expanded
            weight_fp8, weight_scale = deep_gemm.per_block_cast_to_fp8(
                weight_bf16,
                use_ue8m0=True,
                gran_k=block_out,
            )
            self.weight.copy_(weight_fp8)
            self.weight_scale.copy_(weight_scale)
            self._fp8_weight_scale_for_deepgemm = deep_gemm.transform_sf_into_required_layout(
                self.weight_scale.unsqueeze(0),
                self.weight.size(0),
                self.weight.size(1),
                (1, 128, 128),
                1,
                False,
                False,
            ).squeeze(0)
        return self.weight, self._fp8_weight_scale_for_deepgemm

    def _forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("FP8 dense linear requires CUDA input")
        from minisgl.kernel import deepgemm as deep_gemm

        input_2d = x.contiguous().view(-1, x.shape[-1])
        input_fp8, input_scale = deep_gemm.per_token_cast_to_fp8(
            input_2d,
            use_ue8m0=True,
            gran_k=128,
            use_packed_ue8m0=True,
        )
        weight, weight_scale = self._fp8_weight_for_deepgemm()
        output = torch.empty(
            (input_fp8.size(0), weight.size(0)),
            dtype=torch.bfloat16,
            device=input_fp8.device,
        )
        deep_gemm.fp8_gemm_nt(
            (input_fp8, input_scale),
            (weight, weight_scale),
            output,
            disable_ue8m0_cast=False,
        )
        if self.bias is not None:
            output = output + self.bias
        return output.to(dtype=x.dtype).view(*x.shape[:-1], weight.size(0))

    def forward_fp8_prequant(
        self,
        input_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        *,
        output_shape_prefix: tuple[int, ...],
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Run an FP8 dense linear from an already quantized activation.

        This is intentionally narrow: decode-only fused norm+quant paths can
        avoid materializing a bf16 normalized activation just to have
        ``_forward_fp8`` immediately quantize it again.
        """
        if self.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError("forward_fp8_prequant requires an FP8 weight")
        if _dense_fp8_use_bf16_cublas() or self._use_bf16_cublas_for_fp8:
            raise RuntimeError("forward_fp8_prequant is incompatible with bf16 cublas mode")
        from minisgl.kernel import deepgemm as deep_gemm

        weight, weight_scale = self._fp8_weight_for_deepgemm()
        output = torch.empty(
            (input_fp8.size(0), weight.size(0)),
            dtype=torch.bfloat16,
            device=input_fp8.device,
        )
        deep_gemm.fp8_gemm_nt(
            (input_fp8, input_scale),
            (weight, weight_scale),
            output,
            disable_ue8m0_cast=False,
        )
        if self.bias is not None:
            output = output + self.bias
        return output.to(dtype=output_dtype).view(*output_shape_prefix, weight.size(0))


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
            quant=None,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        quant: Any | None = None,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(
            input_size,
            output_size,
            input_size,
            tp_output_size,
            has_bias,
            quant=quant,
        )


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        quant: Any | None = None,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            quant=quant,
        )
        self._use_bf16_cublas_for_fp8 = False


class LinearQKVGatedMerged(_LinearTPImpl):
    """QKV projection whose Q rows carry per-head query and output gate."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        quant: Any | None = None,
    ):
        tp_info = get_tp_info()
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        super().__init__(
            hidden_size,
            (2 * num_qo_heads + 2 * num_kv_heads) * head_dim,
            hidden_size,
            (2 * local_num_qo + 2 * local_num_kv) * head_dim,
            has_bias,
            quant=quant,
        )
        self._use_bf16_cublas_for_fp8 = False


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quant: Any | None = None,
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            quant=quant,
        )
        self._use_bf16_cublas_for_fp8 = False

    def forward(self, x: torch.Tensor, skip_allreduce: bool = False) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1 and not skip_allreduce:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quant: Any | None = None,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size,
            output_size,
            local_input_size,
            local_output_size,
            has_bias,
            quant=quant,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
