"""Native ModelOpt NVFP4 routed-expert utilities for Blackwell.

Checkpoint tensors remain packed FP4 with per-16 FP8-E4M3 block scales.  The
only persistent conversion is the byte/scale reorder required by FlashInfer's
TRT-LLM MoE kernel; no BF16 or FP8-weight copy is retained.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
import os
from pathlib import Path
import sys
import warnings

import torch


@contextmanager
def _flashinfer_cuda_target() -> Iterator[Path | None]:
    """Temporarily point FlashInfer JIT at Conda's CUDA target tree.

    MoE runner modules are imported by attention and expert workers alike.
    Keeping this process-global change scoped prevents a runner import from
    redirecting unrelated TVM-FFI extensions such as pynccl to the target-only
    CUDA tree.
    """

    candidates: list[Path] = []
    for value in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if value:
            candidates.append(Path(value))
    prefix = Path(sys.prefix)
    candidates.append(prefix)
    candidates.extend(sorted((prefix / "targets").glob("*")))
    target: Path | None = None
    for root in candidates:
        if not (root / "include" / "cuda_bf16.h").is_file():
            continue
        target = root
        break
    if target is None:
        yield None
        return

    keys = ("CUDA_HOME", "CUDA_PATH", "PATH", "FLASHINFER_NVCC")
    previous = {key: os.environ.get(key) for key in keys}
    compiler_dirs = tuple(
        path
        for path in (prefix / "nvvm" / "bin", prefix / "bin", target / "bin")
        if path.is_dir()
    )
    os.environ["CUDA_HOME"] = str(target)
    os.environ["CUDA_PATH"] = str(target)
    os.environ["PATH"] = os.pathsep.join(
        (*map(str, compiler_dirs), previous.get("PATH") or "")
    )
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ["FLASHINFER_NVCC"] = str(nvcc)
    try:
        yield target
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class NvFp4PreparedMoEWeights:
    gate_up_proj: torch.Tensor
    gate_up_proj_scale: torch.Tensor
    down_proj: torch.Tensor
    down_proj_scale: torch.Tensor
    input_global_scale_inv: torch.Tensor
    output1_scale: torch.Tensor
    output1_gate_scale: torch.Tensor
    output2_scale: torch.Tensor
    hidden_size: int
    intermediate_size: int
    global_num_experts: int
    local_expert_offset: int

    @property
    def local_num_experts(self) -> int:
        return int(self.gate_up_proj.shape[0])

    def persistent_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            value
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), torch.Tensor)
        )


def pack_topk_ids_weights(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Pack expert id and BF16 route weight in FlashInfer/vLLM's int32 format."""

    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"top-k id/weight shape mismatch: {topk_ids.shape} != {topk_weights.shape}"
        )
    return (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(
        torch.int16
    )


def _unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype not in (torch.uint8, torch.int8, torch.float4_e2m1fn_x2):
        raise TypeError(f"packed NVFP4 tensor must be byte/float4, got {packed.dtype}")
    byte = packed.view(torch.uint8)
    code = torch.empty(
        (*byte.shape[:-1], byte.shape[-1] * 2),
        dtype=torch.uint8,
        device=byte.device,
    )
    code[..., 0::2] = byte & 0x0F
    code[..., 1::2] = byte >> 4
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=byte.device,
    )
    magnitude = values[(code & 0x07).long()]
    return torch.where((code & 0x08) != 0, -magnitude, magnitude)


def _expand_global_weight_scale(
    global_scale: torch.Tensor,
    *,
    weight_shape: tuple[int, ...],
) -> torch.Tensor:
    scale = global_scale.float()
    if scale.numel() == 1:
        return scale
    if len(weight_shape) != 3:
        raise ValueError(
            "non-scalar NVFP4 global scale requires batched expert weights, "
            f"got weight={weight_shape} scale={tuple(scale.shape)}"
        )
    experts, output_rows, _ = weight_shape
    if scale.shape == (experts,):
        return scale.view(experts, 1, 1)
    if scale.shape == (experts, 2):
        if output_rows % 2:
            raise ValueError("merged gate/up output rows must be even")
        return scale.repeat_interleave(output_rows // 2, dim=1).unsqueeze(-1)
    raise ValueError(
        f"unsupported NVFP4 global scale shape {tuple(scale.shape)} for {weight_shape}"
    )


def dequantize_nvfp4_weight(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    group_size: int = 16,
) -> torch.Tensor:
    """Independent FP32 dequantization oracle for ModelOpt NVFP4 weights."""

    values = _unpack_fp4_e2m1(packed)
    if block_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"NVFP4 block scales must be FP8-E4M3, got {block_scale.dtype}"
        )
    if block_scale.shape[:-1] != packed.shape[:-1]:
        raise ValueError(
            f"NVFP4 scale prefix mismatch: {block_scale.shape} vs {packed.shape}"
        )
    logical_k = packed.shape[-1] * 2
    if block_scale.shape[-1] * group_size != logical_k:
        raise ValueError(
            f"NVFP4 group-{group_size} scale K mismatch: packed logical K={logical_k}, "
            f"scale={block_scale.shape[-1]}"
        )
    per_value_scale = torch.repeat_interleave(
        block_scale.float(), group_size, dim=-1
    )
    tensor_scale = _expand_global_weight_scale(
        global_scale, weight_shape=tuple(values.shape)
    )
    return values * per_value_scale * tensor_scale


def _reorder_gate_up(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint [gate, up] rows to TRT-LLM's [up, gate] rows."""

    if weight.shape[-2] % 2:
        raise ValueError(f"merged gate/up rows must be even, got {weight.shape}")
    half = weight.shape[-2] // 2
    gate, up = weight.split(half, dim=-2)
    gate_scale, up_scale = scale.split(half, dim=-2)
    return (
        torch.cat((up, gate), dim=-2).contiguous(),
        torch.cat((up_scale, gate_scale), dim=-2).contiguous(),
    )


def _prepare_static_weights_for_trtllm_configured(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder native tensors exactly as FlashInfer TRT-LLM NVFP4 MoE expects."""

    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    num_experts = int(gate_up_proj.shape[0])
    epilogue_tile_m = 128
    cache: dict[torch.Size, torch.Tensor] = {}
    w1_out: list[torch.Tensor] = []
    s1_out: list[torch.Tensor] = []
    w2_out: list[torch.Tensor] = []
    s2_out: list[torch.Tensor] = []

    w1 = gate_up_proj.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )
    s1 = gate_up_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )
    w2 = down_proj.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )
    s2 = down_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 16
    )

    for expert in range(num_experts):
        w1_idx = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            w1[expert].view(torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=True,
        )
        w1_out.append(
            w1[expert].view(torch.uint8)[w1_idx.to(w1.device)].contiguous()
        )
        s1_idx = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            s1[expert].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        s1_out.append(
            nvfp4_block_scale_interleave(
                s1[expert].view(torch.uint8)[s1_idx.to(s1.device)].contiguous()
            )
        )

        w2_idx = get_w2_permute_indices_with_cache(
            cache, w2[expert].view(torch.uint8), epilogue_tile_m
        )
        w2_out.append(
            w2[expert].view(torch.uint8)[w2_idx.to(w2.device)].contiguous()
        )
        s2_idx = get_w2_permute_indices_with_cache(
            cache,
            s2[expert].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        s2_out.append(
            nvfp4_block_scale_interleave(
                s2[expert].view(torch.uint8)[s2_idx.to(s2.device)].contiguous()
            )
        )

    prepared_s1 = (
        torch.stack(s1_out)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
    )
    prepared_s2 = (
        torch.stack(s2_out)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return torch.stack(w1_out), prepared_s1, torch.stack(w2_out), prepared_s2


def _prepare_static_weights_for_trtllm(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with _flashinfer_cuda_target():
        return _prepare_static_weights_for_trtllm_configured(
            gate_up_proj,
            down_proj,
            gate_up_scale,
            down_scale,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )


def _validate_checkpoint_tensors(
    gate_up_proj: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_scale_2: torch.Tensor,
    down_proj: torch.Tensor,
    down_scale: torch.Tensor,
    down_scale_2: torch.Tensor,
) -> tuple[int, int, int]:
    if gate_up_proj.dtype != torch.uint8 or down_proj.dtype != torch.uint8:
        raise TypeError(
            "ModelOpt NVFP4 expert weights must remain packed uint8; got "
            f"gate_up={gate_up_proj.dtype} down={down_proj.dtype}"
        )
    if (
        gate_up_scale.dtype != torch.float8_e4m3fn
        or down_scale.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("ModelOpt NVFP4 per-16 scales must be FP8-E4M3")
    if gate_up_proj.dim() != 3 or down_proj.dim() != 3:
        raise ValueError("NVFP4 expert weights must have [E, N, packed-K] layout")
    experts, twice_intermediate, packed_hidden = gate_up_proj.shape
    hidden = packed_hidden * 2
    intermediate = twice_intermediate // 2
    expected = {
        "gate_up_scale": (experts, twice_intermediate, hidden // 16),
        "down_proj": (experts, hidden, intermediate // 2),
        "down_scale": (experts, hidden, intermediate // 16),
        "gate_up_scale_2": (experts, 2),
        "down_scale_2": (experts,),
    }
    actual = {
        "gate_up_scale": tuple(gate_up_scale.shape),
        "down_proj": tuple(down_proj.shape),
        "down_scale": tuple(down_scale.shape),
        "gate_up_scale_2": tuple(gate_up_scale_2.shape),
        "down_scale_2": tuple(down_scale_2.shape),
    }
    bad = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if bad:
        raise ValueError(f"invalid ModelOpt NVFP4 expert tensor shapes: {bad}")
    if hidden % 512 != 0:
        raise ValueError(f"FlashInfer TRT-LLM NVFP4 MoE requires hidden % 512 == 0, got {hidden}")
    if intermediate % 16 != 0:
        raise ValueError(f"NVFP4 intermediate size must be divisible by 16, got {intermediate}")
    return experts, hidden, intermediate


def prepare_nvfp4_moe_weights(
    gate_up_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    gate_up_proj_weight_scale_2: torch.Tensor,
    gate_up_proj_input_scale: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_scale: torch.Tensor,
    down_proj_weight_scale_2: torch.Tensor,
    down_proj_input_scale: torch.Tensor,
    *,
    global_num_experts: int | None = None,
    local_expert_offset: int = 0,
) -> NvFp4PreparedMoEWeights:
    """Prepare one local expert shard without changing its FP4 values."""

    experts, hidden, intermediate = _validate_checkpoint_tensors(
        gate_up_proj,
        gate_up_proj_scale,
        gate_up_proj_weight_scale_2,
        down_proj,
        down_proj_scale,
        down_proj_weight_scale_2,
    )
    if gate_up_proj.device.type != "cuda":
        raise RuntimeError("native NVFP4 weight preparation requires CUDA")
    if not torch.allclose(
        gate_up_proj_weight_scale_2[:, 0], gate_up_proj_weight_scale_2[:, 1]
    ):
        warnings.warn(
            "gate_proj and up_proj weight_scale_2 differ; matching vLLM by using gate_proj",
            RuntimeWarning,
            stacklevel=2,
        )

    input1 = gate_up_proj_input_scale.float().max()
    input2 = down_proj_input_scale.float().max()
    if not bool((input1 > 0).item()) or not bool((input2 > 0).item()):
        raise ValueError("ModelOpt NVFP4 input global scales must be positive")
    w1_global = gate_up_proj_weight_scale_2[:, 0].float().contiguous()
    w2_global = down_proj_weight_scale_2.float().contiguous()

    reordered_w1, reordered_s1 = _reorder_gate_up(
        gate_up_proj, gate_up_proj_scale
    )
    prepared_w1, prepared_s1, prepared_w2, prepared_s2 = (
        _prepare_static_weights_for_trtllm(
            reordered_w1,
            down_proj,
            reordered_s1,
            down_proj_scale,
            hidden_size=hidden,
            intermediate_size=intermediate,
        )
    )
    output1_gate_scale = (w1_global * input1).contiguous()
    output1_scale = (output1_gate_scale / input2).contiguous()
    output2_scale = (w2_global * input2).contiguous()
    return NvFp4PreparedMoEWeights(
        gate_up_proj=prepared_w1,
        gate_up_proj_scale=prepared_s1,
        down_proj=prepared_w2,
        down_proj_scale=prepared_s2,
        input_global_scale_inv=(1.0 / input1).reshape(1).contiguous(),
        output1_scale=output1_scale,
        output1_gate_scale=output1_gate_scale,
        output2_scale=output2_scale,
        hidden_size=hidden,
        intermediate_size=intermediate,
        global_num_experts=int(global_num_experts or experts),
        local_expert_offset=int(local_expert_offset),
    )


def _run_nvfp4_moe_configured(
    prepared: NvFp4PreparedMoEWeights,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run pre-routed native NVFP4 experts through FlashInfer TRT-LLM MoE."""

    if hidden_states.device.type != "cuda":
        raise RuntimeError("native NVFP4 MoE requires CUDA")
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"NVFP4 MoE input must be BF16/FP16, got {hidden_states.dtype}")
    if hidden_states.shape[-1] != prepared.hidden_size:
        raise ValueError(
            f"hidden size mismatch: {hidden_states.shape[-1]} != {prepared.hidden_size}"
        )

    import flashinfer
    from flashinfer.fp4_quantization import fp4_quantize

    # The TRT-LLM MoE launcher consumes a true row-major [M, K/16] activation
    # scale tensor.  FlashInfer's higher-level nvfp4_quantize wrapper always
    # emits a padded 8x4/128x4 swizzle, including for its `layout_linear` enum.
    # Use the lower-level unswizzled path, matching vLLM's
    # moe_kernel_quantize_input(..., is_fp4_scale_swizzled=False).
    hidden_q, hidden_scale = fp4_quantize(
        hidden_states.contiguous(),
        prepared.input_global_scale_inv,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
        is_sf_8x4_layout=False,
    )
    kernel_output = torch.empty_like(hidden_states)
    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
        topk_ids=pack_topk_ids_weights(topk_ids, topk_weights).contiguous(),
        routing_bias=None,
        hidden_states=hidden_q,
        hidden_states_scale=hidden_scale.view(torch.float8_e4m3fn),
        gemm1_weights=prepared.gate_up_proj,
        gemm1_weights_scale=prepared.gate_up_proj_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=prepared.down_proj,
        gemm2_weights_scale=prepared.down_proj_scale,
        gemm2_bias=None,
        output1_scale_scalar=prepared.output1_scale,
        output1_scale_gate_scalar=prepared.output1_gate_scale,
        output2_scale_scalar=prepared.output2_scale,
        num_experts=prepared.global_num_experts,
        top_k=int(topk_ids.shape[1]),
        n_group=0,
        topk_group=0,
        intermediate_size=prepared.intermediate_size,
        local_expert_offset=prepared.local_expert_offset,
        local_num_experts=prepared.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,
        do_finalize=True,
        activation_type=3,
        output=kernel_output,
    )
    if output is None:
        return kernel_output
    if output.shape != hidden_states.shape:
        raise ValueError(
            f"output shape mismatch: {output.shape} != {hidden_states.shape}"
        )
    output.copy_(kernel_output)
    return output


def run_nvfp4_moe(
    prepared: NvFp4PreparedMoEWeights,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run native NVFP4 MoE while containing FlashInfer's CUDA JIT config."""

    with _flashinfer_cuda_target():
        return _run_nvfp4_moe_configured(
            prepared,
            hidden_states,
            topk_ids,
            topk_weights,
            output=output,
        )


__all__ = [
    "NvFp4PreparedMoEWeights",
    "dequantize_nvfp4_weight",
    "pack_topk_ids_weights",
    "prepare_nvfp4_moe_weights",
    "run_nvfp4_moe",
]
