"""vLLM db9527a4's router top-k (``topkGating``, csrc/jit/vllm_topk_softmax.cu) for MINISGL_QWEN35_VLLM_ROUTER.

Host side of vLLM ``fused_topk`` (fused_moe/router/fused_topk_router.py:80-113, sha256_12 1c80f4af4d78) ->
``vllm_topk_softmax`` (:26-40) -> ``_custom_ops.topk_softmax`` (_custom_ops.py:2300-2317) -> ``topk_softmax``
(csrc/libtorch_stable/moe/topk_softmax_kernels.cu:822-860) for the Qwen3.5-397B router: bf16 logits [M, 512], top-10,
renormalize (config norm_topk_prob), no correction bias, ``is_padding`` = vLLM's persistent padding mask
(VLLM_MOE_SKIP_PADDING=1 default; all false here: FastAFD's decode batches carry no padding rows, and a padding row
would get id -1, which the AFD dispatcher does not take). ``token_expert_indices`` is vLLM's third output
(int32 [M, K]); FastAFD does not use it, it is written as in vLLM. vLLM also allocates an M x 512 fp32 workspace that
the 512-expert path never reads; no kernel runs for it, so it is not allocated.
"""

from __future__ import annotations

import functools

import torch

_PAD: dict[torch.device, torch.Tensor] = {}


@functools.cache
def _module():
    from minisgl.kernel.utils import load_jit

    return load_jit(
        "vllm_topk_softmax",
        cuda_files=["vllm_topk_softmax.cu"],
        cuda_wrappers=[("topk_softmax", "VllmTopkSoftmaxKernel::run")],
    )


def _is_padding(m: int, device: torch.device) -> torch.Tensor:
    """A persistent all-false padding mask (vLLM: input_buffers.is_padding, a zeros(max_num_tokens) bool buffer).
    Grown outside CUDA-graph capture: the eager warm-up at each size runs before capture."""
    buf = _PAD.get(device)
    if buf is None or buf.numel() < m:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"vllm_topk_softmax: padding mask of {m} rows first requested inside CUDA-graph capture")
        buf = torch.zeros(max(m, 8192), dtype=torch.bool, device=device)
        _PAD[device] = buf
    return buf[:m]


def vllm_topk_softmax_supported(gating_output: torch.Tensor, topk_ids: torch.Tensor) -> bool:
    return (gating_output.dim() == 2 and gating_output.shape[1] == 512
            and gating_output.dtype == torch.bfloat16 and gating_output.is_contiguous()
            and topk_ids.dtype == torch.int32)


def vllm_topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor, gating_output: torch.Tensor,
                      renormalize: bool) -> None:
    if not vllm_topk_softmax_supported(gating_output, topk_ids):
        raise ValueError("vllm_topk_softmax: bf16 contiguous [M, 512] logits and int32 ids required, got "
                         f"{tuple(gating_output.shape)} {gating_output.dtype} / {topk_ids.dtype}")
    m = gating_output.shape[0]
    token_expert_indices = torch.empty_like(topk_ids)   # fused_topk_router.py:101-103
    pad = _is_padding(m, gating_output.device)
    _module().topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, bool(renormalize),
                           pad.view(torch.uint8))


__all__ = ["vllm_topk_softmax", "vllm_topk_softmax_supported"]
