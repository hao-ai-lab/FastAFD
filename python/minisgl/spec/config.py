"""Speculative decoding for AFD: configuration, acceptance, and the reply encoding.

A speculative decode step verifies ``block`` tokens per request: the previous
step's bonus token followed by ``block - 1`` draft tokens. The target scores all
of them in one forward; the request accepts the longest draft prefix that matches
the target's greedy prediction and appends the target's own next token (the
bonus). A step therefore commits ``n = accepted + 1`` tokens, 1 <= n <= block.

The coordinator schedules the verify step as an extend of ``block`` tokens per
request on top of the committed KV and learns ``n`` from the AG reply. KV
written for rejected positions is never read: the next step starts at the
committed length and overwrites it. Pages are allocated ``block`` positions
ahead of the step so the draft model can stage its block at those positions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

_ENV = "MINISGL_AFD_SPEC"


@dataclass(frozen=True)
class AfdSpecConfig:
    method: str = ""
    block: int = 1
    draft_path: str = ""
    sim_accept: float = 0.0
    sim_seed: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.method)

    @property
    def lookahead(self) -> int:
        """Positions past a step's tokens whose pages are allocated with the step."""
        return self.block if self.enabled else 0


def afd_spec_config() -> AfdSpecConfig:
    """Read the speculative decoding config from MINISGL_AFD_SPEC*.

    MINISGL_AFD_SPEC=eagle3 enables it; MINISGL_AFD_SPEC_BLOCK is the verify block
    (bonus + drafts); MINISGL_AFD_SPEC_DRAFT is the EAGLE3 draft checkpoint;
    MINISGL_AFD_SPEC_SIM_ACCEPT > 0 forces that mean acceptance length (tokens per
    step), committing the real draft tokens, for performance studies only.
    """
    method = os.environ.get(_ENV, "").strip().lower()
    if method in ("", "0", "none", "off"):
        return AfdSpecConfig()
    if method != "eagle3":
        raise ValueError(f"{_ENV}={method!r} is not supported; use eagle3")
    block = int(os.environ.get(f"{_ENV}_BLOCK", "8"))
    if block < 2:
        raise ValueError(f"{_ENV}_BLOCK must be >= 2, got {block}")
    draft = os.environ.get(f"{_ENV}_DRAFT", "").strip()
    if not draft:
        raise ValueError(f"{_ENV}={method} needs {_ENV}_DRAFT (the draft checkpoint)")
    sim = float(os.environ.get(f"{_ENV}_SIM_ACCEPT", "0") or 0)
    if sim and not 1.0 <= sim <= block:
        raise ValueError(f"{_ENV}_SIM_ACCEPT must be in [1, {block}], got {sim}")
    seed = int(os.environ.get(f"{_ENV}_SIM_SEED", "0") or 0)
    return AfdSpecConfig(method=method, block=block, draft_path=draft, sim_accept=sim, sim_seed=seed)


def check_spec_supported(spec: AfdSpecConfig, *, prefix_cache: str) -> None:
    """Raise at startup for a deployment speculative decoding does not implement.

    A verify step extends every request by ``block`` tokens and then rolls its length
    back to what it commits, with the attention workers' own page allocator staging
    pages ``lookahead`` positions ahead. Explicit page grants (prefix caching hands the
    workers the coordinator's page ids and expects lengths to agree every step) do not
    model that.
    """
    if not spec.enabled:
        return
    if prefix_cache != "naive":
        raise ValueError(f"{_ENV}={spec.method} does not support MINISGL_AFD_PREFIX_CACHE={prefix_cache}")


def greedy_accept(
    block_ids: torch.Tensor,
    posterior: torch.Tensor,
    *,
    sim_accept: float = 0.0,
    generator: torch.Generator | None = None,
    uniforms: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy verification of ``[R, B]`` blocks.

    ``block_ids[r] = [bonus, d_1 .. d_{B-1}]`` are the verified inputs and
    ``posterior[r, j]`` is the target's greedy token after input ``j``. Returns
    ``(n, emitted)``: ``n[r]`` tokens are committed, and ``emitted[r, :n[r]]`` are the
    new output tokens (the accepted drafts, then the bonus at index ``n[r] - 1``).

    ``sim_accept > 0`` replaces the acceptance with a draw whose mean is exactly
    ``sim_accept``: ``n = 1 + floor(sim_accept - 1 + U)``, U ~ U[0, 1). The forced
    drafts are committed as if accepted, so outputs are not the target's.
    ``uniforms`` ([R] on the device) supplies U directly (CUDA-graph replays).
    """
    if block_ids.shape != posterior.shape or block_ids.dim() != 2:
        raise ValueError(
            f"greedy_accept expects matching [R, B] tensors, got {tuple(block_ids.shape)} "
            f"and {tuple(posterior.shape)}"
        )
    rows, block = block_ids.shape
    if sim_accept > 0:
        u = uniforms
        if u is None:
            u = torch.rand((rows,), generator=generator, device="cpu").to(block_ids.device)
        accepted = torch.floor(u + (float(sim_accept) - 1.0)).to(torch.int64)
    else:
        match = (block_ids[:, 1:] == posterior[:, :-1]).to(torch.int64)
        accepted = match.cumprod(dim=1).sum(dim=1)
    accepted = accepted.clamp_(0, block - 1)
    bonus = posterior.gather(1, accepted.unsqueeze(1))
    emitted = torch.zeros_like(block_ids)
    emitted[:, : block - 1] = block_ids[:, 1:]
    emitted.scatter_(1, accepted.unsqueeze(1), bonus)
    return accepted + 1, emitted


def pack_spec_reply(n: torch.Tensor, emitted: torch.Tensor) -> torch.Tensor:
    """One int32 row per request: ``[n, emitted_0 .. emitted_{B-1}]`` (flattened)."""
    return torch.cat((n.to(torch.int32).unsqueeze(1), emitted.to(torch.int32)), dim=1).flatten()


def unpack_spec_reply(flat: list[int], block: int) -> list[list[int]]:
    """Per-request committed tokens from ``pack_spec_reply`` output read on the host."""
    width = block + 1
    if len(flat) % width:
        raise ValueError(f"spec reply length {len(flat)} is not a multiple of {width}")
    out = []
    for row in range(len(flat) // width):
        base = row * width
        n = int(flat[base])
        if not 1 <= n <= block:
            raise ValueError(f"spec reply row {row} commits {n} tokens, block={block}")
        out.append([int(t) for t in flat[base + 1 : base + 1 + n]])
    return out


__all__ = [
    "AfdSpecConfig",
    "afd_spec_config",
    "greedy_accept",
    "pack_spec_reply",
    "unpack_spec_reply",
]
