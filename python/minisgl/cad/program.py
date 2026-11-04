"""CAD decode-step programs: the fixed task order of one decode step on an attention GPU and on a model GPU (pure Python).

A program is a tuple of Task(kind, layer, mb); the runtime (cad/runtime.py) calls `getattr(tasks, task.kind)(task.layer,
task.mb)` in this order, then bumps the epoch. Whole-step tasks (embed, final, sample) carry layer = mb = -1.

  attention GPU   per layer l, per microbatch mb:   pre(l, mb)    wait for (l, mb)'s q|k|v rows, store K / V into the pool
                                                    attn(l, mb)   decode FMHA into the model GPU's o rows, release o_flag
  model GPU       embed; per mb hqkv(0, mb); per layer l, per mb: hpost(l, mb), ffn(l, mb), hqkv(l + 1, mb) (l + 1 < L);
                  final; sample

m = 2 (the microbatch ping-pong): while the model GPU runs mb 0's post-attention + FFN of layer l and puts its
layer l + 1 rows, the attention GPU runs mb 1's FMHA of layer l, and the other way round. m = 1 is the named control
(no overlap). Nothing in the order depends on the step's data, so one CUDA graph per bucket replays it (no
planner, no program cache).

Flag traffic (flag_traffic): every qkv flag (attention GPU a, l, mb) is released once per step by a's model GPU in
hqkv(l, mb) and waited once by a in pre(l, mb); every o flag (model GPU m, l, mb, peer j) is released once by m's peer j in
attn(l, mb) and waited once by m in hpost(l, mb). Any other count means a step that hangs or races.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

ATTN_KINDS = ("pre", "attn")
MODEL_KINDS = ("embed", "hqkv", "hpost", "ffn", "final", "sample")
STEP_KINDS = ("embed", "final", "sample")   # whole-step tasks: layer = mb = -1
ROLES = ("attn", "model")


@dataclass(frozen=True)
class Task:
    kind: str
    layer: int
    mb: int

    def __post_init__(self) -> None:
        if self.kind not in ATTN_KINDS + MODEL_KINDS:
            raise ValueError(f"unknown CAD task kind {self.kind!r}")
        whole = self.kind in STEP_KINDS
        if whole != (self.layer == -1 and self.mb == -1):
            raise ValueError(f"{self.kind}: whole-step tasks take layer = mb = -1, per-layer tasks layer, mb >= 0; "
                             f"got ({self.layer}, {self.mb})")


def _check(num_layers: int, num_mb: int) -> None:
    if num_layers < 1 or num_mb not in (1, 2):
        raise ValueError(f"CAD programs need num_layers >= 1 and num_mb in (1, 2) (m = 2 overlapped, m = 1 control), "
                         f"got L {num_layers}, m {num_mb}")


def attn_program(num_layers: int, num_mb: int) -> tuple[Task, ...]:
    _check(num_layers, num_mb)
    return tuple(Task(kind, layer, mb) for layer in range(num_layers) for mb in range(num_mb) for kind in ATTN_KINDS)


def model_program(num_layers: int, num_mb: int) -> tuple[Task, ...]:
    _check(num_layers, num_mb)
    out = [Task("embed", -1, -1)]
    out += [Task("hqkv", 0, mb) for mb in range(num_mb)]
    for layer in range(num_layers):
        for mb in range(num_mb):
            out += [Task("hpost", layer, mb), Task("ffn", layer, mb)]
            if layer + 1 < num_layers:
                out.append(Task("hqkv", layer + 1, mb))
    out += [Task("final", -1, -1), Task("sample", -1, -1)]
    return tuple(out)


def program(role: str, num_layers: int, num_mb: int) -> tuple[Task, ...]:
    if role == "attn":
        return attn_program(num_layers, num_mb)
    if role == "model":
        return model_program(num_layers, num_mb)
    raise ValueError(f"CAD role {role!r}: one of {ROLES}")


def flag_traffic(num_layers: int, num_mb: int, num_attn: int, num_model: int) -> dict:
    """{flag id: (releases, waits)} of one decode step over every GPU's program, with layout.model_of / peers_of's
    block mapping. Flag ids: ("qkv", a, l, mb) on attention GPU a, ("o", m, l, mb, j) on model GPU m."""
    from minisgl.cad.layout import model_of, peer_index, peers_of

    rel, wait = Counter(), Counter()
    for a in range(num_attn):
        m, j = model_of(a, num_attn, num_model), peer_index(a, num_attn, num_model)
        for t in attn_program(num_layers, num_mb):
            if t.kind == "pre":
                wait[("qkv", a, t.layer, t.mb)] += 1
            elif t.kind == "attn":
                rel[("o", m, t.layer, t.mb, j)] += 1
    for m in range(num_model):
        peers = peers_of(m, num_attn, num_model)
        for t in model_program(num_layers, num_mb):
            if t.kind == "hqkv":
                for a in peers:
                    rel[("qkv", a, t.layer, t.mb)] += 1
            elif t.kind == "hpost":
                for j in range(len(peers)):
                    wait[("o", m, t.layer, t.mb, j)] += 1
    return {k: (rel[k], wait[k]) for k in sorted(set(rel) | set(wait))}


def check_balanced(num_layers: int, num_mb: int, num_attn: int, num_model: int) -> None:
    bad = {k: v for k, v in flag_traffic(num_layers, num_mb, num_attn, num_model).items() if v != (1, 1)}
    if bad:
        raise ValueError(f"CAD flag traffic not balanced (release, wait) != (1, 1): {sorted(bad.items())[:8]}")


__all__ = ["ATTN_KINDS", "MODEL_KINDS", "ROLES", "STEP_KINDS", "Task", "attn_program", "check_balanced",
           "flag_traffic", "model_program", "program"]
