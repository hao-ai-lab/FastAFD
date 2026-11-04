"""CAD model-GPU tasks: everything of a decode layer but the core attention, for the rows of the GPU's r attention GPUs.

  embed(-1, -1)  the step's token per row (host or last sampled, minisgl.cad.tokens) -> embedding = layer 0's input
  hqkv(l, mb)    fused add + RMSNorm, QKV FP8 GEMM, q/k cross-head norm + RoPE (q in place, K / V into send rows), then per
                 peer p one put of its B rows (q | k | v, segment-major) into p's window slot + release qkv_flag(l, mb)
  hpost(l, mb)   wait for the r o flags of (l, mb), O-proj FP8 GEMM over the r B o rows, fused add + RMSNorm, router
  ffn(l, mb)     the MoE round (minisgl.cad.m2n_dual.M2NDual: AG 1 -> EG -> AG 2) into the mb's MoE output rows
  final(-1, -1)  per mb the final fused add + RMSNorm
  sample(-1, -1) LM head over every row, greedy argmax, the tokens into their last-token slots
The runtime (minisgl.cad.runtime) walks program.model_program and bumps the epoch; this class is the task
executor (bind / fill / fill_dummy + the kinds). Every op is stream-ordered with fixed buffers per bucket (graphs).

Kernels come in as `ops` (ModelOps = the AFD path's: the MiniMaxM2AfdAGLayer modules, in-tree DeepGEMM FP8, flashinfer norms,
the minimax fused q/k norm + RoPE kernel, the AFD router; a fake in the CPU tests) and `hop` (kernel/cad_hop.py).
MiniMax-M2.5 only (cross-head q/k norm, no shared expert, no routed scaling).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .tokens import CadRow, CadTokenInputs, greedy


class ModelOps:
    """The AFD path's kernels, through the model's own modules (MiniMaxM2AfdDenseRouterForCausalLM)."""

    def input_norm(self, layer, x, residual):
        return layer.input_layernorm.forward(x, residual)

    def qkv(self, layer, h):
        return layer.self_attn.qkv_proj.forward(h)

    def qk_rope_send(self, layer, qkv, positions, k_send, v_send, loc):
        from minisgl.kernel import minimax_qk_cross_head_norm_rope_store_inplace

        a = layer.self_attn
        minimax_qk_cross_head_norm_rope_store_inplace(
            qkv, a.q_norm.weight, a.k_norm.weight, positions, a.attn.rotary.cos_sin_cache, k_send, v_send, loc,
            q_dim=a.attn.qo_attn_dim, kv_dim=a.attn.kv_attn_dim, head_dim=a.attn.head_dim, eps=a.q_norm.eps)

    def o_proj(self, layer, o):
        return layer.self_attn.o_proj.forward(o)

    def post_norm(self, layer, h, residual):
        return layer.post_attention_layernorm.forward(h, residual)

    def route(self, layer, h):
        t = layer.route(h)
        return t.topk_ids, t.topk_weights

    def embed(self, model, ids):
        return model.model.embed_tokens.forward(ids)

    def final_norm(self, model, x, residual):
        return model.model.norm.forward(x, residual)[0]

    def lm_head(self, model, h):
        lm = model.lm_head
        weight = (lm.tied_embedding or lm).weight      # TP1: ParallelLMHead.forward needs a batch context
        return torch.nn.functional.linear(h, weight, lm.bias)


class CadModelTasks:
    def __init__(self, *, window, model, moe, tokens: CadTokenInputs, num_layers: int, hidden: int, ops=None,
                 hop=None, device="cuda"):
        if hop is None:
            from minisgl.kernel import cad_hop as hop
        lay = window.layout
        if window.role != "model":
            raise ValueError("CadModelTasks runs on a model GPU window")
        if (tokens.r, tokens.num_mb, tokens.t_max) != (lay.r, lay.num_mb, lay.t_max) or lay.num_layers != num_layers:
            raise ValueError(f"tokens r / mb / t_max {tokens.r, tokens.num_mb, tokens.t_max} and layout "
                             f"{lay.r, lay.num_mb, lay.t_max, lay.num_layers} disagree (L {num_layers})")
        self.win, self.lay, self.model, self.moe, self.tok = window, lay, model, moe, tokens
        self.ops, self.hop = (ops or ModelOps()), hop
        self.L, self.m, self.r, self.hidden = int(num_layers), lay.num_mb, lay.r, int(hidden)
        n_max = lay.r * lay.t_max
        bf16 = torch.bfloat16
        self.k_send = [torch.zeros(n_max, lay.kv_dim, dtype=bf16, device=device) for _ in range(self.m)]
        self.v_send = [torch.zeros(n_max, lay.kv_dim, dtype=bf16, device=device) for _ in range(self.m)]
        self.loc = torch.arange(n_max, dtype=torch.int32, device=device)
        self.y = [torch.zeros(n_max, self.hidden, dtype=bf16, device=device) for _ in range(self.m)]
        self.final_h = torch.zeros(self.m * n_max, self.hidden, dtype=bf16, device=device)
        self.x, self.res, self.work = [None] * self.m, [None] * self.m, [None] * self.m
        self.bucket, self.where = 0, []

    def layer(self, l: int):
        return self.model.model.layers.op_list[int(l)]

    # ---- runtime protocol
    def bind(self, bucket: int) -> None:
        self.tok.bind(bucket)
        self.bucket = int(bucket)

    def fill(self, cmd) -> None:
        """The step's rows (AfdRunCadStepCmd.model_rows: CadRow per request of this GPU's r attention GPUs)."""
        rows: Sequence[CadRow] = cmd.model_rows
        self.where = self.tok.fill(rows)

    def fill_dummy(self, bucket: int) -> None:
        self.tok.fill_dummy()
        self.where = []

    @property
    def n(self) -> int:
        return self.r * self.bucket

    # ---- tasks
    def embed(self, layer: int, mb: int) -> None:
        e = self.ops.embed(self.model, self.tok.input_ids())
        for k in range(self.m):
            self.x[k], self.res[k] = e[self.tok.mb_rows(k)], None

    def hqkv(self, layer: int, mb: int) -> None:
        lay, win, ops, B, n = self.lay, self.win, self.ops, self.bucket, self.n
        L = self.layer(layer)
        h, self.res[mb] = ops.input_norm(L, self.x[mb], self.res[mb])
        qkv = ops.qkv(L, h)
        ks, vs = self.k_send[mb][:n], self.v_send[mb][:n]
        ops.qk_rope_send(L, qkv, self.tok.pos[self.tok.mb_rows(mb)], ks, vs, self.loc[:n])
        q16, k16, v16 = qkv.view(torch.int16), ks.view(torch.int16), vs.view(torch.int16)
        for p in range(self.r):
            a, b = lay.model_rows(p, B)
            dq, dk, dv = win.qkv_segments_at(p, mb, B)
            self.hop.put_rows([(q16[a:b, :lay.q_dim], dq), (k16[a:b], dk), (v16[a:b], dv)],
                              win.qkv_flag_at(p, layer, mb), win.epoch)

    def _o_rows(self, mb: int) -> torch.Tensor:
        """The r B o rows of mb as one [r B, o_dim] tensor: peer p's rows start at row p t_max of the o slot (a view at
        B = t_max, the steady state of a KV-full batch; else one strided copy)."""
        lay, B = self.lay, self.bucket
        o = self.win.o_slot(mb)
        if B == lay.t_max:
            return o
        return o.view(lay.r, lay.t_max, lay.o_dim)[:, :B].reshape(lay.r * B, lay.o_dim)

    def hpost(self, layer: int, mb: int) -> None:
        win, ops = self.win, self.ops
        L = self.layer(layer)
        self.hop.wait_range(win.o_flags(layer, mb), win.epoch, win.err, win.timeout_ns)
        h = ops.o_proj(L, self._o_rows(mb))
        h, self.res[mb] = ops.post_norm(L, h, self.res[mb])
        ids, wts = ops.route(L, h)
        self.work[mb] = (h, ids, wts)

    def ffn(self, layer: int, mb: int) -> None:
        h, ids, wts = self.work[mb]
        y = self.y[mb][:self.n]
        self.moe.run(layer, mb, h, ids, wts, y, self.n)
        self.x[mb], self.work[mb] = y, None

    def final(self, layer: int, mb: int) -> None:
        for k in range(self.m):
            self.final_h[self.tok.mb_rows(k)].copy_(self.ops.final_norm(self.model, self.x[k], self.res[k]))

    def sample(self, layer: int, mb: int) -> None:
        logits = self.ops.lm_head(self.model, self.final_h[:self.tok.rows])
        self.tok.record(greedy(logits))


__all__ = ["CadModelTasks", "ModelOps"]
