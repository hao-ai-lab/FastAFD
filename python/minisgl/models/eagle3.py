"""EAGLE3 draft model for AFD speculative decoding (attention side).

Port of the SGLang/vLLM ``LlamaForCausalLMEagle3`` forward for one-layer EAGLE3
drafts such as ``lmsys/Qwen3-235B-A22B-EAGLE3``. The draft conditions on the
target's residual stream after ``eagle_aux_hidden_state_layer_ids`` (the
concatenation passes ``fc``) and on the next token:

- draft position ``p`` pairs the target feature at ``p`` with token ``x_{p+1}``;
  the layer input is ``cat(input_layernorm(embed(x_{p+1})), hidden_norm(h_p))``,
  so its K/V depend only on that pair. Committed context therefore needs only
  the K/V projections (``write_context``), no attention.
- drafting (``propose``) runs the layer one position at a time: step 1 at the last
  committed position (feature ``fc(h_{s-1})``, token = bonus), then steps 2..k at
  positions ``s .. s+k-2`` on the previous step's pre-norm output and drafted
  token. Each step attends causally to the draft KV through its own position
  (TRT-LLM decode, one query per request).

Draft KV is one layer, paged with the target page table (a context entry lives
at the same token slot as the target's KV). The draft vocabulary maps to target
ids through ``d2t`` (``target = draft + d2t[draft]``) when the checkpoint has one;
otherwise the target ``lm_head`` scores the full vocabulary.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def resolve_draft_path(path: str) -> Path:
    local = Path(path)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path, allow_patterns=["*.json", "*.safetensors"]))


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # transformers Qwen3RMSNorm: normalize in fp32, cast back, then scale.
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


DRAFT_ATTN_AUTO = -1


def _draft_attn_sms() -> int:
    """MINISGL_AFD_DRAFT_ATTN_SMS: the SM count the draft's attention launch is sized by (unset or 0:
    FlashInfer's count; "auto": see draft_attn_launch_sms). trtllm-gen splits a request's KV over several
    CTAs only while requests x KV heads stay below that count, so a batch just past it runs unsplit in two
    waves (MiniMax-M2.5 draft, 32K, 20 requests: 160 units on 152 SMs, 481 us per call) and a batch near
    half of it leaves SMs idle; a larger count forces the split (608: 421 us). The partial results are
    reduced, so outputs move at rounding level."""
    raw = os.environ.get("MINISGL_AFD_DRAFT_ATTN_SMS", "0") or "0"
    if raw == "auto":
        return DRAFT_ATTN_AUTO
    sms = int(raw)
    if sms < 0:
        raise ValueError(f"MINISGL_AFD_DRAFT_ATTN_SMS={sms} must be >= 0 or auto")
    return sms


def draft_attn_launch_sms(setting: int, units: int, device_sms: int) -> int:
    """The count to size one draft attention launch by (0 = FlashInfer's own). units = requests x KV heads.
    auto: force a 4x-device split when the unsplit launch would overflow one wave or fill under 85% of the
    SMs (MiniMax draft: 32K x 20 requests 1898 -> 1652 us per 3-step propose, 64K x 10 -> -3%); leave a
    near-full single wave alone (32K x 17: 136 units, the split is 1.7% slower)."""
    if setting != DRAFT_ATTN_AUTO:
        return setting
    if units > device_sms or units < 0.85 * device_sms:
        return 4 * device_sms
    return 0


def _norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm of the draft: FlashInfer's one-kernel norm on CUDA (as SGLang serves EAGLE3)."""
    if x.is_cuda:
        from flashinfer import rmsnorm

        return rmsnorm(x, weight, eps)
    return _rms_norm(x, weight, eps)


@dataclass(frozen=True)
class Eagle3Config:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: tuple[tuple[str, float], ...] | None
    target_layer_ids: tuple[int, ...]

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of draft K and V per token slot (one bf16 layer)."""
        return 2 * self.num_kv_heads * self.head_dim * 2


def load_eagle3_config(path: str, target_num_layers: int | None = None) -> Eagle3Config:
    raw = json.loads((resolve_draft_path(path) / "config.json").read_text())
    if int(raw.get("num_hidden_layers", 1)) != 1:
        raise ValueError(f"EAGLE3 draft with {raw.get('num_hidden_layers')} layers is not supported")
    eagle = raw.get("eagle_config") or {}
    if eagle.get("use_aux_hidden_state") is False:
        raise ValueError("EAGLE3 draft without aux hidden states is not supported")
    layer_ids = eagle.get("eagle_aux_hidden_state_layer_ids")
    # Decoder layers whose outputs feed fc. SGLang (and SpecForge, which trains these
    # drafts) reads the config ids as layer outputs; vLLM reads id k as the input of
    # layer k (the output of k - 1). MINISGL_AFD_SPEC_EAGLE3_LAYERS overrides them.
    override = os.environ.get("MINISGL_AFD_SPEC_EAGLE3_LAYERS", "").strip()
    if override:
        layer_ids = [int(x) for x in override.split(",")]
    if not layer_ids:
        if target_num_layers is None:
            raise ValueError("EAGLE3 config lacks eagle_aux_hidden_state_layer_ids")
        # SGLang/vLLM default: outputs of layers 1, L/2 - 1 and L - 4.
        layer_ids = [1, target_num_layers // 2 - 1, target_num_layers - 4]
    scaling = raw.get("rope_scaling")
    if scaling:
        if scaling.get("rope_type", scaling.get("type")) != "llama3":
            raise ValueError(f"EAGLE3 rope_scaling {scaling} is not supported")
        scaling = tuple(sorted((k, float(v)) for k, v in scaling.items() if k not in ("rope_type", "type")))
    return Eagle3Config(
        hidden_size=int(raw["hidden_size"]),
        num_heads=int(raw["num_attention_heads"]),
        num_kv_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]),
        intermediate_size=int(raw["intermediate_size"]),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        rope_theta=float(raw.get("rope_theta", 10000.0)),
        rope_scaling=scaling or None,
        target_layer_ids=tuple(int(x) for x in layer_ids),
    )


def _inv_freq(cfg: Eagle3Config, device: torch.device) -> torch.Tensor:
    dim = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
    if cfg.rope_scaling is None:
        return inv
    s = dict(cfg.rope_scaling)
    factor, low, high = s["factor"], s["low_freq_factor"], s["high_freq_factor"]
    old = s["original_max_position_embeddings"]
    low_wavelen, high_wavelen = old / low, old / high
    wavelen = 2 * math.pi / inv
    scaled = torch.where(wavelen > low_wavelen, inv / factor, inv)
    smooth = (old / wavelen - low) / (high - low)
    mid = (1 - smooth) * scaled / factor + smooth * scaled
    medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(medium, mid, scaled)


class Eagle3Draft:
    """Draft weights, its paged KV pool, the context projection and the drafting chain."""

    def __init__(
        self,
        path: str,
        *,
        device: torch.device,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        num_pages: int,
        page_size: int,
        max_positions: int,
        target_num_layers: int,
    ) -> None:
        from safetensors.torch import load_file

        self.cfg = load_eagle3_config(path, target_num_layers)
        cfg = self.cfg
        self.device = device
        self.dtype = torch.bfloat16
        self.page_size = int(page_size)
        if embed_weight.shape[1] != cfg.hidden_size:
            raise ValueError(f"EAGLE3 hidden {cfg.hidden_size} != target hidden {embed_weight.shape[1]}")
        weights: dict[str, torch.Tensor] = {}
        for file in sorted(resolve_draft_path(path).glob("*.safetensors")):
            weights.update(load_file(str(file), device=str(device)))
        self.d2t = weights.pop("d2t", None)
        weights.pop("t2d", None)
        self.w = {k: v.to(self.dtype) for k, v in weights.items()}
        self.embed_weight = self.w.pop("embed_tokens.weight", embed_weight)
        self.lm_head_weight = self.w.pop("lm_head.weight", lm_head_weight)
        if self.d2t is not None:
            self.d2t = self.d2t.to(device=device, dtype=torch.int64)
            if self.d2t.shape[0] != self.lm_head_weight.shape[0]:
                raise ValueError(f"EAGLE3 d2t has {self.d2t.shape[0]} entries, lm_head {self.lm_head_weight.shape[0]} rows")
        self._check_weights()
        shape = (int(num_pages), self.page_size, cfg.num_kv_heads, cfg.head_dim)
        self.k_cache = torch.zeros(shape, dtype=self.dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=self.dtype, device=device)
        # TRT-LLM decode scratch for the drafting chain's attention.
        self.workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.attn_sms = _draft_attn_sms()
        self.device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        freqs = torch.outer(torch.arange(int(max_positions), device=device, dtype=torch.float32), _inv_freq(cfg, device))
        # [positions, head_dim]: cos | sin halves, FP32 (FlashInfer's NeoX RoPE).
        self.cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
        self.max_positions = int(max_positions)
        # The K/V rows of q_proj's input stay separate so context writes skip q.
        p = "midlayer.self_attn."
        self.w_kv = torch.cat((self.w[p + "k_proj.weight"], self.w[p + "v_proj.weight"]), dim=0)
        self.w_qkv = torch.cat((self.w[p + "q_proj.weight"], self.w_kv), dim=0)
        self.w_gate_up = torch.cat((self.w["midlayer.mlp.gate_proj.weight"], self.w["midlayer.mlp.up_proj.weight"]), dim=0)

    def _check_weights(self) -> None:
        cfg = self.cfg
        need = ["fc.weight", "norm.weight"] + [f"midlayer.{n}" for n in (
            "hidden_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        )]
        missing = [k for k in need if k not in self.w]
        if missing:
            raise ValueError(f"EAGLE3 draft checkpoint lacks {missing}")
        fc_in = self.w["fc.weight"].shape[1]
        if fc_in != len(cfg.target_layer_ids) * cfg.hidden_size:
            raise ValueError(f"EAGLE3 fc expects {fc_in} features, config implies "
                             f"{len(cfg.target_layer_ids)} x {cfg.hidden_size}")
        if self.w["midlayer.self_attn.q_proj.weight"].shape[1] != 2 * cfg.hidden_size:
            raise ValueError("EAGLE3 q_proj must take cat(embed, hidden) of 2 x hidden_size")

    # ----- helpers -----

    def _rope_(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> None:
        """NeoX RoPE in place on ``[T, heads * head_dim]`` rows (row-strided views are fine)."""
        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        apply_rope_with_cos_sin_cache_inplace(positions, q, k, self.cfg.head_dim, self.cos_sin, is_neox=True)

    def _layer_input(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """cat(input_layernorm(embed(tokens)), hidden_norm(hidden)): [T, 2H]."""
        eps = self.cfg.rms_norm_eps
        x = torch.empty((tokens.shape[0], 2 * self.cfg.hidden_size), dtype=self.dtype, device=self.device)
        h = self.cfg.hidden_size
        x[:, :h] = _norm(F.embedding(tokens.to(torch.int64), self.embed_weight), self.w["midlayer.input_layernorm.weight"], eps)
        x[:, h:] = _norm(hidden, self.w["midlayer.hidden_norm.weight"], eps)
        return x

    def _store(self, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
        cfg = self.cfg
        self.k_cache.view(-1, cfg.num_kv_heads, cfg.head_dim)[slots] = k
        self.v_cache.view(-1, cfg.num_kv_heads, cfg.head_dim)[slots] = v

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """fc over the concatenated target features: [T, nL*H] -> [T, H]."""
        return F.linear(features.to(self.dtype), self.w["fc.weight"])

    # ----- context -----

    @torch.inference_mode()
    def write_context(
        self,
        features: torch.Tensor,
        next_tokens: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Store draft K/V at ``slots`` for target positions (feature at p, token x_{p+1})."""
        if features.shape[0] == 0:
            return
        cfg = self.cfg
        x = self._layer_input(next_tokens, self.project_features(features))
        kv = F.linear(x, self.w_kv)
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        # Only K is rotated here; a one-head scratch stands in for the query.
        scratch = torch.empty((kv.shape[0], cfg.head_dim), dtype=self.dtype, device=self.device)
        self._rope_(scratch, kv[:, :kv_dim], positions)
        k = kv[:, :kv_dim].view(-1, cfg.num_kv_heads, cfg.head_dim)
        v = kv[:, kv_dim:].view(-1, cfg.num_kv_heads, cfg.head_dim)
        self._store(k, v, slots.to(torch.int64))

    # ----- drafting -----

    def _attend(self, q: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor, max_kv_len: int) -> torch.Tensor:
        """The draft's decode attention; MINISGL_AFD_DRAFT_ATTN_SMS sizes its launch (see _draft_attn_sms)."""
        import flashinfer.decode as fid
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        cfg = self.cfg
        call = lambda: trtllm_batch_decode_with_kv_cache(  # noqa: E731
            query=q, kv_cache=(self.k_cache, self.v_cache),
            workspace_buffer=self.workspace, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_kv_len,
            bmm1_scale=cfg.head_dim ** -0.5, bmm2_scale=1.0, kv_layout="NHD", out_dtype=self.dtype,
        )
        sms = draft_attn_launch_sms(self.attn_sms, int(q.shape[0]) * cfg.num_kv_heads, self.device_sms)
        if sms <= 0:
            return call()
        # The count is looked up per call (flashinfer.decode.get_device_sm_count);
        # restore whatever was there.
        prev = fid.get_device_sm_count
        fid.get_device_sm_count = lambda device, _s=sms: _s
        try:
            return call()
        finally:
            fid.get_device_sm_count = prev

    @torch.inference_mode()
    def propose(
        self,
        last_features: torch.Tensor,
        bonus: torch.Tensor,
        start: torch.Tensor,
        page_rows: torch.Tensor,
        steps: int,
        max_kv_len: int,
    ) -> torch.Tensor:
        """Draft ``steps`` tokens after each request's bonus token.

        ``bonus[r]`` sits at position ``start[r]`` (device: the committed length);
        ``last_features[r]`` is the target feature at ``start[r] - 1``, whose context
        K/V (paired with the bonus) is already stored. ``page_rows[r]`` is the request's
        target page-table row in token slots, allocated through ``start + steps - 1``.
        Returns ``[R, steps]`` int64 target-vocabulary drafts.
        """
        cfg = self.cfg
        rows = int(bonus.shape[0])
        if rows == 0:
            return torch.empty((0, steps), dtype=torch.int64, device=self.device)
        eps = cfg.rms_norm_eps
        start = start.to(torch.int64)
        page = self.page_size
        block_tables = (page_rows[:, ::page] // page).to(torch.int32)
        hidden = self.project_features(last_features)
        token = bonus.to(torch.int64)
        q_dim = cfg.num_heads * cfg.head_dim
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        out = torch.empty((rows, steps), dtype=torch.int64, device=self.device)
        # Every step's positions, KV lengths and store slots up front: [steps, rows], one row per step.
        pos_all = (start - 1)[None, :] + torch.arange(steps, dtype=torch.int64, device=self.device)[:, None]
        seq_all = (pos_all + 1).to(torch.int32)
        slots_all = page_rows.gather(1, pos_all.t()).t().contiguous().to(torch.int64)
        for step in range(steps):
            pos = pos_all[step]
            x = self._layer_input(token, hidden)
            qkv = F.linear(x, self.w_qkv)
            self._rope_(qkv[:, :q_dim], qkv[:, q_dim : q_dim + kv_dim], pos)
            q = qkv[:, :q_dim].view(rows, cfg.num_heads, cfg.head_dim)
            if step > 0:
                k = qkv[:, q_dim : q_dim + kv_dim].view(rows, cfg.num_kv_heads, cfg.head_dim)
                v = qkv[:, q_dim + kv_dim :].view(rows, cfg.num_kv_heads, cfg.head_dim)
                self._store(k, v, slots_all[step])
            attn = self._attend(q.contiguous(), block_tables, seq_all[step], int(max_kv_len))
            h = F.linear(attn.reshape(rows, q_dim), self.w["midlayer.self_attn.o_proj.weight"]) + hidden
            x = _norm(h, self.w["midlayer.post_attention_layernorm.weight"], eps)
            gate, up = F.linear(x, self.w_gate_up).chunk(2, dim=-1)
            hidden = h + F.linear(F.silu(gate) * up, self.w["midlayer.mlp.down_proj.weight"])
            logits = F.linear(_norm(hidden, self.w["norm.weight"], eps), self.lm_head_weight)
            token = logits.argmax(dim=-1)
            if self.d2t is not None:
                token = token + self.d2t[token]
            out[:, step] = token
        return out


__all__ = ["Eagle3Config", "Eagle3Draft", "load_eagle3_config"]
