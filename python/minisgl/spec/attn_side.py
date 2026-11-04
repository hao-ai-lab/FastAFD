"""Attention-side speculative decoding: the EAGLE3 draft and the verify bookkeeping.

The AG worker owns everything request-local, so it owns speculation too: it
captures the target hidden states the draft conditions on, scores a verify block,
accepts, extends the draft's context KV, proposes the next block into its
token_pool, and replies with the committed tokens (spec.config.pack_spec_reply). The coordinator only learns how far
each request advanced.

Token layout of a verify step: every padded slot holds ``block`` tokens and each
microbatch lists its real requests before its padding. The finish runs over every
slot (static shapes, CUDA-graph capturable); ``batch.afd_spec_real_slots`` names
the slots whose reply rows the host keeps, in request order.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .config import AfdSpecConfig, greedy_accept, pack_spec_reply
from .device_inputs import SpecLengths, commit_lengths, spec_device_inputs_enabled

# The draft model is imported where it is used: `minisgl.models` pulls in the CUDA
# kernels at package scope, and the KV sizing below runs (spec off) wherever the
# attention state is sized, including the CPU tests.


def _draft_config(spec: AfdSpecConfig):
    from ..models.eagle3 import load_eagle3_config

    return load_eagle3_config(spec.draft_path, target_num_layers=0)


def draft_kv_bytes_per_token(spec: AfdSpecConfig) -> int:
    """Paged draft KV per token slot (the draft's attention layer)."""
    if not spec.enabled:
        return 0
    return _draft_config(spec).kv_bytes_per_token


def draft_ring_bytes(spec: AfdSpecConfig, rows: int, max_seq_len: int) -> int:
    """Fixed bytes of the draft's per-row state for ``rows`` request rows: the feature
    of each row's last prefill position, held until the next chunk supplies the
    following token."""
    del max_seq_len
    if not spec.enabled:
        return 0
    cfg = _draft_config(spec)
    return int(rows) * len(cfg.target_layer_ids) * cfg.hidden_size * 2


class AfdSpecAG:
    def __init__(self, spec: AfdSpecConfig, *, model: Any, runtime: Any, device: torch.device):
        self.spec = spec
        self.model = model
        self.runtime = runtime
        self.device = device
        state = runtime.state
        embed = model.model.embed_tokens
        lm_head = model.lm_head
        if int(getattr(embed, "tp_size", 1)) != 1:
            raise ValueError("AFD speculative decoding needs attention TP=1 (full-vocab embed/lm_head)")
        num_layers = len(model.model.layers.op_list)
        common = dict(
            device=device,
            embed_weight=embed.weight,
            lm_head_weight=(lm_head.tied_embedding or lm_head).weight,
            num_pages=int(state.num_pages) + 1,
            page_size=int(runtime.config.page_size),
            max_positions=int(runtime.token_pool.shape[1]) + 2 * spec.block,
        )
        rows = int(runtime.token_pool.shape[0])
        if any(hasattr(layer, "linear_attn") for layer in model.model.layers.op_list):
            raise ValueError("AFD speculative decoding supports full-attention targets only")
        from ..models.eagle3 import Eagle3Draft

        self.draft = Eagle3Draft(spec.draft_path, target_num_layers=num_layers, **common)
        cfg = self.draft.cfg
        # Feature of each row's last prefill position, written to the draft context
        # once the next chunk (or the first sampled token) supplies x_{p+1}.
        self._pending_feature = torch.empty(
            (rows, len(cfg.target_layer_ids) * cfg.hidden_size), dtype=torch.bfloat16, device=device
        )
        self._pending_pos: dict[int, int] = {}
        if max(cfg.target_layer_ids) >= num_layers:
            raise ValueError(
                f"draft target layers {cfg.target_layer_ids} exceed the target's {num_layers} layers"
            )
        # Decoder layer index -> column block of the concatenated feature.
        self.capture_cols = {int(layer): col for col, layer in enumerate(cfg.target_layer_ids)}
        # Host bound on a draft block's KV length (sizes the draft attention's
        # split grid; fixed so the finish can be graph-captured).
        self.max_kv_len = int(runtime.token_pool.shape[1]) + 2 * spec.block
        self.hidden_size = int(cfg.hidden_size)
        # MINISGL_AFD_SPEC_DEVICE_INPUTS=1 (spec.device_inputs): committed lengths stay on the device
        # so the next verify step can be launched before this one's reply reaches the host.
        self.device_inputs = spec_device_inputs_enabled()
        self.committed = torch.zeros((rows,), dtype=torch.int32, device=device)
        runtime._spec_committed_dev = self.committed
        runtime._spec_lengths = SpecLengths(int(spec.block)) if self.device_inputs else None
        # Simulated acceptance draws its uniforms on the device: a host draw copied in is a blocking
        # copy, so the host would wait for the step's forward before it could prepare the next step.
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(spec.sim_seed))
        self.stats_steps = 0
        self.stats_reqs = 0
        self.stats_tokens = 0
        # MINISGL_AFD_SPEC_TIMING=1: CUDA-event breakdown of each verify step.
        self.timing = os.environ.get("MINISGL_AFD_SPEC_TIMING", "0") == "1"
        self._marks: list[tuple[str, torch.cuda.Event]] = []
        self.timing_sums: dict[str, float] = {}
        self.timing_steps = 0

    # ----- timing -----

    def mark(self, name: str) -> None:
        if self.timing and not torch.cuda.is_current_stream_capturing():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._marks.append((name, event))

    def collect_timing(self) -> None:
        """Fold this step's marks into the running sums (synchronizes on the last mark)."""
        if not self.timing or len(self._marks) < 2:
            self._marks.clear()
            return
        self._marks[-1][1].synchronize()
        for (_, start), (name, end) in zip(self._marks, self._marks[1:]):
            self.timing_sums[name] = self.timing_sums.get(name, 0.0) + start.elapsed_time(end)
        self.timing_sums["total"] = self.timing_sums.get("total", 0.0) + self._marks[0][1].elapsed_time(
            self._marks[-1][1]
        )
        self.timing_steps += 1
        self._marks.clear()

    # ----- feature capture -----

    def new_feature_buffer(self, num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
        width = len(self.capture_cols) * self.hidden_size
        return torch.empty((int(num_tokens), width), dtype=dtype, device=self.device)

    # ----- verify -----

    @torch.inference_mode()
    def finish_verify(self, batch: Any, final: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Eager finish of a verify step; returns the packed reply of every slot."""
        batch.afd_spec_cached = [(int(r.table_idx), int(r.cached_len)) for r in batch.reqs]
        self.mark("forward")
        uniforms = None
        if self.spec.sim_accept > 0:
            slots = batch.input_ids.shape[0] // self.spec.block
            uniforms = torch.rand((slots,), generator=self._generator, device=self.device)
        return self.finish_slots(
            final, batch.input_ids, batch.positions, batch.out_loc, batch.state_indices,
            features, batch.afd_mb_subbatches, uniforms,
        )

    @torch.inference_mode()
    def finish_slots(
        self,
        final: torch.Tensor,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        state_indices: torch.Tensor,
        features: torch.Tensor,
        subs: list,
        uniforms: torch.Tensor | None,
    ) -> torch.Tensor:
        """Accept, commit, extend the draft context and propose for every padded slot.

        Every slot holds ``block`` tokens (padding slots are dummy sequences on the
        dummy row, whose writes land on the dummy page and token-pool row), so shapes depend only on the slot count and the whole finish can be
        captured in a CUDA graph. The host keeps the rows of the real slots.
        """
        reply, pending = self.finish_head(
            final, block_ids, positions, out_loc, state_indices, features, subs, uniforms
        )
        self.finish_draft(pending)
        return reply

    @torch.inference_mode()
    def finish_head(
        self,
        final: torch.Tensor,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        state_indices: torch.Tensor,
        features: torch.Tensor,
        subs: list,
        uniforms: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Accept and commit; returns the packed reply and the inputs of ``finish_draft``.

        The reply depends only on this part, so the host can read it (and prepare
        the next step) while the draft still runs.
        """
        block = self.spec.block
        tokens = block_ids.shape[0]
        slots = tokens // block
        posterior = self._greedy(final[:tokens]).view(slots, block)
        n, emitted = greedy_accept(
            block_ids[:tokens].view(slots, block).to(torch.int64), posterior,
            sim_accept=self.spec.sim_accept, uniforms=uniforms,
        )
        self.mark("verify_head")
        # The committed length stays on the device: the next block starts after the
        # n committed tokens. The host learns n with the reply (apply_reply).
        start = positions[:tokens].view(slots, block)[:, 0].to(torch.int64) + n
        if self.device_inputs:
            commit_lengths(self.committed, state_indices[:tokens].view(slots, block)[:, 0].to(torch.int64), start,
                           int(self.runtime._verify_dummy_req.table_idx))
        pending = {
            "n": n,
            "emitted": emitted,
            "start": start,
            "bonus": emitted.gather(1, (n - 1).unsqueeze(1)).squeeze(1),
            "table_idx": state_indices[:tokens].view(slots, block)[:, 0].to(torch.int64),
            "features": features[:tokens],
            "positions": positions[:tokens],
            "out_loc": out_loc[:tokens],
        }
        return pack_spec_reply(n, emitted), pending

    @torch.inference_mode()
    def finish_draft(self, pending: dict[str, torch.Tensor]) -> None:
        """Extend the draft context and write the next block to the token pool."""
        n, emitted, features = pending["n"], pending["emitted"], pending["features"]
        start, bonus, table_idx = pending["start"], pending["bonus"], pending["table_idx"]
        slots = n.shape[0]
        # Draft position p pairs the feature at p with x_{p+1}: the committed
        # tokens emitted[:, :n] (the bonus last). Positions past n get the
        # rejected drafts, rewritten by the drafting chain or the next step
        # before any query reads them.
        self.draft.write_context(features, emitted.flatten(), pending["positions"], pending["out_loc"])
        self.mark("draft_context")
        last = features.view(slots, self.spec.block, -1).gather(
            1, (n - 1).view(slots, 1, 1).expand(-1, 1, features.shape[-1])
        ).squeeze(1)
        self._propose_eagle3(last, bonus, start, table_idx)
        self.mark("propose")

    def real_reply(self, packed_all: list[int], real_slots: list[int]) -> list[int]:
        """Keep the reply rows of the real slots, in request order."""
        width = self.spec.block + 1
        out: list[int] = []
        for slot in real_slots:
            out.extend(packed_all[slot * width : (slot + 1) * width])
        return out

    def apply_reply(
        self, cached: list[tuple[int, int]], packed: list[int], committed: list[int] | None = None,
        gens: list[int] | None = None,
    ) -> None:
        """Advance the AG's committed lengths once a verify reply is on the host.

        ``committed`` (device-input mode): each real slot's absolute length after the step, from
        the reply tail; the cached lengths recorded at materialize are only upper bounds then.
        """
        width = self.spec.block + 1
        runtime = self.runtime
        lengths = getattr(runtime, "_spec_lengths", None)
        total = 0
        for row, (table_idx, cached_len) in enumerate(cached):
            accepted = int(packed[row * width])
            if lengths is not None:
                exact = int(committed[row]) if committed is not None else cached_len + accepted
                if lengths.finish_step(table_idx, gens[row] if gens else 0, exact):
                    runtime._table_cached_len[table_idx] = exact + self.spec.block * lengths.inflight.get(int(table_idx), 0)
            else:
                runtime._table_cached_len[table_idx] = cached_len + accepted
            total += accepted
        self.stats_steps += 1
        self.stats_reqs += len(cached)
        self.stats_tokens += total

    # ----- prefill -----

    @torch.inference_mode()
    def finish_prefill(self, batch: Any, features: torch.Tensor, next_tokens: torch.Tensor) -> None:
        """Extend the draft context with the prompt and draft after each sampled token."""
        self._finish_prefill_eagle3(batch, features, next_tokens)

    def _finish_prefill_eagle3(self, batch: Any, features: torch.Tensor, next_tokens: torch.Tensor) -> None:
        """EAGLE3 context for a prefill chunk, then the first drafts of finished prompts.

        Position p needs x_{p+1}: inside the chunk that is the chunk's next token; for
        the chunk's last position it is the next chunk's first token (the feature is
        held in ``_pending_feature`` until then) or, when the prompt ends, the sampled
        token, which also starts the drafting chain.
        """
        reqs = list(batch.reqs)
        lens = [int(r.device_len) - int(r.cached_len) for r in reqs]
        real = sum(lens)
        if real == 0:
            return
        dev = self.device
        ids = batch.input_ids[:real].to(torch.int64)
        positions = batch.positions[:real].to(torch.int64)
        slots = batch.out_loc[:real].to(torch.int64)
        page_table = self.runtime.state.page_table
        ends, offset = [], 0
        for n_tok in lens:
            offset += n_tok
            ends.append(offset - 1)
        # Inside each request's chunk: pair position t with token t + 1.
        keep = torch.ones((real,), dtype=torch.bool, device=dev)
        keep[torch.tensor(ends, dtype=torch.int64, device=dev)] = False
        nxt = torch.empty_like(ids)
        nxt[:-1] = ids[1:]
        nxt[-1] = 0
        idx = keep.nonzero().squeeze(1)
        self.draft.write_context(features[:real][idx], nxt[idx], positions[idx], slots[idx])
        # The previous chunk's last position, now that this chunk's first token is known.
        pend_rows, pend_pos, first_tok = [], [], []
        offset = 0
        for req, n_tok in zip(reqs, lens):
            row = int(req.table_idx)
            if n_tok and self._pending_pos.get(row) == int(req.cached_len) - 1:
                pend_rows.append(row)
                pend_pos.append(int(req.cached_len) - 1)
                first_tok.append(offset)
                del self._pending_pos[row]
            offset += n_tok
        if pend_rows:
            rows_t = torch.tensor(pend_rows, dtype=torch.int64, device=dev)
            pos_t = torch.tensor(pend_pos, dtype=torch.int64, device=dev)
            self.draft.write_context(
                self._pending_feature.index_select(0, rows_t),
                ids.index_select(0, torch.tensor(first_tok, dtype=torch.int64, device=dev)),
                pos_t, page_table[rows_t, pos_t].to(torch.int64),
            )
        # Finished prompts draft after the sampled token; the others hold their last feature.
        wb_table, wb_pos, wb_batch = batch.afd_writeback_host
        done = set(int(b) for b in wb_batch)
        hold_rows, hold_tok = [], []
        for i, req in enumerate(reqs):
            if lens[i] and i not in done:
                hold_rows.append(int(req.table_idx))
                hold_tok.append(ends[i])
                self._pending_pos[int(req.table_idx)] = int(req.device_len) - 1
        if hold_rows:
            self._pending_feature[torch.tensor(hold_rows, dtype=torch.int64, device=dev)] = features[
                torch.tensor(hold_tok, dtype=torch.int64, device=dev)
            ].to(self._pending_feature.dtype)
        if not wb_table:
            return
        sel = torch.tensor([ends[int(b)] for b in wb_batch], dtype=torch.int64, device=dev)
        bonus = next_tokens.index_select(0, torch.tensor(wb_batch, dtype=torch.int64, device=dev)).to(torch.int64)
        last = features[:real].index_select(0, sel)
        self.draft.write_context(last, bonus, positions.index_select(0, sel), slots.index_select(0, sel))
        table_idx = torch.tensor(wb_table, dtype=torch.int64, device=dev)
        start = torch.tensor([int(p) for p in wb_pos], dtype=torch.int64, device=dev)
        for row in wb_table:
            self._pending_pos.pop(int(row), None)
        self._propose_eagle3(last, bonus, start, table_idx)

    def _propose_eagle3(
        self, last: torch.Tensor, bonus: torch.Tensor, start: torch.Tensor, table_idx: torch.Tensor
    ) -> None:
        """Draft block - 1 tokens after ``bonus`` and write [bonus, drafts] to token_pool[start:]."""
        block = self.spec.block
        runtime = self.runtime
        page_rows = runtime.state.page_table.index_select(0, table_idx)
        drafts = self.draft.propose(last, bonus, start, page_rows, block - 1, max_kv_len=self.max_kv_len)
        cols = start[:, None] + torch.arange(block, device=self.device)[None, :]
        values = torch.cat((bonus[:, None], drafts), dim=1).to(runtime.token_pool.dtype)
        runtime.token_pool[table_idx[:, None].expand(-1, block), cols] = values

    # ----- shared -----

    def _greedy(self, hidden: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """Target argmax over the vocabulary, chunked to bound the logits buffer."""
        head = self.model.lm_head
        weight = (head.tied_embedding or head).weight
        out = torch.empty((hidden.shape[0],), dtype=torch.int64, device=hidden.device)
        for lo in range(0, hidden.shape[0], chunk):
            hi = min(lo + chunk, hidden.shape[0])
            out[lo:hi] = torch.nn.functional.linear(hidden[lo:hi], weight).argmax(dim=-1)
        return out

__all__ = ["AfdSpecAG", "draft_kv_bytes_per_token", "draft_ring_bytes"]
