from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    # Tokens a running request may hold past max_device_len (speculative lookahead).
    extra_reserve: int = 0
    # Keep decode requests keyed by uid so every TP rank can rebuild the same batch order.
    running_reqs: Dict[int, Req] = field(default_factory=dict)
    _ordered_uids: list[int] | None = field(default=None, init=False)
    _ordered_reqs: list[Req] | None = field(default=None, init=False)

    def _invalidate_order_cache(self) -> None:
        self._ordered_uids = None
        self._ordered_reqs = None

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        changed = False
        for req in reqs:
            if req.can_decode:
                old = self.running_reqs.get(req.uid)
                if old is None or old is not req:
                    changed = True
                self.running_reqs[req.uid] = req
            else:
                if self.running_reqs.pop(req.uid, None) is not None:
                    changed = True
        if changed:
            self._invalidate_order_cache()

    def remove_req(self, req: Req) -> None:
        if self.running_reqs.pop(req.uid, None) is not None:
            self._invalidate_order_cache()

    def abort_req(self, uid: int) -> Req | None:
        req = self.running_reqs.pop(uid, None)
        if req is not None:
            self._invalidate_order_cache()
        return req

    @property
    def inflight_tokens(self) -> int:
        # 1 page reserved, plus any speculative lookahead past max_device_len
        tokens_reserved = (self.page_size - 1 + self.extra_reserve) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs.values()) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        if self._ordered_reqs is None:
            self._ordered_uids = sorted(self.running_reqs)
            self._ordered_reqs = [self.running_reqs[uid] for uid in self._ordered_uids]
        # Batch owns the list container; Req objects are shared scheduler state.
        reqs = list(self._ordered_reqs)
        return Batch(reqs=reqs, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
