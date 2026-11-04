"""CAD model-GPU MoE: the MegaMoE M2N kernels with every model GPU in TWO positions of a node-local union.

The experts are FP8, through the same pipeline as moe/megamoe_m2n_afd.py register_eg_layer_weights (per-32 UE8M0
requant, transform, tensormap descriptors).

Model GPU p of the model node (M GPUs) is AG position p and EG position M + p of a union of 2 M positions; its two
symmetric buffers are rendezvoused over the node's M GPUs and share one 2 M-entry peer table (AG buffers first). One
round per (layer, mb), all on the caller's stream (no resident EG, no other lane):

  AG phase 1   in-kernel per-32 FP8 quant + dispatch-send              (ag_sms CTAs)
  EG           FP8 experts of this GPU's E / M slice, pull + push      (eg_sms CTAs, 0 = all)
  AG phase 2   combine wait + top-k reduce into y

Counter parity (P2P): a round's position among the GPU's FFN rounds in program order, mod 2, NOT
the layer's parity (with 2 microbatches both rounds of a layer would get the same parity back to back). A step has
num_layers x num_mb rounds; an even count keeps the parity alternating across steps (ffn_parity / check_even_rounds).
The EG reads only the parity of layer_base, so layer_base is the layer itself or the one before (>= 0).

The expert tensors are the prefill EG adapter's own (layers_from_adapter: no second copy) when that adapter is FP8 and
its EP slice is this GPU's local index on the model node; else load_fp8_layer builds them from the checkpoint tensors.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

P2P_CTRL_BYTES = 64          # kernel/megamoe_m2n_mega.py: the control words that end the buffer
BARRIER_SIGNAL = slice(20, 24)
DEFAULT_AG_SMS = 24          # the AG grid; the AG_SPLIT lanes of the AFD path are not used here
DEFAULT_PREFETCH = 64 << 20


@dataclass
class DualLayer:
    """One layer's experts of this EP slice, in the M2N EG kernel's form (FP8 weights, per-32 UE8M0 scales)."""

    l1_t: tuple
    l2_t: tuple
    blob: torch.Tensor        # uint8 [1, 4, 128] tensormaps on the device
    ptrs: torch.Tensor        # int64 [1]: the L1 base (L2 prefetch)
    l1_nbytes: int


def ffn_parity(layer: int, mb: int, num_mb: int) -> int:
    """The counter parity of round (layer, mb): its position in the program order (layer-major, mb inner), mod 2."""
    return (int(layer) * int(num_mb) + int(mb)) & 1


def check_even_rounds(num_layers: int, num_mb: int) -> None:
    if (int(num_layers) * int(num_mb)) % 2:
        raise ValueError(f"{num_layers} layers x {num_mb} microbatches = an odd number of M2N rounds per step: the "
                         "counter parity would not alternate across steps")


def layer_base(layer: int, parity: int) -> int:
    """A model layer of counter parity `parity` (the kernel reads only its parity), the layer itself
    or the one before; layer 0 at parity 1 takes layer 1 (the kernel asserts layer_base >= 0)."""
    base = int(layer) - ((int(layer) ^ int(parity)) & 1)
    return base if base >= 0 else base + 2


def make_dual_buffers(mm, symm_mem, dist, *, group, local: int, size: int, max_tokens: int, num_experts: int,
                      num_topk: int, hidden: int, intermediate: int, device="cuda"):
    """The AG-role (position local) and EG-role (size + local) symmetric buffers of this GPU: barrier signal seeded with
    the union's 2 size positions, P2P credit = size EG ranks.
    `group`: the model node's `size` GPUs (both rendezvous run over it)."""
    ext = mm._ext()
    max_tokens = mm.align(int(max_tokens), mm.get_token_alignment())
    nbytes, pool, slicer = ext.get_symm_buffer_size_for_mega_moe_m2n(size, size, num_experts, max_tokens, num_topk,
                                                                      hidden, intermediate)
    bufs, handles = [], []
    for _ in range(2):
        b = symm_mem.empty(int(nbytes), dtype=torch.int8, device=device)
        handles.append(symm_mem.rendezvous(b, group))
        bufs.append(b)
    ptrs = [int(x) for x in handles[0].buffer_ptrs] + [int(x) for x in handles[1].buffer_ptrs]
    out = []
    for role, b in zip(("ag", "eg"), bufs):
        out.append(seed_buffer(b, nbytes=int(nbytes), size=size, slicer=slicer, ptrs=ptrs, num_experts=num_experts,
                               num_topk=num_topk, hidden=hidden, intermediate=intermediate, max_tokens=max_tokens,
                               pool=int(pool), position=(local if role == "ag" else size + local)))
    torch.cuda.synchronize()
    dist.barrier(group=group)
    return out[0], out[1], handles


def seed_buffer(b, *, nbytes: int, size: int, slicer, ptrs, num_experts: int, num_topk: int, hidden: int,
                intermediate: int, max_tokens: int, pool: int, position: int):
    """Zero a role buffer, seed its barrier signal (2 size: a completed phase-0 round of the union) and its P2P credit
    word (size EG credits), and wrap it as the namespace the kernels' wrappers read."""
    b.zero_()
    b[BARRIER_SIGNAL].view(torch.int32).fill_(2 * size)
    ctrl = int(nbytes) - P2P_CTRL_BYTES
    b[ctrl:ctrl + 4].view(torch.int32).fill_(size)
    views = slicer(b)
    return types.SimpleNamespace(buffer=b, buffer_ptrs=list(ptrs), ag_size=size, eg_size=size,
                                 num_experts=num_experts, num_topk=num_topk, hidden=hidden,
                                 intermediate_hidden=intermediate, num_max_tokens_per_rank=max_tokens,
                                 num_max_pool_tokens=pool, x=views[0], x_sf=views[1], topk_weights=views[3],
                                 position=int(position))


class M2NDual:
    """The model GPU's MoE round: run(layer, mb, h, ids, wts, y, n) = AG phase 1 -> EG -> AG phase 2 on the current
    stream. `mm` is kernel.megamoe_m2n_mega (a fake in the CPU tests); `expected_tokens` maps n (rows of this GPU in
    the round) to the EG's expected_num_tokens_per_rank (the adapter's heuristic; identity by default)."""

    def __init__(self, *, ag_buf, eg_buf, layers: Mapping[int, DualLayer], local: int, size: int, num_mb: int,
                 experts_per_rank: int, eg_sms: int = 0, ag_sms: int = DEFAULT_AG_SMS, prefetch: int = DEFAULT_PREFETCH,
                 expected_tokens: Callable[[int], int] | None = None, mm=None):
        if mm is None:
            from minisgl.kernel import megamoe_m2n_mega as mm
        if not 0 <= int(local) < int(size):
            raise ValueError(f"local {local} outside 0..{size - 1}")
        if int(ag_sms) < 1 or int(eg_sms) < 0 or int(ag_sms) + int(eg_sms) > 1024:
            raise ValueError(f"bad grids ag {ag_sms} eg {eg_sms}")
        self.mm, self.ag_buf, self.eg_buf, self.layers = mm, ag_buf, eg_buf, dict(layers)
        self.local, self.size, self.num_mb = int(local), int(size), int(num_mb)
        self.epr, self.eg_sms, self.ag_sms, self.prefetch = int(experts_per_rank), int(eg_sms), int(ag_sms), int(prefetch)
        self.expected_tokens = expected_tokens or (lambda n: int(n))

    def _ag(self, y, h, ids, wts, parity: int, phase: int) -> None:
        self.mm.mega_moe_m2n_ag(y, h, ids, wts, self.ag_buf, rank=self.local, num_sms=self.ag_sms, renormalize=False,
                                phase=phase, p2p=True, layer_parity=parity)

    def _eg(self, layer: int, parity: int, n: int) -> None:
        w = self.layers[int(layer)]
        self.mm.mega_moe_m2n_eg(w.blob, w.ptrs, self.eg_buf, rank=self.size + self.local,
                                expected_num_tokens_per_rank=int(self.expected_tokens(n)), num_experts_per_rank=self.epr,
                                l1_weights_nbytes=int(w.l1_nbytes), num_layers=1, use_fp8_weights=True,
                                num_sms=self.eg_sms, num_prefetch_bytes=self.prefetch,
                                p2p=True, layer_base=layer_base(layer, parity))

    def precompile(self, layers: Sequence[int], n: int) -> None:
        """Build and load the EG instance of every layer off the protocol path (compile_only launches nothing)."""
        for layer in layers:
            w = self.layers[int(layer)]
            self.mm.mega_moe_m2n_eg(w.blob, w.ptrs, self.eg_buf, rank=self.size + self.local,
                                    expected_num_tokens_per_rank=int(self.expected_tokens(n)),
                                    num_experts_per_rank=self.epr, l1_weights_nbytes=int(w.l1_nbytes), num_layers=1,
                                    use_fp8_weights=True, num_sms=self.eg_sms,
                                    num_prefetch_bytes=self.prefetch, p2p=True, layer_base=int(layer) & 1,
                                    compile_only=True)

    def run(self, layer: int, mb: int, h, ids, wts, y, n: int) -> None:
        par = ffn_parity(layer, mb, self.num_mb)
        self._ag(y, h, ids, wts, par, 1)
        self._eg(layer, par, n)
        self._ag(y, h, ids, wts, par, 2)


# ------------------------------------------------------------------ expert tensors

def adapter_problems(adapter, layers: Sequence[int], local: int, *, num_experts: int, hidden: int,
                     intermediate: int, size: int) -> list[str]:
    """Why the prefill EG adapter cannot lend its expert tensors to the dual-role rounds (empty: it can)."""
    if adapter is None:
        return ["no prefill EG adapter"]
    out = []
    if not getattr(adapter, "is_eg", False):
        out.append("the adapter is not an EG rank here")
    if getattr(adapter, "exact_block_scales", False):
        out.append("it keeps exact block scales (residual pointers), which the dual round does not pass")
    if (int(adapter.num_experts), int(adapter.hidden_size), int(adapter.intermediate_size)) != (num_experts, hidden,
                                                                                              intermediate):
        out.append(f"its shapes {adapter.num_experts} / {adapter.hidden_size} / {adapter.intermediate_size} are not "
                   f"{num_experts} / {hidden} / {intermediate}")
    if int(adapter.eg_size) != int(size):
        out.append(f"its EP size {adapter.eg_size} is not the model node's {size}")
    if int(getattr(adapter, "shared_experts_per_rank", 0) or 0):
        out.append("it carries a shared-expert slot")
    if int(adapter.rank) - int(adapter.ag_size) != int(local):
        out.append(f"its EP slice {int(adapter.rank) - int(adapter.ag_size)} is not this GPU's local index {int(local)}")
    missing = [int(x) for x in layers if int(x) not in adapter._eg_global_to_compact]
    if missing:
        out.append(f"layers {missing[:8]}{'...' if len(missing) > 8 else ''} are not registered on it")
    return out


def layers_from_adapter(adapter, layers: Sequence[int], device) -> dict[int, DualLayer]:
    """The layers as views of the adapter's own registered tensors (call adapter_problems first)."""
    out = {}
    for layer in layers:
        c = adapter._eg_global_to_compact[int(layer)]
        l1_t, l2_t = adapter._eg_weights[c]
        out[int(layer)] = DualLayer(l1_t=l1_t, l2_t=l2_t, blob=adapter._eg_desc_blobs[c].unsqueeze(0),
                                    ptrs=torch.tensor([int(adapter._eg_l1_ptrs[c])], dtype=torch.int64, device=device),
                                    l1_nbytes=int(adapter._eg_l1_nbytes))
    return out


def load_fp8_layer(mm, l1: tuple[torch.Tensor, torch.Tensor], l2: tuple[torch.Tensor, torch.Tensor], *,
                   experts_per_rank: int, hidden: int, intermediate: int, device) -> DualLayer:
    """One layer from this slice's FP8 checkpoint tensors (gate|up [E, 2I, H] / down [E, H, I] e4m3 with 128x128 block
    scales), through the adapter's FP8 pipeline: requant per-32 UE8M0, transform, descriptors."""
    q1 = mm.requant_qwen_fp8_weights_per32(l1[0], l1[1], inplace=True)
    q2 = mm.requant_qwen_fp8_weights_per32(l2[0], l2[1], inplace=True)
    l1_t, l2_t = mm.transform_weights_for_mega_moe(q1, q2)
    blob = mm.make_m2n_weight_descs(l1_t, l2_t, num_experts_per_rank=experts_per_rank, hidden=hidden,
                                    intermediate_hidden=intermediate).to(device).unsqueeze(0).contiguous()
    return DualLayer(l1_t=l1_t, l2_t=l2_t, blob=blob,
                     ptrs=torch.tensor([l1_t[0].data_ptr()], dtype=torch.int64, device=device),
                     l1_nbytes=int(l1_t[0].numel()))


__all__ = ["DEFAULT_AG_SMS", "DualLayer", "M2NDual", "adapter_problems", "check_even_rounds", "ffn_parity", "layer_base",
           "layers_from_adapter", "load_fp8_layer", "make_dual_buffers", "seed_buffer"]
