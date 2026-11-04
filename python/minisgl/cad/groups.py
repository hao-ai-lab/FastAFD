"""CAD process groups, created COLLECTIVELY by every rank of the AFD union world (attention ranks 0 .. A - 1 first, then the
model ranks A .. A + M - 1, as AfdTopology lays them out), in this fixed order, by the attention workers and
the model workers alike: torch.distributed.new_group must be called by every rank of the default group.

  model group   the M model ranks: the dual-role MegaMoE buffers rendezvous over it (minisgl.cad.m2n_dual)
  window group  the whole union world: every GPU's CadWindow rendezvous over it (minisgl.cad.window)
"""

from __future__ import annotations

from dataclasses import dataclass

from .layout import check_topology, model_of, peer_index, peers_of


@dataclass(frozen=True)
class CadRanks:
    """This rank's CAD role and peers in the union world."""

    rank: int
    num_attn: int
    num_model: int

    def __post_init__(self) -> None:
        check_topology(self.num_attn, self.num_model)
        if not 0 <= self.rank < self.num_attn + self.num_model:
            raise ValueError(f"rank {self.rank} outside the union of {self.num_attn} + {self.num_model}")

    @property
    def role(self) -> str:
        return "attn" if self.rank < self.num_attn else "model"

    @property
    def r(self) -> int:
        return self.num_attn // self.num_model

    @property
    def model_ranks(self) -> list[int]:
        return list(range(self.num_attn, self.num_attn + self.num_model))

    @property
    def local(self) -> int:
        """A model rank's index on the model node(s) (= its MegaMoE EP slice); an attention rank's attention index."""
        return self.rank - self.num_attn if self.role == "model" else self.rank

    @property
    def peers(self) -> list[int]:
        """Union ranks this GPU's window talks to: a model GPU's r attention GPUs (peer order), an attention GPU's model GPU."""
        if self.role == "model":
            return peers_of(self.local, self.num_attn, self.num_model)
        return [self.num_attn + model_of(self.rank, self.num_attn, self.num_model)]

    @property
    def peer_j(self) -> int | None:
        return peer_index(self.rank, self.num_attn, self.num_model) if self.role == "attn" else None


def cad_groups(dist, ranks: CadRanks):
    """(model group, window group), every rank calling in this order (new_group is collective over the default group)."""
    model_group = dist.new_group(ranks=ranks.model_ranks)
    window_group = dist.group.WORLD
    return model_group, window_group


__all__ = ["CadRanks", "cad_groups"]
