from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from chemprop.nn import MABBondMessagePassing
from torch import Tensor, nn

from .data import SiteBatch


@dataclass(frozen=True)
class SiteModelConfig:
    architecture: str = "legacy"
    atom_dim: int = 72
    bond_dim: int = 14
    hidden_dim: int = 300
    depth: int = 3
    dropout: float = 0.1
    ffn_hidden_dim: int = 300
    shell_count: int = 3
    distance_decay: float = 0.1

    def __post_init__(self) -> None:
        if self.architecture != "legacy":
            raise ValueError("the Cube-pKa V4 release supports the legacy readout only")
        if self.shell_count < 1:
            raise ValueError("shell_count must be positive")
        if self.distance_decay < 0:
            raise ValueError("distance_decay must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SiteDMPNN(nn.Module):
    """Checkpoint-compatible Cube-pKa V4 candidate-site D-MPNN."""

    def __init__(self, config: SiteModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MABBondMessagePassing(
            d_v=config.atom_dim,
            d_e=config.bond_dim,
            d_h=config.hidden_dim,
            depth=config.depth,
            dropout=config.dropout,
            return_vertex_embeddings=True,
            return_edge_embeddings=False,
        )
        head_input_dim = config.hidden_dim * 2 + 1
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, config.ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_dim, 1),
        )

    def _local_pool(self, atom_h: Tensor, batch: SiteBatch) -> Tensor:
        weights = torch.exp(
            -self.config.distance_decay * batch.subgraph_distance.to(atom_h.dtype)
        )
        member_h = atom_h[batch.subgraph_atom_index]
        candidate_count = batch.candidate_atom_index.numel()
        sums = atom_h.new_zeros((candidate_count, atom_h.shape[1]))
        sums.index_add_(
            0,
            batch.subgraph_candidate_index,
            member_h * weights.unsqueeze(1),
        )
        denominators = atom_h.new_zeros(candidate_count)
        denominators.index_add_(0, batch.subgraph_candidate_index, weights)
        return sums / denominators.clamp_min(1e-12).unsqueeze(1)

    def forward(self, batch: SiteBatch) -> Tensor:
        atom_h, _ = self.encoder(batch.bmg)
        if atom_h is None:
            raise RuntimeError("Chemprop encoder did not return atom embeddings")
        center_h = atom_h[batch.candidate_atom_index]
        local_h = self._local_pool(atom_h, batch)
        task = batch.candidate_task.to(atom_h.dtype).unsqueeze(1)
        return self.head(torch.cat((center_h, local_h, task), dim=1)).squeeze(1)


def aggregate_site_pka(site_pka: Tensor, task: str) -> Tensor:
    if site_pka.ndim != 1 or site_pka.numel() == 0:
        raise ValueError("site_pka must be a non-empty 1-D tensor")
    if task == "acid":
        return site_pka.min()
    if task == "base":
        return site_pka.max()
    raise ValueError(f"unsupported task: {task}")
