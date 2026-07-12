"""Reusable topology-feature encoder; topology remains audit-weighted at zero."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.contracts import DenseFeatureBatch
from triqto.model.tensor_ops import masked_zero


class TopologyEncoder(nn.Module):
    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.network = nn.Sequential(
            nn.Linear(config.topology_input_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )

    def forward(self, batch: DenseFeatureBatch | None, graph_count: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        if batch is None:
            return reference.new_zeros((graph_count, reference.shape[-1])), torch.zeros(graph_count, dtype=torch.bool, device=reference.device)
        return masked_zero(self.network(batch.features), batch.available_mask), batch.available_mask


__all__ = ["TopologyEncoder"]
