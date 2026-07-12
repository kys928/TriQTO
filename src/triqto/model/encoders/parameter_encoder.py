"""Ragged parameter-manifold encoder."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.contracts import ParameterTensorBatch
from triqto.model.tensor_ops import masked_zero, segment_max, segment_mean


class ParameterEncoder(nn.Module):
    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.element = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.pool = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )

    def forward(self, batch: ParameterTensorBatch | None, graph_count: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        if batch is None:
            return reference.new_zeros((graph_count, reference.shape[-1])), torch.zeros(graph_count, dtype=torch.bool, device=reference.device)
        if batch.values.numel() == 0:
            return reference.new_zeros((graph_count, reference.shape[-1])), batch.available_mask
        elements = self.element(torch.stack((batch.values, batch.sin, batch.cos), dim=1))
        pooled = self.pool(
            torch.cat(
                (
                    segment_mean(elements, batch.batch_index, graph_count),
                    segment_max(elements, batch.batch_index, graph_count),
                ),
                dim=1,
            )
        )
        return masked_zero(pooled, batch.available_mask), batch.available_mask


__all__ = ["ParameterEncoder"]
