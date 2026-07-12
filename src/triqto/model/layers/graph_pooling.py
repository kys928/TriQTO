"""Permutation-invariant graph pooling for variable-size circuit batches."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.tensor_ops import segment_max, segment_mean


class GraphPooling(nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float = 0.0, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )

    def forward(self, node_embeddings: Tensor, node_batch: Tensor, graph_count: int) -> Tensor:
        mean = segment_mean(node_embeddings, node_batch, graph_count)
        maximum = segment_max(node_embeddings, node_batch, graph_count)
        return self.output(torch.cat((mean, maximum), dim=1))
