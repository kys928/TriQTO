"""Complex-inspired phase-coupled message passing without Q/K/V attention."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.tensor_ops import segment_sum


class PhaseCoupledMessagePassing(nn.Module):
    """Aggregate directed lattice messages through learned cosine/sine quadratures.

    This is a classical graph layer inspired by phasor algebra. It does not execute a
    quantum circuit and it does not imitate transformer query/key/value attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim must be positive and even")
        context_dim = hidden_dim * 3
        self.in_phase = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.quadrature = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phase = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor,
        edge_embeddings: Tensor,
        edge_gate_embeddings: Tensor,
    ) -> Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_count = edge_index.shape[1]
        if edge_embeddings.shape[0] != edge_count or edge_gate_embeddings.shape[0] != edge_count:
            raise ValueError("edge embedding row counts must equal edge count")
        if edge_count == 0:
            return node_embeddings
        source = edge_index[0]
        destination = edge_index[1]
        source_embedding = node_embeddings.index_select(0, source)
        context = torch.cat((source_embedding, edge_embeddings, edge_gate_embeddings), dim=1)
        phase_context = torch.cat((edge_embeddings, edge_gate_embeddings), dim=1)
        angle = torch.pi * torch.tanh(self.phase(phase_context))
        message = self.in_phase(context) * torch.cos(angle) + self.quadrature(context) * torch.sin(angle)
        aggregate = segment_sum(message, destination, node_embeddings.shape[0])
        degree = segment_sum(
            torch.ones(edge_count, dtype=node_embeddings.dtype, device=node_embeddings.device),
            destination,
            node_embeddings.shape[0],
        ).clamp_min(1.0)
        aggregate = aggregate / degree.unsqueeze(1)
        update = self.update(torch.cat((node_embeddings, aggregate), dim=1))
        return self.norm(node_embeddings + update)


__all__ = ["PhaseCoupledMessagePassing"]
