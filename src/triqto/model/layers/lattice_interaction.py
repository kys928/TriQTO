"""Stacked graph/lattice interaction used by the circuit encoder."""
from __future__ import annotations

from torch import Tensor, nn

from .phase_coupled_message_passing import PhaseCoupledMessagePassing


class LatticeInteractionStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        *,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be positive")
        self.layers = nn.ModuleList(
            PhaseCoupledMessagePassing(
                hidden_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(layers)
        )

    def forward(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor,
        edge_embeddings: Tensor,
        edge_gate_embeddings: Tensor,
    ) -> Tensor:
        hidden = node_embeddings
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_embeddings, edge_gate_embeddings)
        return hidden


__all__ = ["LatticeInteractionStack"]
