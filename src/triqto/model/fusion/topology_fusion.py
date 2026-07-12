"""Optional topology-feature residual fusion."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class TopologyFusion(nn.Module):
    """Add topology only where it is available and permitted by the head mask."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(self, base: Tensor, topology: Tensor, topology_available: Tensor) -> Tensor:
        if base.shape != topology.shape:
            raise ValueError("base and topology embeddings must have equal shape")
        if topology_available.dtype != torch.bool or topology_available.shape != (base.shape[0],):
            raise ValueError("topology_available must be bool with shape [B]")
        combined = torch.cat((base, topology), dim=1)
        update = self.gate(combined) * self.update(combined)
        update = update * topology_available.to(dtype=base.dtype).unsqueeze(1)
        return self.norm(base + update)


__all__ = ["TopologyFusion"]
