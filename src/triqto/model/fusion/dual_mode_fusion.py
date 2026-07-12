"""Simulation/hardware mode conditioning for mask-aware manifold fusion."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .tri_manifold_fusion import TriManifoldFusion


class DualModeFusion(nn.Module):
    """Fuse streams and encode whether each row is hardware-masked simulation/hardware mode."""

    def __init__(
        self,
        hidden_dim: int,
        stream_count: int,
        *,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.fusion = TriManifoldFusion(
            hidden_dim,
            stream_count,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.mode_embedding = nn.Embedding(2, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )

    def forward(
        self,
        streams: Tensor,
        available_mask: Tensor,
        hardware_mode_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        fused, weights = self.fusion(streams, available_mask)
        mode = self.mode_embedding(hardware_mode_mask.to(dtype=torch.long))
        return self.output(torch.cat((fused, mode), dim=1)), weights


__all__ = ["DualModeFusion"]
