"""Mask-aware fusion of graph, parameter, phasor, Hilbert, Born, and backend streams."""
from __future__ import annotations

from torch import Tensor, nn

from triqto.model.layers import MaskAwareFusion


class TriManifoldFusion(nn.Module):
    """Fuse non-topology streams; unavailable inputs contribute exactly zero."""

    def __init__(
        self,
        hidden_dim: int,
        stream_count: int,
        *,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.fusion = MaskAwareFusion(
            stream_count,
            hidden_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, streams: Tensor, available_mask: Tensor) -> tuple[Tensor, Tensor]:
        return self.fusion(streams, available_mask)


__all__ = ["TriManifoldFusion"]
