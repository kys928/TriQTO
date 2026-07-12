"""Mask-aware stream gating for tri-manifold fusion."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MaskAwareFusion(nn.Module):
    """Fuse available streams through scalar gates, not Q/K/V attention."""

    def __init__(
        self,
        stream_count: int,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if stream_count <= 0 or hidden_dim <= 0:
            raise ValueError("stream_count and hidden_dim must be positive")
        self.stream_count = stream_count
        self.stream_embeddings = nn.Parameter(torch.empty(stream_count, hidden_dim))
        nn.init.normal_(self.stream_embeddings, std=0.02)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mask_projection = nn.Linear(stream_count, hidden_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )

    def forward(self, streams: Tensor, available_mask: Tensor) -> tuple[Tensor, Tensor]:
        if streams.ndim != 3 or streams.shape[1] != self.stream_count:
            raise ValueError(f"streams must have shape [B,{self.stream_count},H]")
        if available_mask.dtype != torch.bool or available_mask.shape != streams.shape[:2]:
            raise ValueError("available_mask must be bool with shape [B,S]")
        if bool((available_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every fused row must expose at least one stream")
        typed = streams + self.stream_embeddings.unsqueeze(0)
        logits = self.gate(typed).squeeze(-1)
        logits = logits.masked_fill(~available_mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(available_mask, weights, torch.zeros_like(weights))
        weighted = (weights.unsqueeze(-1) * streams).sum(dim=1)
        count = available_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (streams * available_mask.unsqueeze(-1)).sum(dim=1) / count
        mask_embedding = self.mask_projection(available_mask.to(dtype=streams.dtype))
        return self.output(torch.cat((weighted, mean, mask_embedding), dim=1)), weights


__all__ = ["MaskAwareFusion"]
