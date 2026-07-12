"""Task-level heteroscedastic uncertainty head."""
from __future__ import annotations

from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.outputs import UncertaintyHeadOutput


class UncertaintyHead(nn.Module):
    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, len(config.uncertainty_targets)),
        )

    def forward(self, graph_latent: Tensor, graph_available_mask: Tensor) -> UncertaintyHeadOutput:
        mask = graph_available_mask.to(graph_latent.dtype).unsqueeze(1)
        return UncertaintyHeadOutput(
            log_variance=self.network(graph_latent).clamp(min=-12.0, max=12.0) * mask,
            graph_available_mask=graph_available_mask,
        )


__all__ = ["UncertaintyHead"]
