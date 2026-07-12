"""Non-activated topology audit prediction head."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.outputs import TopologyHeadOutput


class TopologyHead(nn.Module):
    """Predict topology summaries from non-topology streams for future audits.

    Phase 13 exposes the architecture but marks every supervised topology target as
    unavailable. The hard stream policy also forbids direct topology-input copying.
    The default and enforced topology loss weight remains zero.
    """

    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.features = nn.Linear(hidden, config.topology_prediction_dim)
        self.confidence = nn.Linear(hidden, 1)

    def forward(self, graph_latent: Tensor, graph_available_mask: Tensor) -> TopologyHeadOutput:
        hidden = self.trunk(graph_latent)
        mask = graph_available_mask.to(hidden.dtype)
        return TopologyHeadOutput(
            feature_prediction=self.features(hidden) * mask.unsqueeze(1),
            confidence_logit=self.confidence(hidden).squeeze(1) * mask,
            graph_available_mask=graph_available_mask,
            supervised_target_available_mask=torch.zeros_like(graph_available_mask),
        )


__all__ = ["TopologyHead"]
