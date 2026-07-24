"""Distortion diagnosis outputs at graph and qubit levels."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from triqto.model.config import TriQTOModelConfig
from triqto.model.outputs import DistortionHeadOutput

_STRENGTH_SCALE_FLOOR = 0.05
_STRENGTH_LOG_SCALE_MAX = 3.0


class DistortionHead(nn.Module):
    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.graph_trunk = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.classifier = nn.Linear(hidden, len(config.distortion_labels))
        self.strength = nn.Linear(hidden, 2)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        graph_latent: Tensor,
        node_embeddings: Tensor,
        node_batch: Tensor,
        graph_available_mask: Tensor,
    ) -> DistortionHeadOutput:
        hidden = self.graph_trunk(graph_latent)
        strength = self.strength(hidden)
        # The second channel is an unconstrained raw scale parameter. Converting it
        # through softplus avoids the e^24 precision permitted by the old -12 clamp.
        # The floor is expressed in the normalized strength-target units and keeps
        # the Gaussian NLL finite without removing heteroscedastic uncertainty.
        strength_scale = F.softplus(strength[:, 1]) + _STRENGTH_SCALE_FLOOR
        strength_log_scale = torch.log(strength_scale).clamp(
            max=_STRENGTH_LOG_SCALE_MAX
        )
        node_context = hidden.index_select(0, node_batch)
        affected = self.node_classifier(
            torch.cat((node_embeddings, node_context), dim=1)
        ).squeeze(1)
        graph_mask = graph_available_mask.to(hidden.dtype)
        node_mask = graph_available_mask.index_select(0, node_batch).to(hidden.dtype)
        return DistortionHeadOutput(
            class_logits=self.classifier(hidden) * graph_mask.unsqueeze(1),
            strength_mean=strength[:, 0] * graph_mask,
            strength_log_scale=strength_log_scale * graph_mask,
            affected_qubit_logits=affected * node_mask,
            graph_available_mask=graph_available_mask,
        )


__all__ = ["DistortionHead"]
