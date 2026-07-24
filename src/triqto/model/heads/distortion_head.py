"""Distortion diagnosis outputs at graph and qubit levels."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from triqto.model.config import TriQTOModelConfig
from triqto.model.outputs import DistortionHeadOutput

_STRENGTH_SCALE_FLOOR = 0.05
_STRENGTH_DEFAULT_SCALE = 0.50
_STRENGTH_LOG_SCALE_MAX = 3.0


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse-softplus input must be finite and positive")
    return math.log(math.expm1(value))


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
        self.reset_output_baselines()

    def reset_output_baselines(self) -> None:
        """Start diagnosis outputs at neutral, calibrated baselines.

        The top-level model applies a generic Xavier pass after constructing all
        submodules, then calls this method again. Zero output weights keep the first
        optimization step focused on learning each head's readout before sending a
        large random gradient into shared fusion. Strength starts at mean zero with a
        finite default scale of 0.5 in normalized target units.
        """
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.zeros_(self.strength.weight)
        nn.init.zeros_(self.strength.bias)
        raw_scale = _inverse_softplus(
            _STRENGTH_DEFAULT_SCALE - _STRENGTH_SCALE_FLOOR
        )
        with torch.no_grad():
            self.strength.bias[1] = raw_scale
        final_node = self.node_classifier[-1]
        if not isinstance(final_node, nn.Linear):
            raise TypeError("diagnosis node classifier must end with a Linear layer")
        nn.init.zeros_(final_node.weight)
        nn.init.zeros_(final_node.bias)

    def forward(
        self,
        graph_latent: Tensor,
        node_embeddings: Tensor,
        node_batch: Tensor,
        graph_available_mask: Tensor,
    ) -> DistortionHeadOutput:
        hidden = self.graph_trunk(graph_latent)
        strength = self.strength(hidden)
        # The second channel is an unconstrained raw scale parameter. Softplus keeps
        # the reported scale positive; the robust Student-t training objective bounds
        # the influence of residuals even if the learned scale approaches this floor.
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
