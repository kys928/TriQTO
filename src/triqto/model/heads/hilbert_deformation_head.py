"""Observable-to-hidden Hilbert deformation prediction head."""
from __future__ import annotations

from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.outputs import HilbertDeformationHeadOutput


class HilbertDeformationHead(nn.Module):
    """Predict a bounded-dimensional hidden-state deformation summary.

    The head intentionally does not consume the Hilbert stream under the hard Phase 13
    policy, preventing trivial copying of privileged simulator information.
    """

    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        output_dim = config.hilbert_deformation_dim
        self.trunk = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.mean = nn.Linear(hidden, output_dim)
        self.log_scale = nn.Linear(hidden, output_dim)

    def forward(self, graph_latent: Tensor, graph_available_mask: Tensor) -> HilbertDeformationHeadOutput:
        hidden = self.trunk(graph_latent)
        mask = graph_available_mask.to(hidden.dtype).unsqueeze(1)
        return HilbertDeformationHeadOutput(
            mean=self.mean(hidden) * mask,
            log_scale=self.log_scale(hidden).clamp(min=-12.0, max=8.0) * mask,
            graph_available_mask=graph_available_mask,
        )


__all__ = ["HilbertDeformationHead"]
