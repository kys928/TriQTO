"""Final narrow Step 7.1 revision around the Step 7 late-concat champion."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from triqto.step7.contracts import Step7ModelBatch
from triqto.step7.model import Step7DiagnosticModel, Step7ModelOutput


STEP71_VARIANTS = {
    "late_concat",
    "late_concat_parity_residual",
    "late_concat_no_parity",
}


class Step71DiagnosticModel(Step7DiagnosticModel):
    """Late-concat champion plus one explicit global-parity residual candidate.

    Every Step 7.1 variant instantiates the same parameter set.  The candidate uses
    the already-existing Step-7 global parity/graph interaction modules and one
    scalar residual gate.  That scalar is initialized to exactly zero, so before
    training the candidate representation is exactly the late-concat champion
    representation.
    """

    def __init__(
        self,
        *,
        variant: str,
        hidden_dim: int = 64,
        graph_message_passing_layers: int = 3,
        residual_mlp_layers: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initialization_seed: int = 1701,
    ) -> None:
        if variant not in STEP71_VARIANTS:
            raise ValueError(f"unknown Step 7.1 variant {variant!r}")
        super().__init__(
            variant="late_concat",
            hidden_dim=hidden_dim,
            graph_message_passing_layers=graph_message_passing_layers,
            residual_mlp_layers=residual_mlp_layers,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            initialization_seed=initialization_seed,
        )
        self.step71_variant = variant
        self.parity_residual_logit = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def _late_concat_representation(
        self,
        batch: Step7ModelBatch,
    ) -> tuple[Tensor, Tensor]:
        use_parity = self.step71_variant != "late_concat_no_parity"
        _, _, parity, diagnostic_graph = self.diagnostic_encoder(
            batch,
            use_pairwise=True,
            use_parity=use_parity,
        )
        graph_output = self.graph_encoder(batch.graph)
        representation = self.late_concat_fusion(
            torch.cat((graph_output.graph_embedding, diagnostic_graph), dim=1)
        )
        if self.step71_variant == "late_concat_parity_residual":
            parity_interaction = self.global_norm(
                graph_output.graph_embedding
                + parity
                + graph_output.graph_embedding * torch.sigmoid(self.global_gate(parity))
            )
            residual_scale = torch.tanh(self.parity_residual_logit)
            representation = representation + residual_scale * parity_interaction
        return representation, graph_output.graph_embedding

    def forward(self, batch: Step7ModelBatch) -> Step7ModelOutput:
        self.validate_batch(batch)
        representation, _ = self._late_concat_representation(batch)
        magnitude = self.magnitude_features(batch)
        magnitude_embedding = self.magnitude_encoder(magnitude)
        effect_logit = self.effect_head_with_magnitude(
            torch.cat((representation, magnitude_embedding), dim=1)
        ).squeeze(1)
        mechanism_logits = self.mechanism_head(representation)
        probability = torch.sigmoid(effect_logit)
        eps = torch.finfo(probability.dtype).eps
        clipped = probability.clamp(min=eps, max=1.0 - eps)
        uncertainty = -(
            clipped * torch.log(clipped)
            + (1.0 - clipped) * torch.log(1.0 - clipped)
        ) / math.log(2.0)
        return Step7ModelOutput(
            effect_logit=effect_logit,
            effect_probability=probability,
            effect_uncertainty=uncertainty,
            mechanism_logits=mechanism_logits,
            representation=representation,
            magnitude_features=magnitude,
        )


__all__ = ["STEP71_VARIANTS", "Step71DiagnosticModel"]
