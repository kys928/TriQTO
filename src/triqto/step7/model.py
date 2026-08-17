"""First serious Step 7 TriQTO diagnostic model and frozen neural ablations."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.encoders import CircuitGraphEncoder
from triqto.model.tensor_ops import segment_mean, segment_sum

from .contracts import Step7ModelBatch


PRIMARY_VARIANTS = {
    "diagnostic_only",
    "graph_only",
    "late_concat",
    "structured_interaction",
}
ABLATION_VARIANTS = {
    "structured_no_magnitude",
    "structured_no_pairwise",
    "structured_no_parity",
}
ALL_VARIANTS = PRIMARY_VARIANTS | ABLATION_VARIANTS


def _projection(input_dim: int, hidden: int, dropout: float, eps: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden, eps=eps),
    )


def _head(input_dim: int, hidden: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_dim),
    )


@dataclass(slots=True)
class Step7ModelOutput:
    effect_logit: Tensor
    effect_probability: Tensor
    effect_uncertainty: Tensor
    mechanism_logits: Tensor
    representation: Tensor
    magnitude_features: Tensor


class DiagnosticEncoder(nn.Module):
    """Encode signed local/pair/parity B_delta without treating it as Born probability."""

    def __init__(self, hidden: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.local_projection = _projection(6, hidden, dropout, eps)
        self.pair_projection = _projection(6, hidden, dropout, eps)
        self.parity_projection = _projection(6, hidden, dropout, eps)
        self.graph_fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=eps),
        )

    def forward(
        self,
        batch: Step7ModelBatch,
        *,
        use_pairwise: bool = True,
        use_parity: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        diagnostic = batch.diagnostic
        local_input = torch.cat((diagnostic.local_values, diagnostic.local_values.abs()), dim=1)
        local = self.local_projection(local_input)
        if diagnostic.pair_values.shape[0]:
            pair_input = torch.cat((diagnostic.pair_values, diagnostic.pair_values.abs()), dim=1)
            pair = self.pair_projection(pair_input)
        else:
            pair = local.new_zeros((0, local.shape[1]))
        parity_input = torch.cat((diagnostic.global_parity, diagnostic.global_parity.abs()), dim=1)
        parity = self.parity_projection(parity_input)
        if not use_pairwise:
            pair = torch.zeros_like(pair)
        if not use_parity:
            parity = torch.zeros_like(parity)
        count = batch.graph.graph_count
        pooled_local = segment_mean(local, batch.graph.node_batch, count)
        pooled_pair = segment_mean(pair, diagnostic.pair_batch, count) if pair.shape[0] else local.new_zeros((count, local.shape[1]))
        graph = self.graph_fusion(torch.cat((pooled_local, pooled_pair, parity), dim=1))
        graph = graph * diagnostic.available_mask.to(dtype=graph.dtype).unsqueeze(1)
        return local, pair, parity, graph


class Step7DiagnosticModel(nn.Module):
    """Graph-conditioned diagnostic model with explicit effect magnitude pathway."""

    def __init__(
        self,
        *,
        variant: str = "structured_interaction",
        hidden_dim: int = 64,
        graph_message_passing_layers: int = 3,
        residual_mlp_layers: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initialization_seed: int = 1701,
    ) -> None:
        super().__init__()
        if variant not in ALL_VARIANTS:
            raise ValueError(f"unknown Step 7 variant {variant!r}")
        if hidden_dim < 8 or hidden_dim % 2:
            raise ValueError("Step 7 hidden_dim must be even and >= 8")
        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.graph_config = TriQTOModelConfig(
            model_name=f"step7_{variant}",
            hidden_dim=int(hidden_dim),
            graph_message_passing_layers=int(graph_message_passing_layers),
            residual_mlp_layers=int(residual_mlp_layers),
            dropout=float(dropout),
            layer_norm_eps=float(layer_norm_eps),
            initialization_seed=int(initialization_seed),
            use_hilbert=False,
            use_backend=False,
            use_topology=False,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.graph_encoder = CircuitGraphEncoder(self.graph_config)
            self.diagnostic_encoder = DiagnosticEncoder(hidden_dim, dropout, layer_norm_eps)
            self.local_gate = nn.Linear(hidden_dim, hidden_dim)
            self.pair_geometry = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            )
            self.pair_gate = nn.Linear(hidden_dim, hidden_dim)
            self.global_gate = nn.Linear(hidden_dim, hidden_dim)
            self.local_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
            self.pair_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
            self.global_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
            self.structured_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            )
            self.late_concat_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            )
            self.diagnostic_only_fusion = _projection(hidden_dim, hidden_dim, dropout, layer_norm_eps)
            self.graph_only_fusion = _projection(hidden_dim, hidden_dim, dropout, layer_norm_eps)
            self.magnitude_encoder = _projection(8, hidden_dim, dropout, layer_norm_eps)
            self.effect_head_with_magnitude = _head(hidden_dim * 2, hidden_dim, 1, dropout)
            self.effect_head_without_magnitude = _head(hidden_dim, hidden_dim, 1, dropout)
            self.mechanism_head = _head(hidden_dim, hidden_dim, 3, dropout)
            self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def validate_batch(self, batch: Step7ModelBatch) -> None:
        batch.validate(self.graph_config)

    def magnitude_features(self, batch: Step7ModelBatch) -> Tensor:
        diagnostic = batch.diagnostic
        count = batch.graph.graph_count
        local_energy = segment_sum(
            diagnostic.local_values.square().sum(dim=1), batch.graph.node_batch, count
        )
        local_count = segment_sum(
            torch.full_like(batch.graph.node_batch, 3, dtype=diagnostic.local_values.dtype),
            batch.graph.node_batch,
            count,
        )
        if diagnostic.pair_values.shape[0]:
            pair_energy = segment_sum(
                diagnostic.pair_values.square().sum(dim=1), diagnostic.pair_batch, count
            )
            pair_count = segment_sum(
                torch.full_like(diagnostic.pair_batch, 3, dtype=diagnostic.pair_values.dtype),
                diagnostic.pair_batch,
                count,
            )
        else:
            pair_energy = local_energy.new_zeros(count)
            pair_count = local_energy.new_zeros(count)
        parity_energy = diagnostic.global_parity.square().sum(dim=1)
        local_rms = torch.sqrt(local_energy / local_count.clamp_min(1.0))
        pair_rms = torch.sqrt(pair_energy / pair_count.clamp_min(1.0))
        parity_rms = torch.sqrt(parity_energy / 3.0)
        total_energy = local_energy + pair_energy + parity_energy
        total_count = local_count + pair_count + 3.0
        total_rms = torch.sqrt(total_energy / total_count.clamp_min(1.0))
        observed = diagnostic.observed_shots.to(dtype=total_rms.dtype).mean(dim=1).clamp_min(1.0)
        reference = diagnostic.reference_shots.to(dtype=total_rms.dtype).mean(dim=1).clamp_min(1.0)
        sqrt_observed = torch.sqrt(observed)
        return torch.stack(
            (
                local_rms,
                pair_rms,
                parity_rms,
                total_rms,
                torch.log2(observed),
                torch.rsqrt(observed),
                total_rms * sqrt_observed,
                torch.log2(reference),
            ),
            dim=1,
        )

    def _structured_representation(
        self,
        batch: Step7ModelBatch,
        graph_output: object,
        local: Tensor,
        pair: Tensor,
        parity: Tensor,
        diagnostic_graph: Tensor,
    ) -> Tensor:
        node = graph_output.node_embeddings
        graph_embedding = graph_output.graph_embedding
        local_interaction = self.local_norm(
            node + local + node * torch.sigmoid(self.local_gate(local))
        )
        pooled_local = segment_mean(local_interaction, batch.graph.node_batch, batch.graph.graph_count)
        if pair.shape[0]:
            left = node.index_select(0, batch.diagnostic.pair_index[0])
            right = node.index_select(0, batch.diagnostic.pair_index[1])
            geometry = self.pair_geometry(
                torch.cat((left, right, (left - right).abs(), left * right), dim=1)
            )
            pair_interaction = self.pair_norm(
                geometry + pair + geometry * torch.sigmoid(self.pair_gate(pair))
            )
            pooled_pair = segment_mean(
                pair_interaction, batch.diagnostic.pair_batch, batch.graph.graph_count
            )
        else:
            pooled_pair = graph_embedding.new_zeros(graph_embedding.shape)
        global_interaction = self.global_norm(
            graph_embedding
            + parity
            + graph_embedding * torch.sigmoid(self.global_gate(parity))
        )
        return self.structured_fusion(
            torch.cat(
                (graph_embedding, pooled_local, pooled_pair, global_interaction, diagnostic_graph),
                dim=1,
            )
        )

    def forward(self, batch: Step7ModelBatch) -> Step7ModelOutput:
        self.validate_batch(batch)
        use_pairwise = self.variant != "structured_no_pairwise"
        use_parity = self.variant != "structured_no_parity"
        local, pair, parity, diagnostic_graph = self.diagnostic_encoder(
            batch, use_pairwise=use_pairwise, use_parity=use_parity
        )
        graph_output = self.graph_encoder(batch.graph)
        if self.variant == "diagnostic_only":
            representation = self.diagnostic_only_fusion(diagnostic_graph)
        elif self.variant == "graph_only":
            representation = self.graph_only_fusion(graph_output.graph_embedding)
        elif self.variant == "late_concat":
            representation = self.late_concat_fusion(
                torch.cat((graph_output.graph_embedding, diagnostic_graph), dim=1)
            )
        else:
            representation = self._structured_representation(
                batch, graph_output, local, pair, parity, diagnostic_graph
            )
        magnitude = self.magnitude_features(batch)
        use_magnitude = self.variant not in {"graph_only", "structured_no_magnitude"}
        if use_magnitude:
            magnitude_embedding = self.magnitude_encoder(magnitude)
            effect_logit = self.effect_head_with_magnitude(
                torch.cat((representation, magnitude_embedding), dim=1)
            ).squeeze(1)
        else:
            effect_logit = self.effect_head_without_magnitude(representation).squeeze(1)
        mechanism_logits = self.mechanism_head(representation)
        probability = torch.sigmoid(effect_logit)
        eps = torch.finfo(probability.dtype).eps
        clipped = probability.clamp(min=eps, max=1.0 - eps)
        uncertainty = -(
            clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped)
        ) / math.log(2.0)
        return Step7ModelOutput(
            effect_logit=effect_logit,
            effect_probability=probability,
            effect_uncertainty=uncertainty,
            mechanism_logits=mechanism_logits,
            representation=representation,
            magnitude_features=magnitude,
        )


__all__ = [
    "ABLATION_VARIANTS",
    "ALL_VARIANTS",
    "DiagnosticEncoder",
    "PRIMARY_VARIANTS",
    "Step7DiagnosticModel",
    "Step7ModelOutput",
]
