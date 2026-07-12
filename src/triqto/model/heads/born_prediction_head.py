"""Variable-support Born-distribution prediction head."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.contracts import OutcomeQueryTensorBatch
from triqto.model.encoders.born_encoder import BasisBitEncoder
from triqto.model.outputs import BornPredictionHeadOutput
from triqto.model.tensor_ops import segment_softmax


class BornPredictionHead(nn.Module):
    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.basis = BasisBitEncoder(
            hidden,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.query = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        graph_latent: Tensor,
        queries: OutcomeQueryTensorBatch | None,
        graph_active_mask: Tensor,
    ) -> BornPredictionHeadOutput:
        graph_count = graph_latent.shape[0]
        device = graph_latent.device
        if graph_active_mask.dtype != torch.bool or graph_active_mask.shape != (graph_count,):
            raise ValueError("graph_active_mask must be bool with shape [graph_count]")
        if queries is None or queries.outcome_bits.shape[0] == 0:
            empty_float = graph_latent.new_zeros((0,))
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            return BornPredictionHeadOutput(
                outcome_logits=empty_float,
                probabilities=empty_float,
                outcome_batch=empty_long,
                graph_available_mask=torch.zeros(graph_count, dtype=torch.bool, device=device),
            )
        basis = self.basis(queries.outcome_bits, queries.outcome_bit_mask)
        context = graph_latent.index_select(0, queries.batch_index)
        logits = self.query(torch.cat((basis, context), dim=1)).squeeze(1)
        row_mask = graph_active_mask.index_select(0, queries.batch_index)
        probabilities = segment_softmax(
            logits,
            queries.batch_index,
            graph_count,
            row_mask,
        )
        logits = logits * row_mask.to(logits.dtype)
        graph_available = queries.available_mask & graph_active_mask
        return BornPredictionHeadOutput(
            outcome_logits=logits,
            probabilities=probabilities,
            outcome_batch=queries.batch_index,
            graph_available_mask=graph_available,
        )


__all__ = ["BornPredictionHead"]
