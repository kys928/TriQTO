"""Optional, global-phase-invariant Hilbert-state encoder."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.contracts import HilbertTensorBatch
from triqto.model.tensor_ops import (
    canonicalize_global_phase,
    masked_zero,
    segment_max,
    segment_mean,
    segment_sum,
)

from .born_encoder import BasisBitEncoder


class HilbertEncoder(nn.Module):
    """Encode simulator-only pure states after deterministic global-phase removal."""

    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.basis = BasisBitEncoder(
            hidden,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.amplitude = nn.Sequential(
            nn.Linear(hidden + 5, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.pool = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )

    def forward(self, batch: HilbertTensorBatch | None, graph_count: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        if batch is None:
            return reference.new_zeros((graph_count, reference.shape[-1])), torch.zeros(graph_count, dtype=torch.bool, device=reference.device)
        if batch.amplitudes_real_imag.numel() == 0:
            return reference.new_zeros((graph_count, reference.shape[-1])), batch.available_mask
        canonical = canonicalize_global_phase(
            batch.amplitudes_real_imag,
            batch.batch_index,
            graph_count,
        )
        real = canonical[:, 0]
        imag = canonical[:, 1]
        probability = canonical.square().sum(dim=1)
        magnitude = probability.sqrt()
        phase_cos = real / magnitude.clamp_min(torch.finfo(real.dtype).tiny)
        phase_sin = imag / magnitude.clamp_min(torch.finfo(imag.dtype).tiny)
        basis = self.basis(batch.basis_bits, batch.basis_bit_mask)
        amplitude = self.amplitude(
            torch.cat(
                (
                    basis,
                    real.unsqueeze(1),
                    imag.unsqueeze(1),
                    magnitude.unsqueeze(1),
                    phase_sin.unsqueeze(1),
                    phase_cos.unsqueeze(1),
                ),
                dim=1,
            )
        )
        weighted = segment_sum(amplitude * probability.unsqueeze(1), batch.batch_index, graph_count)
        mean = segment_mean(amplitude, batch.batch_index, graph_count)
        maximum = segment_max(amplitude, batch.batch_index, graph_count)
        pooled = self.pool(torch.cat((weighted, mean, maximum), dim=1))
        return masked_zero(pooled, batch.available_mask), batch.available_mask


__all__ = ["HilbertEncoder"]
