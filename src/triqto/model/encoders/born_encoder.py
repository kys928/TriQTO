"""Observable Born-distribution encoder over variable-width basis strings."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.contracts import BornTensorBatch
from triqto.model.tensor_ops import masked_zero, segment_max, segment_mean, segment_sum


class BasisBitEncoder(nn.Module):
    """Shared bit-position encoder that does not assume a fixed qubit count."""

    def __init__(self, hidden_dim: int, *, dropout: float = 0.0, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bit = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(self, bits: Tensor, bit_mask: Tensor) -> Tensor:
        if bits.ndim != 2 or bit_mask.shape != bits.shape:
            raise ValueError("bits and bit_mask must have equal [row,qubit] shape")
        rows, width = bits.shape
        if width == 0:
            return bits.new_zeros((rows, self.hidden_dim))
        positions = torch.arange(width, device=bits.device, dtype=bits.dtype).unsqueeze(0)
        active_count = bit_mask.sum(dim=1, keepdim=True).to(bits.dtype)
        denominator = (active_count - 1.0).clamp_min(1.0)
        position_grid = positions.expand(rows, width) / denominator
        encoded = self.bit(torch.stack((bits, position_grid), dim=2))
        active = bit_mask.to(dtype=encoded.dtype).unsqueeze(2)
        count = active.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * active).sum(dim=1) / count
        return self.output(pooled)


class MeasurementConditionedBasisEncoder(nn.Module):
    """Encode outcome bits jointly with the declared per-qubit Pauli basis."""

    def __init__(self, hidden_dim: int, *, dropout: float = 0.0, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bit = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(
        self,
        bits: Tensor,
        bit_mask: Tensor,
        measurement_basis_codes: Tensor,
    ) -> Tensor:
        if bits.ndim != 2 or bit_mask.shape != bits.shape:
            raise ValueError("bits and bit_mask must have equal [row,qubit] shape")
        if measurement_basis_codes.shape != bits.shape:
            raise ValueError("measurement_basis_codes must match bits shape")
        rows, width = bits.shape
        if width == 0:
            return bits.new_zeros((rows, self.hidden_dim))
        positions = torch.arange(width, device=bits.device, dtype=bits.dtype).unsqueeze(0)
        active_count = bit_mask.sum(dim=1, keepdim=True).to(bits.dtype)
        denominator = (active_count - 1.0).clamp_min(1.0)
        position_grid = positions.expand(rows, width) / denominator
        one_hot = F.one_hot(measurement_basis_codes, num_classes=3).to(bits.dtype)
        encoded = self.bit(
            torch.cat(
                (bits.unsqueeze(2), position_grid.unsqueeze(2), one_hot),
                dim=2,
            )
        )
        active = bit_mask.to(dtype=encoded.dtype).unsqueeze(2)
        count = active.sum(dim=1).clamp_min(1.0)
        return self.output((encoded * active).sum(dim=1) / count)


class BornEncoder(nn.Module):
    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.basis = MeasurementConditionedBasisEncoder(
            hidden,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.outcome = nn.Sequential(
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

    def forward(self, batch: BornTensorBatch | None, graph_count: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        if batch is None:
            return reference.new_zeros((graph_count, reference.shape[-1])), torch.zeros(graph_count, dtype=torch.bool, device=reference.device)
        if batch.probabilities.numel() == 0:
            return reference.new_zeros((graph_count, reference.shape[-1])), batch.available_mask
        basis = self.basis(
            batch.outcome_bits,
            batch.outcome_bit_mask,
            batch.measurement_basis_codes,
        )
        probability = batch.probabilities.clamp_min(torch.finfo(batch.probabilities.dtype).tiny)
        entropy_term = -probability * torch.log(probability)
        active_count = batch.outcome_bit_mask.sum(dim=1).clamp_min(1).to(batch.probabilities.dtype)
        hamming_fraction = (batch.outcome_bits * batch.outcome_bit_mask).sum(dim=1) / active_count
        parity = torch.remainder((batch.outcome_bits * batch.outcome_bit_mask).sum(dim=1), 2.0)
        outcome = self.outcome(
            torch.cat(
                (
                    basis,
                    probability.unsqueeze(1),
                    probability.sqrt().unsqueeze(1),
                    entropy_term.unsqueeze(1),
                    hamming_fraction.unsqueeze(1),
                    parity.unsqueeze(1),
                ),
                dim=1,
            )
        )
        weighted = segment_sum(outcome * batch.probabilities.unsqueeze(1), batch.batch_index, graph_count)
        setting_count = torch.ones(
            graph_count,
            dtype=weighted.dtype,
            device=weighted.device,
        )
        for graph_index in range(graph_count):
            rows = batch.batch_index == graph_index
            if bool(rows.any()):
                setting_count[graph_index] = batch.measurement_setting_index[rows].unique().numel()
        weighted = weighted / setting_count.unsqueeze(1)
        mean = segment_mean(outcome, batch.batch_index, graph_count)
        maximum = segment_max(outcome, batch.batch_index, graph_count)
        pooled = self.pool(torch.cat((weighted, mean, maximum), dim=1))
        return masked_zero(pooled, batch.available_mask), batch.available_mask


__all__ = ["BasisBitEncoder", "BornEncoder", "MeasurementConditionedBasisEncoder"]
