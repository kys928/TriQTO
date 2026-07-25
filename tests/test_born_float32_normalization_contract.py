from __future__ import annotations

import numpy as np
import pytest
import torch

from triqto.model.contracts import BornTensorBatch


def _basis_bits(count: int, width: int) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.long).unsqueeze(1)
    shifts = torch.arange(width - 1, -1, -1, dtype=torch.long)
    return ((values >> shifts) & 1).to(torch.float32)


def _large_float32_distribution() -> torch.Tensor:
    rng = np.random.default_rng(2)
    values = rng.random(4096)
    values /= values.sum()
    return torch.tensor(values, dtype=torch.float32)


def _batch(probabilities: torch.Tensor) -> BornTensorBatch:
    bits = _basis_bits(probabilities.numel(), 12)
    return BornTensorBatch(
        outcome_bits=bits,
        outcome_bit_mask=torch.ones_like(bits, dtype=torch.bool),
        probabilities=probabilities,
        batch_index=torch.zeros(probabilities.numel(), dtype=torch.long),
        available_mask=torch.tensor([True]),
    )


def test_large_valid_float32_born_distribution_passes_contract() -> None:
    probabilities = _large_float32_distribution()

    sequential = torch.zeros(1, dtype=torch.float32)
    sequential.index_add_(
        0,
        torch.zeros(probabilities.numel(), dtype=torch.long),
        probabilities,
    )

    assert abs(float(probabilities.sum()) - 1.0) < 1.0e-6
    assert abs(float(sequential[0]) - 1.0) > 1.0e-6

    _batch(probabilities).validate(1)


def test_materially_unnormalized_born_distribution_still_fails() -> None:
    probabilities = _large_float32_distribution() * 0.999

    with pytest.raises(
        ValueError,
        match="born probabilities must sum to one",
    ):
        _batch(probabilities).validate(1)
