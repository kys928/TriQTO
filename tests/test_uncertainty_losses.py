from __future__ import annotations

import torch

from triqto.evaluation.metrics import evaluate_uncertainty_against_error
from triqto.model.losses.uncertainty_losses import per_example_gaussian_nll, reduce_masked_per_example_loss, uncertainty_error_correlation


def test_per_example_reduction_order_uses_individual_losses() -> None:
    errors = torch.tensor([1.0, 3.0], requires_grad=True)
    logvar = torch.tensor([0.0, 2.0], requires_grad=True)
    mask = torch.tensor([True, True])
    per = per_example_gaussian_nll(errors, logvar, mask)
    expected = 0.5 * torch.stack([errors[0] ** 2 + logvar[0], torch.exp(-logvar[1]) * errors[1] ** 2 + logvar[1]])
    assert torch.allclose(per, expected)
    loss = reduce_masked_per_example_loss(per, mask)
    loss.backward()
    assert logvar.grad is not None and torch.isfinite(logvar.grad).all()


def test_uncertainty_masks_empty_subsets_stably() -> None:
    values = torch.tensor([1.0, 2.0])
    mask = torch.tensor([False, False])
    per = per_example_gaussian_nll(values, values, mask)
    assert torch.equal(per, torch.zeros_like(values))
    assert reduce_masked_per_example_loss(per, mask).item() == 0.0


def test_uncertainty_error_evaluation_separates_calibration_claim() -> None:
    result = evaluate_uncertainty_against_error([0.1, 2.0, 3.0], [0.2, 1.5, 2.5], [True, True, True], trained_calibration_enabled=False)
    assert result.active_count == 3
    assert result.uncertainty_error_correlation > 0
    assert result.calibration_claim_enabled is False


def test_uncertainty_error_correlation_masking() -> None:
    corr = uncertainty_error_correlation(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.0, 2.0, 4.0]), torch.tensor([True, False, True]))
    assert corr > 0.99
