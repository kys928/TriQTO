"""Per-example uncertainty objectives and diagnostics."""
from __future__ import annotations

import torch
from torch import Tensor


def per_example_gaussian_nll(errors: Tensor, log_variance: Tensor, mask: Tensor) -> Tensor:
    """Return masked per-example heteroscedastic Gaussian NLL before reduction."""
    if errors.shape != log_variance.shape or errors.shape != mask.shape:
        raise ValueError("errors, log_variance, and mask must have equal shape")
    if not errors.dtype.is_floating_point or not log_variance.dtype.is_floating_point:
        raise TypeError("errors and log_variance must be floating tensors")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    safe_logvar = log_variance.clamp(min=-12.0, max=12.0)
    nll = 0.5 * (torch.exp(-safe_logvar) * errors.square() + safe_logvar)
    return torch.where(mask, nll, torch.zeros_like(nll))


def reduce_masked_per_example_loss(per_example: Tensor, mask: Tensor) -> Tensor:
    if per_example.shape != mask.shape:
        raise ValueError("per_example and mask must have equal shape")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    active = mask.sum()
    if int(active) == 0:
        return per_example.sum() * 0.0
    return per_example[mask].mean()


def uncertainty_error_correlation(predicted_log_variance: Tensor, observed_error: Tensor, mask: Tensor) -> Tensor:
    if predicted_log_variance.shape != observed_error.shape or mask.shape != observed_error.shape:
        raise ValueError("all inputs must have equal shape")
    if int(mask.sum()) < 2:
        return predicted_log_variance.sum() * 0.0
    x = predicted_log_variance[mask]
    y = observed_error[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denom.detach().cpu()) == 0.0:
        return denom * 0.0
    return (x * y).sum() / denom


__all__ = ["per_example_gaussian_nll", "reduce_masked_per_example_loss", "uncertainty_error_correlation"]
