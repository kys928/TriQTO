"""Mask-aware primitive losses used later by the Phase 14 trainer."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("mask must have bool dtype")
    if values.shape[: mask.ndim] != mask.shape:
        raise ValueError("mask shape must match the leading value dimensions")
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    weights = expanded.to(values.dtype)
    denominator = weights.expand_as(values).sum()
    if int(denominator.detach().cpu()) == 0:
        return values.sum() * 0.0
    return (values * weights).sum() / denominator


def masked_mse_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shape")
    return masked_mean((prediction - target).square(), mask)


def masked_binary_cross_entropy_with_logits(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have equal shape")
    return masked_mean(
        F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none"),
        mask,
    )


def masked_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    if logits.ndim != 2 or target.shape != (logits.shape[0],) or mask.shape != target.shape:
        raise ValueError("masked_cross_entropy expects logits [B,C], target [B], mask [B]")
    losses = F.cross_entropy(logits, target, reduction="none")
    return masked_mean(losses, mask)


def distribution_kl_loss(
    predicted_probabilities: Tensor,
    target_probabilities: Tensor,
    row_mask: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    if predicted_probabilities.shape != target_probabilities.shape or row_mask.shape != predicted_probabilities.shape:
        raise ValueError("distribution tensors and row_mask must have equal shape")
    predicted = predicted_probabilities.clamp_min(epsilon)
    target = target_probabilities.clamp_min(0.0)
    terms = torch.where(target > 0, target * (torch.log(target.clamp_min(epsilon)) - torch.log(predicted)), torch.zeros_like(target))
    return masked_mean(terms, row_mask)


__all__ = [
    "distribution_kl_loss",
    "masked_binary_cross_entropy_with_logits",
    "masked_cross_entropy",
    "masked_mean",
    "masked_mse_loss",
]
