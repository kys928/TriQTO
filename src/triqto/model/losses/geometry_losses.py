"""Geometry-consistency losses defined without activating a trainer."""
from __future__ import annotations

import torch
from torch import Tensor


def pairwise_distance_consistency_loss(
    latent: Tensor,
    target_distance: Tensor,
    pair_mask: Tensor,
) -> Tensor:
    if latent.ndim != 2:
        raise ValueError("latent must have shape [B,D]")
    if target_distance.shape != (latent.shape[0], latent.shape[0]):
        raise ValueError("target_distance must be square with latent batch size")
    if pair_mask.dtype != torch.bool or pair_mask.shape != target_distance.shape:
        raise ValueError("pair_mask must be bool with target_distance shape")
    predicted = torch.cdist(latent, latent, p=2)
    errors = (predicted - target_distance).square()
    active = errors[pair_mask]
    return active.mean() if active.numel() else errors.sum() * 0.0


__all__ = ["pairwise_distance_consistency_loss"]
