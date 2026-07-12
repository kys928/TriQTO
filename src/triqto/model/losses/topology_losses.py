"""Topology loss contract kept inactive in Phase 13."""
from __future__ import annotations

from torch import Tensor

from .task_losses import masked_mse_loss


def topology_feature_discrepancy(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    return masked_mse_loss(prediction, target, mask)


def apply_phase13_topology_weight(raw_loss: Tensor, weight: float = 0.0) -> Tensor:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError("topology weight must be numeric and not bool")
    if float(weight) != 0.0:
        raise ValueError("Phase 13 topology loss weight must remain exactly 0.0")
    return raw_loss * 0.0


__all__ = ["apply_phase13_topology_weight", "topology_feature_discrepancy"]
