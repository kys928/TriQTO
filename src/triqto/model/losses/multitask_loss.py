"""Transparent Phase 13 loss composition contract; no training loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True, slots=True)
class MultiTaskLossWeights:
    task: float = 1.0
    geometry: float = 1.0
    diagnosis: float = 1.0
    action: float = 1.0
    topology: float = 0.0

    def __post_init__(self) -> None:
        for name in ("task", "geometry", "diagnosis", "action", "topology"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
                raise ValueError(f"{name} loss weight must be nonnegative numeric")
        if float(self.topology) != 0.0:
            raise ValueError("Phase 13 topology weight must remain exactly 0.0")


def combine_multitask_losses(
    losses: Mapping[str, Tensor],
    weights: MultiTaskLossWeights | None = None,
) -> Tensor:
    required = {"task", "geometry", "diagnosis", "action", "topology"}
    if set(losses) != required:
        raise ValueError(
            f"loss mapping must contain exactly {sorted(required)}"
        )
    selected = weights or MultiTaskLossWeights()
    return (
        losses["task"] * selected.task
        + losses["geometry"] * selected.geometry
        + losses["diagnosis"] * selected.diagnosis
        + losses["action"] * selected.action
        + losses["topology"] * 0.0
    )


__all__ = ["MultiTaskLossWeights", "combine_multitask_losses"]
