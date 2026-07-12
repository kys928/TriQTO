"""Distortion-diagnosis loss bundle without optimizer logic."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .task_losses import (
    masked_binary_cross_entropy_with_logits,
    masked_cross_entropy,
    masked_mse_loss,
)


@dataclass(slots=True)
class DiagnosisLosses:
    distortion_type: Tensor
    strength: Tensor
    affected_qubits: Tensor

    @property
    def total(self) -> Tensor:
        return self.distortion_type + self.strength + self.affected_qubits


def diagnosis_losses(
    *,
    class_logits: Tensor,
    class_target: Tensor,
    class_mask: Tensor,
    strength_prediction: Tensor,
    strength_target: Tensor,
    strength_mask: Tensor,
    affected_logits: Tensor,
    affected_target: Tensor,
    affected_mask: Tensor,
) -> DiagnosisLosses:
    return DiagnosisLosses(
        distortion_type=masked_cross_entropy(class_logits, class_target, class_mask),
        strength=masked_mse_loss(strength_prediction, strength_target, strength_mask),
        affected_qubits=masked_binary_cross_entropy_with_logits(
            affected_logits,
            affected_target,
            affected_mask,
        ),
    )


__all__ = ["DiagnosisLosses", "diagnosis_losses"]
