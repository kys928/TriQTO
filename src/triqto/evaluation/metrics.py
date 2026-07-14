"""Evaluation metric helpers with explicit claim boundaries."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class UncertaintyEvaluation:
    active_count: int
    error_gap_high_minus_low: float
    uncertainty_error_correlation: float
    calibration_claim_enabled: bool


def evaluate_uncertainty_against_error(predicted_uncertainty: Sequence[float], observed_error: Sequence[float], mask: Sequence[bool], *, trained_calibration_enabled: bool) -> UncertaintyEvaluation:
    u = np.asarray(predicted_uncertainty, dtype=float)
    e = np.asarray(observed_error, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if u.shape != e.shape or e.shape != m.shape:
        raise ValueError("uncertainty, error, and mask shapes must match")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(e)):
        raise ValueError("uncertainty and error values must be finite")
    if int(m.sum()) == 0:
        return UncertaintyEvaluation(0, 0.0, 0.0, False)
    active_u = u[m]
    active_e = e[m]
    threshold = float(np.median(active_u))
    low = active_e[active_u <= threshold]
    high = active_e[active_u > threshold]
    gap = float(high.mean() - low.mean()) if len(low) and len(high) else 0.0
    corr = float(np.corrcoef(active_u, active_e)[0, 1]) if len(active_u) > 1 and np.std(active_u) > 0 and np.std(active_e) > 0 else 0.0
    return UncertaintyEvaluation(int(m.sum()), gap, corr, bool(trained_calibration_enabled))


def describe_contract() -> str:
    return "Evaluation metrics include direct uncertainty/error diagnostics; calibration claims require trained_calibration_enabled."


__all__ = ["UncertaintyEvaluation", "evaluate_uncertainty_against_error"]
