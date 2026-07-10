"""Deterministic simulator-validation reward for Phase 9 candidate ranking."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from triqto.metrics import BornMetricBundle

from .config import ActionEngineConfig
from .constants import PRIMARY_REWARD_METRICS


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    metric_names: np.ndarray
    baseline_metric_values: np.ndarray
    candidate_metric_values: np.ndarray
    improvement_values: np.ndarray
    weighted_improvement: float
    depth_penalty: float
    gate_penalty: float
    edit_penalty: float
    risk_penalty: float
    reward: float
    primary_metric_nonworsening: bool
    dominates_baseline: bool
    exact_born_recovery: bool


def primary_metric_array(bundle: BornMetricBundle) -> np.ndarray:
    """Extract Phase 9 primary Born distances in their fixed versioned order."""
    if not isinstance(bundle, BornMetricBundle):
        raise TypeError("bundle must be BornMetricBundle")
    values: list[float] = []
    for name in PRIMARY_REWARD_METRICS:
        if name not in bundle.metrics:
            raise ValueError(f"BornMetricBundle is missing required metric {name}")
        value = bundle.metrics[name].value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Born metric {name} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Primary Born metric {name} must be finite")
        if numeric < 0.0:
            raise ValueError(f"Primary Born metric {name} must be nonnegative")
        values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def score_action_rollout(
    *,
    baseline_metrics: BornMetricBundle,
    candidate_metrics: BornMetricBundle,
    depth_delta: int,
    gate_delta: int,
    edit_count: int,
    risk_score: float,
    config: ActionEngineConfig,
) -> RewardBreakdown:
    """Score one candidate against the uncorrected distorted baseline.

    The reward is transparent and deterministic. It is training evidence for a future
    learned policy, not itself a learned policy and not a proof that the edit is safe on
    noisy simulation or hardware.
    """
    for name, value in (
        ("depth_delta", depth_delta),
        ("gate_delta", gate_delta),
        ("edit_count", edit_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer and not bool")
    if edit_count < 0:
        raise ValueError("edit_count must be nonnegative")
    if isinstance(risk_score, bool) or not isinstance(risk_score, (int, float)):
        raise TypeError("risk_score must be numeric and not bool")
    risk = float(risk_score)
    if not math.isfinite(risk) or risk < 0.0 or risk > 1.0:
        raise ValueError("risk_score must be finite and in [0, 1]")

    baseline = primary_metric_array(baseline_metrics)
    candidate = primary_metric_array(candidate_metrics)
    improvement = baseline - candidate
    weights = np.asarray(
        [
            config.reward_total_variation_weight,
            config.reward_jensen_shannon_weight,
            config.reward_hellinger_weight,
        ],
        dtype=np.float64,
    )
    weighted_improvement = float(np.dot(weights, improvement))
    depth_penalty = config.depth_penalty_weight * max(depth_delta, 0)
    gate_penalty = config.gate_penalty_weight * max(gate_delta, 0)
    edit_penalty = config.edit_penalty_weight * edit_count
    risk_penalty = config.risk_penalty_weight * risk
    reward = float(
        weighted_improvement
        - depth_penalty
        - gate_penalty
        - edit_penalty
        - risk_penalty
    )
    if not math.isfinite(reward):
        raise ValueError("Computed action reward is non-finite")

    nonworsening = bool(np.all(candidate <= baseline + config.improvement_atol))
    dominates = bool(
        nonworsening
        and np.any(candidate < baseline - config.improvement_atol)
    )
    # Total variation is the direct L1-equivalence criterion for Born distributions.
    # Derived square-root metrics can retain amplified floating noise near exact equality,
    # so they remain ranking metrics but do not define this label.
    exact_recovery = bool(candidate[0] <= config.improvement_atol)
    return RewardBreakdown(
        metric_names=np.asarray(PRIMARY_REWARD_METRICS, dtype="<U32"),
        baseline_metric_values=baseline,
        candidate_metric_values=candidate,
        improvement_values=improvement.astype(np.float64, copy=False),
        weighted_improvement=weighted_improvement,
        depth_penalty=float(depth_penalty),
        gate_penalty=float(gate_penalty),
        edit_penalty=float(edit_penalty),
        risk_penalty=float(risk_penalty),
        reward=reward,
        primary_metric_nonworsening=nonworsening,
        dominates_baseline=dominates,
        exact_born_recovery=exact_recovery,
    )


__all__ = ["RewardBreakdown", "primary_metric_array", "score_action_rollout"]
