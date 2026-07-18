"""Finite-shot uncertainty summaries without treating multinomial outcomes as independent."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
from scipy.stats import beta


def probabilities_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    shots = int(sum(int(value) for value in counts.values()))
    if shots <= 0:
        raise ValueError("shot counts must sum to a positive value")
    return {key: int(counts[key]) / shots for key in sorted(counts)}


def dirichlet_jeffreys_summary(
    counts: Mapping[str, int],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return Jeffreys-marginal intervals and shared multinomial metadata."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    ordered = {str(key): int(value) for key, value in sorted(counts.items())}
    if any(value < 0 for value in ordered.values()):
        raise ValueError("counts must be nonnegative")
    shots = int(sum(ordered.values()))
    if shots <= 0:
        raise ValueError("counts must sum to a positive shot count")
    support_size = len(ordered)
    alpha0 = 0.5 * support_size
    tail = 0.5 * (1.0 - confidence)
    outcomes: dict[str, Any] = {}
    for key, count in ordered.items():
        alpha = count + 0.5
        beta_parameter = shots - count + alpha0 - 0.5
        posterior_mean = alpha / (shots + alpha0)
        lower = float(beta.ppf(tail, alpha, beta_parameter))
        upper = float(beta.ppf(1.0 - tail, alpha, beta_parameter))
        outcomes[key] = {
            "count": count,
            "frequency": count / shots,
            "posterior_mean": posterior_mean,
            "marginal_interval": [lower, upper],
        }
    return {
        "method": "dirichlet_jeffreys_marginals_v1",
        "multinomial_dependence": True,
        "confidence": confidence,
        "shots": shots,
        "support_size": support_size,
        "outcomes": outcomes,
    }


def repeated_batch_summary(
    batches: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    if not batches:
        return {
            "batch_count": 0,
            "support": [],
            "mean_probabilities": {},
            "sample_variances": {},
        }
    support = tuple(sorted({str(key) for batch in batches for key in batch}))
    matrix: list[list[float]] = []
    shot_counts: list[int] = []
    for batch in batches:
        total = int(sum(int(value) for value in batch.values()))
        if total <= 0:
            raise ValueError("each repeated batch must contain positive shots")
        shot_counts.append(total)
        matrix.append([int(batch.get(key, 0)) / total for key in support])
    values = np.asarray(matrix, dtype=float)
    means = np.mean(values, axis=0)
    variances = (
        np.var(values, axis=0, ddof=1)
        if values.shape[0] > 1
        else np.zeros(values.shape[1], dtype=float)
    )
    return {
        "batch_count": len(batches),
        "support": list(support),
        "shot_counts": shot_counts,
        "total_shots": int(sum(shot_counts)),
        "mean_probabilities": {
            key: float(means[index]) for index, key in enumerate(support)
        },
        "sample_variances": {
            key: float(variances[index]) for index, key in enumerate(support)
        },
    }


def bootstrap_distance_interval(
    left_counts: Mapping[str, int],
    right_counts: Mapping[str, int],
    *,
    distance_fn: Any,
    seed: int,
    draws: int = 512,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap a derived distribution distance using multinomial resampling."""
    if draws <= 1:
        raise ValueError("draws must exceed one")
    support = tuple(sorted(set(left_counts) | set(right_counts)))
    left_shots = int(sum(int(value) for value in left_counts.values()))
    right_shots = int(sum(int(value) for value in right_counts.values()))
    if left_shots <= 0 or right_shots <= 0:
        raise ValueError("both count maps must contain positive shots")
    left_probability = np.asarray(
        [int(left_counts.get(key, 0)) / left_shots for key in support], dtype=float
    )
    right_probability = np.asarray(
        [int(right_counts.get(key, 0)) / right_shots for key in support], dtype=float
    )
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for index in range(draws):
        left_sample = rng.multinomial(left_shots, left_probability) / left_shots
        right_sample = rng.multinomial(right_shots, right_probability) / right_shots
        left_map = {key: float(left_sample[position]) for position, key in enumerate(support)}
        right_map = {key: float(right_sample[position]) for position, key in enumerate(support)}
        values[index] = float(distance_fn(left_map, right_map))
    tail = 0.5 * (1.0 - confidence)
    return {
        "method": "multinomial_bootstrap_v1",
        "draws": draws,
        "confidence": confidence,
        "estimate": float(
            distance_fn(
                {key: float(left_probability[position]) for position, key in enumerate(support)},
                {key: float(right_probability[position]) for position, key in enumerate(support)},
            )
        ),
        "interval": [
            float(np.quantile(values, tail)),
            float(np.quantile(values, 1.0 - tail)),
        ],
        "standard_error": float(np.std(values, ddof=1)),
    }
