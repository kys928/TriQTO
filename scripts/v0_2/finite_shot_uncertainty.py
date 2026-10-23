#!/usr/bin/env python3
"""Observed-count-only uncertainty primitives for TriQTO finite-shot geometry.

Exploratory mechanics, not a validated stabilization remedy. No simulator
probabilities or privileged hidden-state inputs are accepted by this module.
"""
from __future__ import annotations

import numpy as np


def observed_mean_covariance(counts: np.ndarray, outcome_features: np.ndarray):
    """Empirical mean and unbiased estimated covariance of its sample mean.

    counts [outcome], outcome_features [outcome, statistic].
    All statistics in one basis are correlated; that covariance is retained.
    N <= 1 is refused rather than assigned a falsely certain zero variance.
    """
    counts = np.asarray(counts)
    features = np.asarray(outcome_features, dtype=np.float64)
    if counts.ndim != 1 or features.ndim != 2 or features.shape[0] != len(counts):
        raise ValueError("counts/outcome-feature shapes do not match")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be nonnegative finite integers")
    if not np.all(np.isfinite(features)):
        raise ValueError("outcome-feature table must be finite")
    shots = float(np.sum(counts, dtype=np.float64))
    if shots <= 1:
        raise ValueError("at least two observed shots are required")
    p = counts.astype(np.float64) / shots
    mean = p @ features
    centered = features - mean[None, :]
    cov_mean = (centered.T * p) @ centered / (shots - 1.0)
    cov_mean = 0.5 * (cov_mean + cov_mean.T)
    return mean, cov_mean


def finite_difference_jacobian_covariance(
    counts: np.ndarray, outcome_features: np.ndarray, probe_strength: float
):
    """Reconstruct three Jacobian axes and per-axis covariance from observed counts.

    counts [mechanism=3, sign=2, basis=3, outcome].
    outcome_features [basis=3, outcome, statistics_per_basis].
    + and - are independent calibration acquisitions under the frozen contract.
    The independent basis blocks are retained, and the within-basis statistic
    covariance is not discarded. No assumption of equal counts is required.
    """
    counts = np.asarray(counts)
    features = np.asarray(outcome_features, dtype=np.float64)
    if counts.ndim != 4 or counts.shape[:3] != (3, 2, 3):
        raise ValueError("expected [3 mechanisms, 2 signs, 3 bases, outcomes]")
    if features.ndim != 3 or features.shape[0] != 3 or features.shape[1] != counts.shape[-1]:
        raise ValueError("expected [3 bases, outcomes, statistics]")
    epsilon = float(probe_strength)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("probe strength must be positive")
    dimension = int(features.shape[2]) * 3
    jacobian = np.empty((dimension, 3), dtype=np.float64)
    covariances = np.zeros((3, dimension, dimension), dtype=np.float64)
    den = 2.0 * epsilon
    for axis in range(3):
        for basis in range(3):
            plus, cov_plus = observed_mean_covariance(counts[axis, 0, basis], features[basis])
            minus, cov_minus = observed_mean_covariance(counts[axis, 1, basis], features[basis])
            low = basis * features.shape[2]
            high = low + features.shape[2]
            jacobian[low:high, axis] = (plus - minus) / den
            covariances[axis, low:high, low:high] = (cov_plus + cov_minus) / (den * den)
    return jacobian, covariances


def observed_count_bootstrap(
    counts: np.ndarray, outcome_features: np.ndarray, probe_strength: float,
    delta: np.ndarray, weights: np.ndarray, *, replicates: int, seed: int,
):
    """Jointly bootstrap coordinate and matched-response uncertainty.

    Uses empirical frequencies from already-observed counts only. Repeats
    the original count totals and passes each sampled Jacobian to the frozen
    feature map. The six jointly estimated variables are [coordinates(3),
    signed matched responses(3)] and retain cross-covariance.
    This plug-in bootstrap can underestimate uncertainty at sparse boundaries.
    """
    if int(replicates) < 3:
        raise ValueError("at least three bootstrap replicates required")
    from analyze_step14_local_frame_canonicalization import canonicalize_evidence
    observed_j, jac_cov = finite_difference_jacobian_covariance(
        counts, outcome_features, probe_strength
    )
    delta = np.asarray(delta, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if delta.shape != (observed_j.shape[0],) or weights.shape != delta.shape:
        raise ValueError("evidence/weight width does not match Jacobian")
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(weights)):
        raise ValueError("delta and weights must be finite")
    observed_features, _, _ = canonicalize_evidence(delta, observed_j, weights)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    samples = np.empty((int(replicates), 6), dtype=np.float64)
    raw_counts = np.asarray(counts)
    for i in range(int(replicates)):
        resampled = np.empty_like(raw_counts)
        for axis in range(3):
            for sign in range(2):
                for basis in range(3):
                    current = raw_counts[axis, sign, basis]
                    shots = int(current.sum())
                    probs = current.astype(np.float64) / shots
                    resampled[axis, sign, basis] = rng.multinomial(shots, probs)
        frame, _ = finite_difference_jacobian_covariance(
            resampled, outcome_features, probe_strength
        )
        result, _, _ = canonicalize_evidence(delta, frame, weights)
        samples[i, :3] = result[:3]
        samples[i, 3:] = result[9:12]
    return {
        "observed_coords_matched": np.r_[observed_features[:3], observed_features[9:12]],
        "bootstrap_mean": np.mean(samples, axis=0),
        "bootstrap_covariance": np.cov(samples, rowvar=False, ddof=1),
        "jacobian_axis_covariances": jac_cov,
        "replicates": int(replicates),
        "seed": int(seed),
        "interpretation": "plug-in finite-shot uncertainty; analysis-only until validated on independent development",
    }
