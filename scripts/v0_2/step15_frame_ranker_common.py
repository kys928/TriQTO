#!/usr/bin/env python3
"""Shared utilities for Step-15 FIT-only frame-ranking adaptation.

The adapter is deliberately narrow: it adds one learned scalar correction per
candidate to the frozen Step-14 density-corrected frame anchor score. Candidate
generation, finite-shot calibration, equivalence smoothing, and the frozen
three-way mechanism head are unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import step14_equivalence_aware_common as equiv

FEATURE_NAMES = (
    "baseline_anchor_logit",
    "frame_evidence",
    "log_kernel_density",
    "profile_x",
    "profile_y",
    "profile_z",
    "profile_max",
    "profile_margin",
    "profile_std",
    "profile_entropy",
    *tuple(f"canonical_{i:02d}" for i in range(24)),
    "candidate_qubit_fraction",
    "candidate_boundary_fraction",
)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    maximum = np.max(x, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), x - maximum, -np.inf)
    summed = np.sum(np.exp(safe), axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.maximum(summed, 1.0e-300)), axis=axis)


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    maximum = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - maximum)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1.0e-300)


def anchor_score_components(
    profiles: np.ndarray,
    pair_distance: np.ndarray,
    pair_gate: np.ndarray,
    *,
    tau: float,
    frame_temperature: float,
) -> dict[str, np.ndarray]:
    profiles = np.asarray(profiles, dtype=np.float64)
    distance = np.asarray(pair_distance, dtype=np.float64)
    gate = np.asarray(pair_gate, dtype=np.bool_)
    count = profiles.shape[0]
    if profiles.shape != (count, 3):
        raise RuntimeError("profiles must have shape [candidate,3]")
    if distance.shape != (count, count) or gate.shape != (count, count):
        raise RuntimeError("pair geometry shape mismatch")
    if not (tau > 0.0 and frame_temperature > 0.0):
        raise ValueError("tau and frame_temperature must be positive")

    kernel = np.where(gate, np.exp(-distance / float(tau)), 0.0)
    np.fill_diagonal(kernel, 1.0)
    density = np.maximum(kernel.sum(axis=1), 1.0e-12)
    frame_evidence = _logsumexp(profiles, axis=1) - math.log(3.0)
    baseline_anchor_logit = frame_evidence / float(frame_temperature) - np.log(density)
    return {
        "kernel": kernel,
        "density": density,
        "frame_evidence": frame_evidence,
        "baseline_anchor_logit": baseline_anchor_logit,
    }


def build_rank_features_one(
    candidate_features: np.ndarray,
    profiles: np.ndarray,
    pair_distance: np.ndarray,
    pair_gate: np.ndarray,
    candidates: Sequence[tuple[int, int]],
    *,
    qubit_count: int,
    gate_count: int,
    tau: float,
    frame_temperature: float,
) -> np.ndarray:
    canonical = np.asarray(candidate_features, dtype=np.float64)
    profiles = np.asarray(profiles, dtype=np.float64)
    count = len(candidates)
    if canonical.shape != (count, 24):
        raise RuntimeError(f"canonical features must be [candidate,24], got {canonical.shape}")
    if profiles.shape != (count, 3):
        raise RuntimeError("profile shape mismatch")
    if qubit_count < 1 or gate_count < 1:
        raise ValueError("qubit_count and gate_count must be positive")

    comp = anchor_score_components(
        profiles, pair_distance, pair_gate,
        tau=tau, frame_temperature=frame_temperature,
    )
    profile_prob = _softmax(profiles, axis=1)
    profile_entropy = -np.sum(profile_prob * np.log(np.maximum(profile_prob, 1.0e-300)), axis=1)
    sorted_profiles = np.sort(profiles, axis=1)
    profile_margin = sorted_profiles[:, -1] - sorted_profiles[:, -2]

    q_den = max(int(qubit_count) - 1, 1)
    b_den = max(int(gate_count), 1)
    structural = np.asarray(
        [[float(q) / q_den, float(boundary) / b_den] for q, boundary in candidates],
        dtype=np.float64,
    )
    raw = np.column_stack(
        [
            comp["baseline_anchor_logit"],
            comp["frame_evidence"],
            np.log(comp["density"]),
            profiles,
            np.max(profiles, axis=1),
            profile_margin,
            np.std(profiles, axis=1),
            profile_entropy,
            canonical,
            structural,
        ]
    )
    if raw.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((raw.shape, len(FEATURE_NAMES)))

    # The adapter should learn relative candidate preference, not query-wide
    # offsets. Query-centering also makes deployment invariant to a constant
    # shift in all candidate profile log-likelihoods.
    return (raw - raw.mean(axis=0, keepdims=True)).astype(np.float32)


def fit_standardizer(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    m = np.asarray(mask, dtype=np.bool_)
    if x.ndim != 3 or m.shape != x.shape[:2]:
        raise RuntimeError("feature/mask shape mismatch")
    valid = x[m]
    if len(valid) == 0:
        raise RuntimeError("cannot fit standardizer with zero valid candidates")
    mean = valid.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = valid.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1.0e-6)
    return mean, std


def apply_standardizer(
    features: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float32) - np.asarray(mean, dtype=np.float32)[None, None, :]) / np.asarray(std, dtype=np.float32)[None, None, :]
    out = x.astype(np.float32, copy=False)
    out[~np.asarray(mask, dtype=np.bool_)] = 0.0
    return out


def _baseline_scores_from_feature_tensor(features: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64)[..., 0]


def hard_negative_mask(
    features: np.ndarray,
    valid_mask: np.ndarray,
    positive_mask: np.ndarray,
    *,
    hard_negative_count: int,
) -> np.ndarray:
    if hard_negative_count < 1:
        raise ValueError("hard_negative_count must be >=1")
    baseline = _baseline_scores_from_feature_tensor(features)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    positive = np.asarray(positive_mask, dtype=np.bool_)
    if valid.shape != positive.shape or valid.shape != baseline.shape:
        raise RuntimeError("hard-negative shape mismatch")
    out = np.zeros_like(valid)
    for i in range(len(valid)):
        neg = np.flatnonzero(valid[i] & ~positive[i])
        if len(neg) == 0:
            continue
        order = neg[np.argsort(-baseline[i, neg], kind="stable")]
        out[i, order[:hard_negative_count]] = True
    return out


def _masked_logsumexp(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    return torch.logsumexp(values.masked_fill(~mask, neg_inf), dim=1)


def train_linear_rank_adapter(
    features: np.ndarray,
    valid_mask: np.ndarray,
    positive_mask: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    pairwise_weight: float,
    pairwise_margin: float,
    hard_negative_count: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if not learning_rate > 0.0 or not weight_decay >= 0.0 or not pairwise_weight >= 0.0:
        raise ValueError("invalid optimizer/loss hyperparameters")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    x_raw = np.asarray(features, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    positive = np.asarray(positive_mask, dtype=np.bool_)
    if x_raw.ndim != 3 or valid.shape != x_raw.shape[:2] or positive.shape != valid.shape:
        raise RuntimeError("rank-adapter tensor shape mismatch")
    if np.any(np.sum(valid, axis=1) < 1) or np.any(np.sum(positive & valid, axis=1) < 1):
        raise RuntimeError("every rank bag must contain at least one valid positive")

    mean, std = fit_standardizer(x_raw, valid)
    x = apply_standardizer(x_raw, valid, mean, std)
    hard = hard_negative_mask(x_raw, valid, positive, hard_negative_count=hard_negative_count)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    weights = nn.Parameter(torch.zeros(x.shape[2], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([weights], lr=float(learning_rate), weight_decay=0.0)
    rng = np.random.default_rng(int(seed))
    last = {"total": 0.0, "listwise": 0.0, "pairwise": 0.0, "l2": 0.0}

    for _epoch in range(int(epochs)):
        order = rng.permutation(len(x))
        totals = np.zeros(4, dtype=np.float64)
        seen = 0
        for start in range(0, len(order), int(batch_size)):
            idx = order[start:start + int(batch_size)]
            xb = torch.from_numpy(x[idx]).to(device)
            vb = torch.from_numpy(valid[idx]).to(device)
            pb = torch.from_numpy(positive[idx]).to(device)
            hb = torch.from_numpy(hard[idx]).to(device)
            baseline = torch.from_numpy(x_raw[idx, :, 0]).to(device)
            correction = torch.einsum("bcf,f->bc", xb, weights)
            scores = baseline + correction

            total_lse = _masked_logsumexp(scores, vb)
            positive_lse = _masked_logsumexp(scores, pb & vb)
            listwise = torch.mean(total_lse - positive_lse)

            pos_count = torch.sum(pb & vb, dim=1).clamp_min(1).to(scores.dtype)
            positive_mean_log_score = positive_lse - torch.log(pos_count)
            pair_terms = torch.nn.functional.softplus(
                float(pairwise_margin) - positive_mean_log_score[:, None] + scores
            )
            hard_count = torch.sum(hb, dim=1)
            per_bag_pair = torch.where(
                hard_count > 0,
                torch.sum(pair_terms * hb.to(pair_terms.dtype), dim=1) / hard_count.clamp_min(1).to(pair_terms.dtype),
                torch.zeros_like(positive_mean_log_score),
            )
            pairwise = torch.mean(per_bag_pair)
            l2 = torch.sum(weights * weights)
            loss = listwise + float(pairwise_weight) * pairwise + float(weight_decay) * l2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = len(idx)
            totals += n * np.asarray([
                float(loss.detach().cpu()),
                float(listwise.detach().cpu()),
                float(pairwise.detach().cpu()),
                float(l2.detach().cpu()),
            ])
            seen += n
        last = dict(zip(("total", "listwise", "pairwise", "l2"), (totals / max(seen, 1)).tolist()))

    return {
        "weights": weights.detach().cpu().numpy().astype(np.float32),
        "mean": mean,
        "std": std,
        "final_loss": last,
        "hard_negative_count": int(hard_negative_count),
        "seed": int(seed),
    }


def adapter_scores(
    features: np.ndarray,
    valid_mask: np.ndarray,
    adapter: Mapping[str, Any],
) -> np.ndarray:
    x = apply_standardizer(
        features,
        valid_mask,
        np.asarray(adapter["mean"], dtype=np.float32),
        np.asarray(adapter["std"], dtype=np.float32),
    )
    w = np.asarray(adapter["weights"], dtype=np.float32)
    if w.shape != (x.shape[2],):
        raise RuntimeError("adapter weight dimension mismatch")
    baseline = _baseline_scores_from_feature_tensor(features)
    scores = baseline + np.einsum("bcf,f->bc", x, w, optimize=True)
    scores = np.asarray(scores, dtype=np.float64)
    scores[~np.asarray(valid_mask, dtype=np.bool_)] = -np.inf
    return scores


def ranking_metrics(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    positive_mask: np.ndarray,
    *,
    ece_bins: int = 10,
) -> dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    positive = np.asarray(positive_mask, dtype=np.bool_)
    if s.shape != valid.shape or positive.shape != valid.shape:
        raise RuntimeError("ranking metric shape mismatch")
    n = len(s)
    top1 = np.zeros(n, dtype=np.float64)
    top3 = np.zeros(n, dtype=np.float64)
    top5 = np.zeros(n, dtype=np.float64)
    reciprocal = np.zeros(n, dtype=np.float64)
    set_prob = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    correctness = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = np.flatnonzero(valid[i])
        if len(idx) < 1:
            raise RuntimeError("empty candidate bag in ranking metrics")
        pos = positive[i, idx]
        if not np.any(pos):
            raise RuntimeError("candidate bag has no positive in ranking metrics")
        local_scores = s[i, idx]
        order_local = np.argsort(-local_scores, kind="stable")
        ordered_idx = idx[order_local]
        ordered_pos = positive[i, ordered_idx]
        first_rank = int(np.flatnonzero(ordered_pos)[0]) + 1
        reciprocal[i] = 1.0 / first_rank
        top1[i] = float(np.any(ordered_pos[:1]))
        top3[i] = float(np.any(ordered_pos[: min(3, len(ordered_pos))]))
        top5[i] = float(np.any(ordered_pos[: min(5, len(ordered_pos))]))
        p = _softmax(local_scores[None, :], axis=1)[0]
        set_prob[i] = float(np.sum(p[pos]))
        j = int(np.argmax(p))
        confidence[i] = float(p[j])
        correctness[i] = float(pos[j])

    nll = float(np.mean(-np.log(np.maximum(set_prob, 1.0e-300))))
    brier = float(np.mean(np.square(1.0 - set_prob)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    for b in range(int(ece_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        member = (confidence >= lo) & ((confidence < hi) if b + 1 < len(edges) - 1 else (confidence <= hi))
        if np.any(member):
            ece += float(np.mean(member)) * abs(float(np.mean(correctness[member])) - float(np.mean(confidence[member])))
    return {
        "top1": float(np.mean(top1)),
        "top3": float(np.mean(top3)),
        "top5": float(np.mean(top5)),
        "mrr": float(np.mean(reciprocal)),
        "set_nll": nll,
        "set_brier": brier,
        "top_candidate_ece": float(ece),
        "mean_positive_posterior_mass": float(np.mean(set_prob)),
    }


def adapter_to_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "weights": [float(v) for v in np.asarray(value["weights"], dtype=np.float64)],
        "mean": [float(v) for v in np.asarray(value["mean"], dtype=np.float64)],
        "std": [float(v) for v in np.asarray(value["std"], dtype=np.float64)],
        "seed": int(value["seed"]),
        "hard_negative_count": int(value["hard_negative_count"]),
        "final_loss": {k: float(v) for k, v in value["final_loss"].items()},
    }


def adapter_from_json(value: Mapping[str, Any]) -> dict[str, Any]:
    if tuple(value.get("feature_names", ())) != tuple(FEATURE_NAMES):
        raise RuntimeError("Step-15 adapter feature schema drift")
    return {
        "weights": np.asarray(value["weights"], dtype=np.float32),
        "mean": np.asarray(value["mean"], dtype=np.float32),
        "std": np.asarray(value["std"], dtype=np.float32),
        "seed": int(value["seed"]),
        "hard_negative_count": int(value["hard_negative_count"]),
        "final_loss": dict(value.get("final_loss", {})),
    }


def equivalence_aware_pool_one_with_anchor_correction(
    candidate_logits: np.ndarray,
    profiles: np.ndarray,
    pair_distance: np.ndarray,
    pair_gate: np.ndarray,
    anchor_correction: np.ndarray,
    *,
    tau: float,
    frame_temperature: float,
) -> np.ndarray:
    correction = np.asarray(anchor_correction, dtype=np.float64)
    count = np.asarray(candidate_logits).shape[0]
    if correction.shape != (count,):
        raise RuntimeError("anchor correction shape mismatch")
    # This explicit delegation is the Step-15 zero-adapter regression contract:
    # zero correction is byte-for-byte the frozen Step-14 implementation.
    if np.count_nonzero(correction) == 0:
        return equiv.equivalence_aware_pool_one(
            candidate_logits, profiles, pair_distance, pair_gate,
            tau=tau, frame_temperature=frame_temperature,
        )

    logits = np.asarray(candidate_logits, dtype=np.float64)
    comp = anchor_score_components(
        profiles, pair_distance, pair_gate,
        tau=tau, frame_temperature=frame_temperature,
    )
    anchor = comp["baseline_anchor_logit"] + correction
    anchor = anchor - float(_logsumexp(anchor[None, :], axis=1)[0])
    kernel = comp["kernel"]
    log_kernel = np.where(kernel > 0.0, np.log(np.maximum(kernel, 1.0e-300)), -np.inf)
    neighborhood_norm = _logsumexp(log_kernel, axis=1)
    smoothed = np.empty((count, 3), dtype=np.float64)
    for mechanism in range(3):
        smoothed[:, mechanism] = (
            _logsumexp(log_kernel + logits[:, mechanism][None, :], axis=1)
            - neighborhood_norm
        )
    return _logsumexp(anchor[:, None] + smoothed, axis=0)


def equivalence_aware_pool_batch_with_adapter(
    candidate_logits: np.ndarray,
    profiles: np.ndarray,
    pair_distance: np.ndarray,
    pair_gate: np.ndarray,
    candidate_mask: np.ndarray,
    rank_features: np.ndarray,
    adapter: Mapping[str, Any],
    *,
    tau: float,
    frame_temperature: float,
) -> np.ndarray:
    mask = np.asarray(candidate_mask, dtype=np.bool_)
    scores = adapter_scores(rank_features, mask, adapter)
    baseline = _baseline_scores_from_feature_tensor(rank_features)
    correction = scores - baseline
    out = np.empty((len(mask), 3), dtype=np.float64)
    for i in range(len(mask)):
        count = int(mask[i].sum())
        out[i] = equivalence_aware_pool_one_with_anchor_correction(
            candidate_logits[i, :count],
            profiles[i, :count],
            pair_distance[i, :count, :count],
            pair_gate[i, :count, :count],
            correction[i, :count],
            tau=tau,
            frame_temperature=frame_temperature,
        )
    return out
