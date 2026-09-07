#!/usr/bin/env python3
"""Shared fit-only utilities for Step-14 equivalence-aware latent-frame inference.

Deployable calculations in this module use only:
- model-visible finite-shot diagnostic evidence,
- the known reference circuit,
- finite-shot +/- X/Y/Z calibration responses for plausible candidate frames.

No true affected-qubit or injection-boundary metadata is accepted by these APIs.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import analyze_step14_latent_frame_inference as latent


def family_fold(family_id: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("fold count must be >=2")
    digest = hashlib.sha256(str(family_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % int(folds)


def _normalized_columns(jacobian: np.ndarray) -> np.ndarray:
    array = np.asarray(jacobian, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise RuntimeError("candidate frame Jacobian must be [evidence,3]")
    norms = np.linalg.norm(array, axis=0)
    return array / np.maximum(norms, 1.0e-12)[None, :]


def _best_nonidentity_assignment_mean(similarity: np.ndarray) -> float:
    best = -np.inf
    for perm in itertools.permutations(range(3)):
        if perm == (0, 1, 2):
            continue
        best = max(best, float(np.mean([similarity[i, perm[i]] for i in range(3)])))
    return float(best)


def finite_frame_pair_geometry(jacobians: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise response distance and mechanism-preserving gate.

    distance[i,j] = 1 - min same-axis |cosine|;
    gate[i,j] is true iff the identity axis assignment beats every non-identity
    permutation. This preserves X/Y/Z mechanism semantics rather than clustering
    arbitrary 3D subspaces.
    """
    normalized = [_normalized_columns(value) for value in jacobians]
    count = len(normalized)
    distance = np.ones((count, count), dtype=np.float32)
    gate = np.zeros((count, count), dtype=np.bool_)
    for i in range(count):
        for j in range(i, count):
            similarity = np.abs(normalized[i].T @ normalized[j])
            diagonal = np.diag(similarity)
            same_axis = float(np.min(diagonal))
            identity_mean = float(np.mean(diagonal))
            margin = identity_mean - _best_nonidentity_assignment_mean(similarity)
            accepted = bool(margin > 0.0)
            d = float(np.clip(1.0 - same_axis, 0.0, 1.0))
            distance[i, j] = distance[j, i] = d
            gate[i, j] = gate[j, i] = accepted
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(gate, True)
    return distance, gate


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), array - maximum, -np.inf)
    summed = np.sum(np.exp(safe), axis=axis, keepdims=True)
    out = maximum + np.log(np.maximum(summed, 1.0e-300))
    return np.squeeze(out, axis=axis)


def _softmax_logweights(log_weights: np.ndarray) -> np.ndarray:
    logw = np.asarray(log_weights, dtype=np.float64)
    norm = _logsumexp(logw[None, :], axis=1)[0]
    return logw - norm


def equivalence_aware_pool_one(
    candidate_logits: np.ndarray,
    profile_scores: np.ndarray,
    pair_distance: np.ndarray,
    pair_gate: np.ndarray,
    *,
    tau: float,
    frame_temperature: float,
) -> np.ndarray:
    """Pool one candidate bag into three mechanism logits."""
    logits = np.asarray(candidate_logits, dtype=np.float64)
    profiles = np.asarray(profile_scores, dtype=np.float64)
    distance = np.asarray(pair_distance, dtype=np.float64)
    gate = np.asarray(pair_gate, dtype=np.bool_)
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise RuntimeError("candidate logits must have shape [candidate,3]")
    count = logits.shape[0]
    if profiles.shape != (count, 3) or distance.shape != (count, count) or gate.shape != (count, count):
        raise RuntimeError("equivalence-aware pooling shape mismatch")
    if not (tau > 0.0 and frame_temperature > 0.0):
        raise ValueError("tau and frame_temperature must be positive")

    kernel = np.where(gate, np.exp(-distance / float(tau)), 0.0)
    np.fill_diagonal(kernel, 1.0)
    density = np.maximum(kernel.sum(axis=1), 1.0e-12)

    frame_evidence = _logsumexp(profiles, axis=1) - math.log(3.0)
    anchor_log_posterior = _softmax_logweights(
        frame_evidence / float(frame_temperature) - np.log(density)
    )

    log_kernel = np.where(kernel > 0.0, np.log(np.maximum(kernel, 1.0e-300)), -np.inf)
    neighborhood_norm = _logsumexp(log_kernel, axis=1)
    smoothed = np.empty((count, 3), dtype=np.float64)
    for mechanism in range(3):
        smoothed[:, mechanism] = (
            _logsumexp(log_kernel + logits[:, mechanism][None, :], axis=1)
            - neighborhood_norm
        )

    return _logsumexp(anchor_log_posterior[:, None] + smoothed, axis=0)


def equivalence_aware_pool_batch(
    candidate_logits: np.ndarray,
    profile_scores: np.ndarray,
    pair_distance: np.ndarray,
    pair_gate: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    tau: float,
    frame_temperature: float,
) -> np.ndarray:
    mask = np.asarray(candidate_mask, dtype=np.bool_)
    output = np.empty((mask.shape[0], 3), dtype=np.float64)
    for i in range(mask.shape[0]):
        count = int(mask[i].sum())
        if count < 1 or not np.all(mask[i, :count]) or np.any(mask[i, count:]):
            raise RuntimeError("candidate mask must be left-packed and nonempty")
        output[i] = equivalence_aware_pool_one(
            candidate_logits[i, :count],
            profile_scores[i, :count],
            pair_distance[i, :count, :count],
            pair_gate[i, :count, :count],
            tau=tau,
            frame_temperature=frame_temperature,
        )
    return output


def train_uniform_candidate_network(
    x_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    mask_eval: np.ndarray,
    *,
    seed: int,
    latent_cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Reproduce the frozen latent candidate network/bag objective.

    Equivalence awareness is intentionally not trained into the network; it is a
    post-candidate inference rule. This keeps the intervention isolated.
    """
    spec = latent_cfg["latent_mil_probe"]
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    valid_train = x_train[mask_train]
    mean = valid_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = valid_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1.0e-6)

    x_train_n = (x_train - mean[None, None, :]) / std[None, None, :]
    x_eval_n = (x_eval - mean[None, None, :]) / std[None, None, :]
    x_train_n[~mask_train] = 0.0
    x_eval_n[~mask_eval] = 0.0

    model = latent.LatentFrameMIL(
        x_train.shape[2],
        int(spec["shared_hidden_dim"]),
        float(spec["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    batch_size = int(spec["batch_size"])
    epochs = int(spec["epochs"])
    generator = np.random.default_rng(int(seed))
    final_loss = 0.0

    model.train()
    for _epoch in range(epochs):
        permutation = generator.permutation(len(y_train))
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(x_train_n[indices]).to(device)
            mb = torch.from_numpy(mask_train[indices]).to(device)
            yb = torch.from_numpy(y_train[indices]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            pooled, _candidate = model(xb, mb)
            loss = nn.functional.cross_entropy(pooled, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(spec["gradient_clip_norm"])
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        final_loss = loss_sum / max(seen, 1)

    model.eval()

    def infer(x_normalized: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pooled_parts: list[np.ndarray] = []
        candidate_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_normalized), batch_size):
                xb = torch.from_numpy(x_normalized[start:start + batch_size]).to(device)
                mb = torch.from_numpy(mask[start:start + batch_size]).to(device)
                pooled, candidate = model(xb, mb)
                pooled_parts.append(pooled.cpu().numpy())
                candidate_parts.append(candidate.cpu().numpy())
        return (
            np.concatenate(pooled_parts).astype(np.float64),
            np.concatenate(candidate_parts).astype(np.float64),
        )

    train_pooled, _train_candidate = infer(x_train_n, mask_train)
    eval_pooled, eval_candidate = infer(x_eval_n, mask_eval)
    train_ba, train_recall = latent.balanced_accuracy(y_train, train_pooled)

    return {
        "seed": int(seed),
        "final_train_loss": float(final_loss),
        "train_balanced_accuracy": float(train_ba),
        "train_recall": [float(v) for v in train_recall],
        "eval_uniform_pooled_logits": eval_pooled,
        "eval_candidate_logits": eval_candidate,
        "mean": mean,
        "std": std,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


def infer_candidate_network(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    state_dict: Mapping[str, torch.Tensor],
    mean: np.ndarray,
    std: np.ndarray,
    latent_cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    spec = latent_cfg["latent_mil_probe"]
    normalized = (x - mean[None, None, :]) / std[None, None, :]
    normalized[~mask] = 0.0
    model = latent.LatentFrameMIL(
        x.shape[2],
        int(spec["shared_hidden_dim"]),
        float(spec["dropout"]),
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    batch_size = int(spec["batch_size"])
    pooled_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            xb = torch.from_numpy(normalized[start:start + batch_size]).to(device)
            mb = torch.from_numpy(mask[start:start + batch_size]).to(device)
            pooled, candidate = model(xb, mb)
            pooled_parts.append(pooled.cpu().numpy())
            candidate_parts.append(candidate.cpu().numpy())
    return (
        np.concatenate(pooled_parts).astype(np.float64),
        np.concatenate(candidate_parts).astype(np.float64),
    )
