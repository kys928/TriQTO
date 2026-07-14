"""Transparent held-out metrics, calibration, and aggregation for Phase 15."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from .identities import evaluation_aggregate_id
from .models import EvaluationAggregate, EvaluationItemResult


def _numpy(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def distribution_metrics_by_graph(
    predicted: Tensor,
    target: Tensor,
    row_mask: Tensor,
    outcome_batch: Tensor,
    graph_count: int,
    *,
    epsilon: float,
) -> list[dict[str, float]]:
    """Return complete lower-is-better distances for each active graph."""
    if predicted.shape != target.shape or row_mask.shape != target.shape:
        raise ValueError("distribution tensors must share one shape")
    if outcome_batch.shape != target.shape or outcome_batch.dtype != torch.long:
        raise ValueError("outcome_batch must be int64 with the distribution shape")
    if graph_count <= 0:
        raise ValueError("graph_count must be positive")
    p_all = _numpy(predicted).astype(np.float64, copy=False)
    q_all = _numpy(target).astype(np.float64, copy=False)
    mask_all = _numpy(row_mask).astype(np.bool_, copy=False)
    batch_all = _numpy(outcome_batch).astype(np.int64, copy=False)
    rows: list[dict[str, float]] = []
    for graph_index in range(graph_count):
        selected = (batch_all == graph_index) & mask_all
        if not selected.any():
            rows.append({})
            continue
        p = np.clip(p_all[selected], 0.0, None)
        q = np.clip(q_all[selected], 0.0, None)
        p_sum = float(p.sum())
        q_sum = float(q.sum())
        if p_sum <= 0.0 or q_sum <= 0.0:
            raise ValueError("active distribution rows must have positive mass")
        p = p / p_sum
        q = q / q_sum
        p_safe = np.clip(p, epsilon, None)
        q_safe = np.clip(q, epsilon, None)
        kl = float(
            np.sum(
                np.where(
                    q > 0.0,
                    q * (np.log(q_safe) - np.log(p_safe)),
                    0.0,
                )
            )
        )
        hellinger = float(
            np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        )
        midpoint = 0.5 * (p + q)
        midpoint_safe = np.clip(midpoint, epsilon, None)
        js = float(
            0.5
            * np.sum(
                np.where(
                    p > 0.0,
                    p * (np.log(p_safe) - np.log(midpoint_safe)),
                    0.0,
                )
            )
            + 0.5
            * np.sum(
                np.where(
                    q > 0.0,
                    q * (np.log(q_safe) - np.log(midpoint_safe)),
                    0.0,
                )
            )
        )
        tv = float(0.5 * np.abs(p - q).sum())
        mae = float(np.abs(p - q).mean())
        rows.append(
            {
                "born_kl": kl,
                "born_hellinger": hellinger,
                "born_js": js,
                "born_total_variation": tv,
                "born_probability_mae": mae,
            }
        )
    return rows


def expected_calibration_error(
    confidences: Iterable[float],
    correctness: Iterable[float],
    *,
    bins: int,
) -> dict[str, float]:
    confidence = np.asarray(list(confidences), dtype=np.float64)
    correct = np.asarray(list(correctness), dtype=np.float64)
    if confidence.shape != correct.shape:
        raise ValueError("calibration confidence/correctness shapes differ")
    if confidence.size == 0:
        return {}
    if not np.isfinite(confidence).all() or not np.isfinite(correct).all():
        raise ValueError("calibration values must be finite")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("calibration confidence must be in [0,1]")
    if np.any((correct < 0.0) | (correct > 1.0)):
        raise ValueError("calibration correctness must be in [0,1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    occupied = 0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (
            (confidence >= lower)
            & (
                confidence <= upper
                if index == bins - 1
                else confidence < upper
            )
        )
        count = int(selected.sum())
        if count == 0:
            continue
        occupied += 1
        gap = abs(
            float(confidence[selected].mean())
            - float(correct[selected].mean())
        )
        ece += count / confidence.size * gap
        mce = max(mce, gap)
    return {
        "calibration_ece": float(ece),
        "calibration_mce": float(mce),
        "calibration_brier": float(np.mean((confidence - correct) ** 2)),
        "calibration_mean_confidence": float(confidence.mean()),
        "calibration_empirical_accuracy": float(correct.mean()),
        "calibration_occupied_bins": float(occupied),
    }


def mean_numeric_maps(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Metric {name} is non-finite")
            values[name].append(numeric)
    return {
        name: float(np.mean(items))
        for name, items in sorted(values.items())
        if items
    }


def build_aggregates(
    item_results: list[EvaluationItemResult],
    *,
    evaluation_run_id: str,
    calibration_bins: int,
) -> list[EvaluationAggregate]:
    """Aggregate by overall/task and declared generalization dimensions."""
    groups: dict[
        tuple[str, str, str, str],
        list[EvaluationItemResult],
    ] = defaultdict(list)
    for item in item_results:
        keys = [
            (item.task, item.ablation, "overall", "all"),
            (item.task, item.ablation, "task", item.task),
            (item.task, item.ablation, "family", item.family or "unknown"),
            (item.task, item.ablation, "n_qubits", str(item.n_qubits)),
            (
                item.task,
                item.ablation,
                "distortion_id",
                item.distortion_id or "unknown",
            ),
        ]
        for key in keys:
            groups[key].append(item)

    aggregates: list[EvaluationAggregate] = []
    for (task, ablation, dimension, value), items in sorted(groups.items()):
        calibration_rows = [
            item.calibration
            for item in items
            if item.calibration
        ]
        confidences = [
            row["confidence"]
            for row in calibration_rows
            if "confidence" in row
        ]
        correctness = [
            row["correct"]
            for row in calibration_rows
            if "correct" in row
        ]
        calibration = (
            expected_calibration_error(
                confidences,
                correctness,
                bins=calibration_bins,
            )
            if confidences
            else {}
        )
        aggregates.append(
            EvaluationAggregate(
                evaluation_aggregate_id=evaluation_aggregate_id(
                    evaluation_run_id,
                    task,
                    ablation,
                    dimension,
                    value,
                ),
                evaluation_run_id=evaluation_run_id,
                task=task,
                ablation=ablation,
                group_dimension=dimension,
                group_value=value,
                item_count=len(items),
                metrics=mean_numeric_maps(item.metrics for item in items),
                calibration=calibration,
                metadata={
                    "heldout_split": "test",
                    "aggregation": "unweighted_mean_across_items",
                },
            )
        )
    return aggregates


__all__ = [
    "build_aggregates",
    "distribution_metrics_by_graph",
    "expected_calibration_error",
    "mean_numeric_maps",
]
