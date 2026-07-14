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
    distribution_index: Tensor | None = None,
) -> list[dict[str, float]]:
    """Return per-setting-complete distances averaged within each graph."""
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
    if distribution_index is None:
        distribution_all = batch_all
    else:
        if (
            distribution_index.shape != target.shape
            or distribution_index.dtype != torch.long
        ):
            raise ValueError(
                "distribution_index must be int64 with the distribution shape"
            )
        distribution_all = _numpy(distribution_index).astype(np.int64, copy=False)
    rows: list[dict[str, float]] = []
    for graph_index in range(graph_count):
        selected = (batch_all == graph_index) & mask_all
        if not selected.any():
            rows.append({})
            continue
        setting_rows: list[dict[str, float]] = []
        for setting in sorted(set(distribution_all[selected].tolist())):
            setting_selected = selected & (distribution_all == setting)
            owners = set(batch_all[setting_selected].tolist())
            if owners != {graph_index}:
                raise ValueError("one measurement distribution must not span graphs")
            p = np.clip(p_all[setting_selected], 0.0, None)
            q = np.clip(q_all[setting_selected], 0.0, None)
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
            setting_rows.append(
                {
                    "born_kl": kl,
                    "born_hellinger": hellinger,
                    "born_js": js,
                    "born_total_variation": float(0.5 * np.abs(p - q).sum()),
                    "born_probability_mae": float(np.abs(p - q).mean()),
                }
            )
        rows.append(mean_numeric_maps(setting_rows))
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


def uncertainty_variance_calibration(
    variances: Iterable[float],
    realized_errors: Iterable[float],
    *,
    bins: int,
) -> dict[str, float]:
    """Compare predicted per-example variance with realized task loss."""
    variance = np.asarray(list(variances), dtype=np.float64)
    error = np.asarray(list(realized_errors), dtype=np.float64)
    if variance.shape != error.shape:
        raise ValueError("uncertainty variance/error shapes differ")
    if variance.size == 0:
        return {}
    if (
        not np.isfinite(variance).all()
        or not np.isfinite(error).all()
        or np.any(variance <= 0.0)
        or np.any(error < 0.0)
    ):
        raise ValueError(
            "uncertainty variances must be positive and errors nonnegative"
        )
    order = np.argsort(variance, kind="stable")
    partitions = np.array_split(order, min(bins, variance.size))
    absolute_gap = 0.0
    maximum_gap = 0.0
    for indices in partitions:
        gap = abs(float(variance[indices].mean() - error[indices].mean()))
        absolute_gap += len(indices) / variance.size * gap
        maximum_gap = max(maximum_gap, gap)
    correlation = (
        float(np.corrcoef(variance, error)[0, 1])
        if variance.size > 1
        and float(np.std(variance)) > 0.0
        and float(np.std(error)) > 0.0
        else 0.0
    )
    return {
        "variance_calibration_error": float(absolute_gap),
        "variance_calibration_max_error": float(maximum_gap),
        "variance_error_mae": float(np.mean(np.abs(variance - error))),
        "variance_error_correlation": correlation,
        "mean_predicted_variance": float(variance.mean()),
        "mean_realized_error": float(error.mean()),
        "occupied_variance_bins": float(len(partitions)),
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
                "distortion_type",
                item.distortion_type or "unknown",
            ),
            (
                item.task,
                item.ablation,
                "backend_id",
                item.backend_id or "unknown",
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
        calibration: dict[str, float] = {}
        for softmax_task in ("diagnosis", "action_ranking"):
            confidence_key = f"softmax_{softmax_task}_confidence"
            correct_key = f"softmax_{softmax_task}_correct"
            task_rows = [
                row
                for row in calibration_rows
                if confidence_key in row and correct_key in row
            ]
            if not task_rows:
                continue
            descriptive = expected_calibration_error(
                [row[confidence_key] for row in task_rows],
                [row[correct_key] for row in task_rows],
                bins=calibration_bins,
            )
            calibration.update(
                {
                    f"softmax_{softmax_task}_{name}": metric
                    for name, metric in descriptive.items()
                }
            )
        for uncertainty_task in (
            "diagnosis",
            "action_ranking",
            "born_prediction",
            "hilbert_to_born",
        ):
            variance_key = f"uncertainty_{uncertainty_task}_variance"
            error_key = f"uncertainty_{uncertainty_task}_realized_error"
            task_rows = [
                row
                for row in calibration_rows
                if variance_key in row and error_key in row
            ]
            if not task_rows:
                continue
            direct = uncertainty_variance_calibration(
                [row[variance_key] for row in task_rows],
                [row[error_key] for row in task_rows],
                bins=calibration_bins,
            )
            calibration.update(
                {
                    f"uncertainty_{uncertainty_task}_{name}": metric
                    for name, metric in direct.items()
                }
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
    "uncertainty_variance_calibration",
]
