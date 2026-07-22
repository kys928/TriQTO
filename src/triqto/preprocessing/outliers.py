"""Multi-view robust outlier tagging; no automatic deletion."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from .config import OutlierConfig
from .records import OutlierRecord, ProcessedSample


def _vectorize_mapping(mapping: Mapping[str, Any], keys: list[str]) -> list[float]:
    vector: list[float] = []
    for key in keys:
        value = mapping.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        vector.append(numeric if math.isfinite(numeric) else 0.0)
    return vector


def _view_matrices(samples: list[ProcessedSample]) -> dict[str, tuple[list[str], np.ndarray]]:
    accepted = [sample for sample in samples if sample.accepted]
    ids = [sample.sample_id for sample in accepted]
    if not accepted:
        return {}
    parameter_keys = sorted({key for sample in accepted for key in sample.parameter_bindings_canonical})
    matrices: dict[str, tuple[list[str], np.ndarray]] = {}
    if parameter_keys:
        matrices["parameter"] = (
            ids,
            np.asarray([
                _vectorize_mapping(sample.parameter_bindings_canonical, parameter_keys)
                for sample in accepted
            ], dtype=float),
        )
    graph_keys = [
        "node_count", "one_qubit_event_count", "two_qubit_event_count",
        "measurement_event_count", "connected_components",
    ]
    matrices["graph"] = (
        ids,
        np.asarray([_vectorize_mapping(sample.graph_features, graph_keys) for sample in accepted], dtype=float),
    )
    hilbert_keys = ["infidelity", "fubini_study", "pure_trace_distance"]
    matrices["hilbert"] = (
        ids,
        np.asarray([_vectorize_mapping(sample.effect_components, hilbert_keys) for sample in accepted], dtype=float),
    )
    born_keys = ["hellinger", "jensen_shannon_distance", "total_variation", "fisher_rao"]
    matrices["born"] = (
        ids,
        np.asarray([_vectorize_mapping(sample.effect_components, born_keys) for sample in accepted], dtype=float),
    )
    metadata_keys = ["shot_count", "distortion_strength", "optimization_level"]
    matrices["metadata"] = (
        ids,
        np.asarray([
            _vectorize_mapping({**sample.provenance, "shot_count": sample.shot_count or 0}, metadata_keys)
            for sample in accepted
        ], dtype=float),
    )
    return matrices


def _robust_standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    scale = np.where(mad > 0.0, 1.4826 * mad, 1.0)
    return (matrix - median) / scale, median, scale


def detect_outliers(
    samples: list[ProcessedSample],
    config: OutlierConfig,
    *,
    fit_sample_ids: set[str] | None = None,
) -> list[OutlierRecord]:
    if not config.enabled:
        return []
    records: list[OutlierRecord] = []
    for view, (sample_ids, matrix) in _view_matrices(samples).items():
        if matrix.size == 0:
            continue
        fit_indices = [
            index for index, sample_id in enumerate(sample_ids)
            if fit_sample_ids is None or sample_id in fit_sample_ids
        ]
        if not fit_indices:
            continue
        fit = matrix[fit_indices]
        _, median, scale = _robust_standardize(fit)
        all_standardized = (matrix - median) / scale
        if "mad" in config.methods:
            scores = np.max(np.abs(all_standardized), axis=1)
            for sample_id, score in zip(sample_ids, scores):
                records.append(OutlierRecord(
                    sample_id=sample_id, view=view, method="mad", score=float(score),
                    threshold=float(config.mad_threshold), is_outlier=bool(score > config.mad_threshold),
                    interpretation="rare_valid_or_anomalous_requires_review" if score > config.mad_threshold else "within_robust_range",
                ))
        if "iqr" in config.methods:
            q1 = np.quantile(fit, 0.25, axis=0)
            q3 = np.quantile(fit, 0.75, axis=0)
            iqr = q3 - q1
            low = q1 - config.iqr_multiplier * iqr
            high = q3 + config.iqr_multiplier * iqr
            outside = np.maximum(low - matrix, 0.0) + np.maximum(matrix - high, 0.0)
            denominator = np.where(iqr > 0.0, iqr, 1.0)
            scores = np.max(outside / denominator, axis=1)
            for sample_id, score in zip(sample_ids, scores):
                records.append(OutlierRecord(
                    sample_id=sample_id, view=view, method="iqr", score=float(score), threshold=0.0,
                    is_outlier=bool(score > 0.0),
                    interpretation="outside_extreme_iqr_fence_requires_review" if score > 0.0 else "inside_extreme_iqr_fence",
                ))
        if "nearest_neighbor" in config.methods and len(fit_indices) >= 2:
            fit_standardized = (fit - median) / scale
            fit_distances = cdist(fit_standardized, fit_standardized)
            np.fill_diagonal(fit_distances, np.inf)
            threshold = float(np.quantile(np.min(fit_distances, axis=1), config.nearest_neighbor_quantile))
            distances = cdist(all_standardized, fit_standardized)
            fit_id_to_columns: dict[str, list[int]] = {}
            for column, source_index in enumerate(fit_indices):
                fit_id_to_columns.setdefault(sample_ids[source_index], []).append(column)
            for row_index, sample_id in enumerate(sample_ids):
                for column in fit_id_to_columns.get(sample_id, []):
                    distances[row_index, column] = np.inf
            nearest = np.min(distances, axis=1)
            for sample_id, score in zip(sample_ids, nearest):
                records.append(OutlierRecord(
                    sample_id=sample_id, view=view, method="nearest_neighbor", score=float(score),
                    threshold=threshold, is_outlier=bool(score > threshold),
                    interpretation="isolated_in_view_requires_review" if score > threshold else "has_nearby_fit_sample",
                ))
    return sorted(records, key=lambda item: (item.view, item.method, item.sample_id))
