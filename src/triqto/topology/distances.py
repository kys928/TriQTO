"""Distance matrices for parameter, pure-state Hilbert, and Born point clouds."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import TopologyAuditConfig
from .constants import DISTANCE_ATOL


def validate_distance_matrix(matrix: np.ndarray, name: str) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if matrix.dtype != np.float64 or matrix.ndim != 2:
        raise TypeError(f"{name} must be a two-dimensional float64 array")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(matrix < -DISTANCE_ATOL):
        raise ValueError(f"{name} must be nonnegative")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=DISTANCE_ATOL):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=DISTANCE_ATOL):
        raise ValueError(f"{name} diagonal must be zero")


def normalize_distance_matrix(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    validate_distance_matrix(matrix, "distance_matrix")
    if matrix.size == 0:
        return matrix.copy(), 0.0
    scale = float(np.max(matrix))
    if scale <= DISTANCE_ATOL:
        return np.zeros_like(matrix, dtype=np.float64), 0.0
    normalized = (matrix / scale).astype(np.float64, copy=False)
    np.fill_diagonal(normalized, 0.0)
    validate_distance_matrix(normalized, "normalized_distance_matrix")
    return normalized, scale


def circular_parameter_distance_matrix(
    coordinates: np.ndarray,
    coordinate_mask: np.ndarray,
) -> np.ndarray:
    """Return root-mean-square wrapped angular distance over shared coordinates.

    Missing cohort parameters are not imputed. A pair with no shared raw coordinates gets raw
    distance zero and is separated only by the downstream pullback components.
    """
    if not isinstance(coordinates, np.ndarray) or coordinates.dtype != np.float64:
        raise TypeError("parameter coordinates must be float64")
    if coordinates.ndim != 2:
        raise ValueError("parameter coordinates must be two-dimensional")
    if not isinstance(coordinate_mask, np.ndarray) or coordinate_mask.dtype != np.bool_:
        raise TypeError("parameter coordinate mask must be bool")
    if coordinate_mask.shape != coordinates.shape:
        raise ValueError("parameter coordinate mask shape mismatch")
    if not np.isfinite(coordinates).all():
        raise ValueError("parameter coordinates must be finite")
    count = coordinates.shape[0]
    result = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            shared = coordinate_mask[i] & coordinate_mask[j]
            if not np.any(shared):
                distance = 0.0
            else:
                delta = coordinates[i, shared] - coordinates[j, shared]
                wrapped = np.arctan2(np.sin(delta), np.cos(delta))
                distance = float(np.sqrt(np.mean((wrapped / math.pi) ** 2)))
            result[i, j] = result[j, i] = distance
    validate_distance_matrix(result, "raw_parameter_distance_matrix")
    return result


def fubini_study_distance_matrix(statevectors: np.ndarray) -> np.ndarray:
    """Return projective pure-state distance, invariant to global phase."""
    if not isinstance(statevectors, np.ndarray) or statevectors.dtype != np.complex128:
        raise TypeError("statevectors must be a complex128 array")
    if statevectors.ndim != 2 or statevectors.shape[0] < 1:
        raise ValueError("statevectors must have shape (points, amplitudes)")
    if not np.isfinite(statevectors.real).all() or not np.isfinite(statevectors.imag).all():
        raise ValueError("statevectors must be finite")
    norms = np.linalg.norm(statevectors, axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("statevectors must have finite nonzero norm")
    normalized = statevectors / norms[:, None]
    count = normalized.shape[0]
    result = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            overlap = float(abs(np.vdot(normalized[i], normalized[j])))
            overlap = min(1.0, max(0.0, overlap))
            distance = float(math.acos(overlap) / (math.pi / 2.0))
            result[i, j] = result[j, i] = distance
    validate_distance_matrix(result, "hilbert_distance_matrix")
    return result


def _validate_probability_rows(probabilities: np.ndarray) -> None:
    if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
        raise TypeError("Born coordinates must be float64")
    if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 1:
        raise ValueError("Born coordinates must have shape (points, outcomes)")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("Born coordinates must be finite and nonnegative")
    totals = probabilities.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Every Born coordinate row must sum to one")


def born_distance_matrix(probabilities: np.ndarray, metric: str) -> np.ndarray:
    """Return Hellinger, Jensen-Shannon, or Fisher-Rao distances."""
    _validate_probability_rows(probabilities)
    if metric not in {"hellinger", "jensen_shannon", "fisher_rao"}:
        raise ValueError(f"Unsupported Born distance metric {metric!r}")
    count = probabilities.shape[0]
    result = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        p = probabilities[i]
        for j in range(i + 1, count):
            q = probabilities[j]
            affinity = float(np.sum(np.sqrt(p * q)))
            affinity = min(1.0, max(0.0, affinity))
            if metric == "hellinger":
                distance = math.sqrt(max(0.0, 1.0 - affinity))
            elif metric == "fisher_rao":
                distance = (2.0 * math.acos(affinity)) / math.pi
            else:
                midpoint = 0.5 * (p + q)
                p_mask = p > 0.0
                q_mask = q > 0.0
                left = float(
                    np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask]))
                )
                right = float(
                    np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask]))
                )
                divergence = max(0.0, min(1.0, 0.5 * (left + right)))
                distance = math.sqrt(divergence)
            result[i, j] = result[j, i] = float(distance)
    validate_distance_matrix(result, "born_distance_matrix")
    return result


def induced_parameter_distance_matrix(
    raw_parameter: np.ndarray,
    born: np.ndarray,
    hilbert: np.ndarray | None,
    config: TopologyAuditConfig,
) -> np.ndarray:
    """Return a pullback-style parameter pseudometric dominated by downstream deformation."""
    validate_distance_matrix(raw_parameter, "raw_parameter_distance_matrix")
    validate_distance_matrix(born, "born_distance_matrix")
    if raw_parameter.shape != born.shape:
        raise ValueError("raw parameter and Born distance shapes must match")
    active_hilbert = hilbert is not None and config.include_hilbert
    if active_hilbert:
        assert hilbert is not None
        validate_distance_matrix(hilbert, "hilbert_distance_matrix")
        if hilbert.shape != born.shape:
            raise ValueError("Hilbert and Born distance shapes must match")
    squared = (
        config.raw_parameter_weight * np.square(raw_parameter)
        + config.born_pullback_weight * np.square(born)
    )
    total_weight = config.raw_parameter_weight + config.born_pullback_weight
    if active_hilbert:
        squared = squared + config.hilbert_pullback_weight * np.square(hilbert)
        total_weight += config.hilbert_pullback_weight
    if total_weight <= 0.0:
        raise ValueError("No active induced-parameter distance component")
    result = np.sqrt(squared / total_weight).astype(np.float64, copy=False)
    np.fill_diagonal(result, 0.0)
    validate_distance_matrix(result, "parameter_distance_matrix")
    return result


def compute_manifold_distance_matrices(
    *,
    parameter_coordinates: np.ndarray,
    parameter_coordinate_mask: np.ndarray,
    born_coordinates: np.ndarray,
    statevectors: np.ndarray | None,
    config: TopologyAuditConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_parameter = circular_parameter_distance_matrix(
        parameter_coordinates,
        parameter_coordinate_mask,
    )
    born = born_distance_matrix(born_coordinates, config.born_distance)
    hilbert = (
        fubini_study_distance_matrix(statevectors)
        if statevectors is not None and config.include_hilbert
        else None
    )
    parameter = induced_parameter_distance_matrix(
        raw_parameter,
        born,
        hilbert,
        config,
    )
    matrices = {
        "parameter": parameter,
        "hilbert": (
            hilbert
            if hilbert is not None
            else np.zeros_like(parameter, dtype=np.float64)
        ),
        "born": born,
    }
    scales: dict[str, float] = {}
    if config.normalize_distance_matrices:
        for name in ("parameter", "born"):
            matrices[name], scales[name] = normalize_distance_matrix(matrices[name])
        if hilbert is not None:
            matrices["hilbert"], scales["hilbert"] = normalize_distance_matrix(
                matrices["hilbert"]
            )
        else:
            scales["hilbert"] = 0.0
    else:
        scales = {
            name: float(np.max(matrix)) if matrix.size else 0.0
            for name, matrix in matrices.items()
        }
    return matrices, {
        "raw_parameter_distance_scale": (
            float(np.max(raw_parameter)) if raw_parameter.size else 0.0
        ),
        "normalization_scales": scales,
        "born_distance": config.born_distance,
        "hilbert_metric": "fubini_study_projective_distance",
        "parameter_metric": "downstream_pullback_pseudometric",
        "parameter_component_weights": {
            "raw_periodic_parameter": config.raw_parameter_weight,
            "born": config.born_pullback_weight,
            "hilbert": (
                config.hilbert_pullback_weight if hilbert is not None else 0.0
            ),
        },
    }


__all__ = [
    "born_distance_matrix",
    "circular_parameter_distance_matrix",
    "compute_manifold_distance_matrices",
    "fubini_study_distance_matrix",
    "induced_parameter_distance_matrix",
    "normalize_distance_matrix",
    "validate_distance_matrix",
]
