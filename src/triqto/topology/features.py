"""Feature extraction from persistence diagrams and Betti curves."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import TopologyAuditConfig
from .models import PersistenceSummary
from .persistent_homology import betti_curve, validate_persistence_diagram


def _unicode_array(values: list[str]) -> np.ndarray:
    width = max([1, *[len(value) for value in values]])
    return np.asarray(values, dtype=f"<U{width}")


def finite_lifetimes(diagram: np.ndarray) -> np.ndarray:
    validate_persistence_diagram(diagram, 0, "persistence diagram")
    finite = np.isfinite(diagram[:, 1])
    lifetimes = diagram[finite, 1] - diagram[finite, 0]
    if np.any(lifetimes < 0.0) or not np.isfinite(lifetimes).all():
        raise ValueError("Finite persistence lifetimes must be finite and nonnegative")
    return np.asarray(lifetimes, dtype=np.float64)


def persistence_entropy(lifetimes: np.ndarray) -> float:
    if not isinstance(lifetimes, np.ndarray) or lifetimes.dtype != np.float64:
        raise TypeError("lifetimes must be a float64 NumPy array")
    if lifetimes.ndim != 1 or not np.isfinite(lifetimes).all():
        raise ValueError("lifetimes must be one-dimensional and finite")
    if np.any(lifetimes < 0.0):
        raise ValueError("lifetimes must be nonnegative")
    total = float(np.sum(lifetimes))
    if total <= 0.0:
        return 0.0
    positive = lifetimes[lifetimes > 0.0] / total
    return float(-np.sum(positive * np.log(positive))) if positive.size else 0.0


def diagram_statistics(
    diagram: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    lifetimes = finite_lifetimes(diagram)
    ordered = np.sort(lifetimes)[::-1]
    top = np.zeros(top_k, dtype=np.float64)
    top[: min(top_k, ordered.size)] = ordered[:top_k]
    return {
        "finite_count": int(lifetimes.size),
        "essential_count": int(np.count_nonzero(np.isposinf(diagram[:, 1]))),
        "total_persistence": float(np.sum(lifetimes)),
        "persistence_entropy": persistence_entropy(lifetimes),
        "max_lifetime": float(np.max(lifetimes)) if lifetimes.size else 0.0,
        "mean_lifetime": float(np.mean(lifetimes)) if lifetimes.size else 0.0,
        "top_lifetimes": top,
    }


def _audit_scores(
    diagrams: dict[int, np.ndarray],
    point_count: int,
    max_filtration: float,
) -> dict[str, float]:
    h0 = finite_lifetimes(diagrams.get(0, np.empty((0, 2), dtype=np.float64)))
    h1 = finite_lifetimes(diagrams.get(1, np.empty((0, 2), dtype=np.float64)))
    if h0.size:
        normalized_mean_h0 = min(1.0, max(0.0, float(np.mean(h0)) / max_filtration))
        collapse_score = 1.0 - normalized_mean_h0
        bridge_score = min(1.0, max(0.0, float(np.max(h0)) / max_filtration))
    else:
        collapse_score = 0.0
        bridge_score = 0.0
    h1_total = float(np.sum(h1))
    loop_score = 1.0 - math.exp(-h1_total / max(1, point_count))
    return {
        "collapse_score": float(collapse_score),
        "loop_score": float(loop_score),
        "late_merge_bridge_score": float(bridge_score),
    }


def build_persistence_summary(
    *,
    manifold: str,
    diagrams: dict[int, np.ndarray],
    filtration_grid: np.ndarray,
    point_count: int,
    config: TopologyAuditConfig,
    metadata: dict[str, Any] | None = None,
) -> PersistenceSummary:
    """Convert diagrams into deterministic model-ready features and Betti curves."""
    if manifold not in {"parameter", "hilbert", "born"}:
        raise ValueError(f"Unsupported manifold {manifold!r}")
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count <= 0:
        raise TypeError("point_count must be a positive integer")
    names: list[str] = []
    values: list[float] = []
    curves: dict[int, np.ndarray] = {}
    per_dimension: dict[str, Any] = {}
    for dimension in config.homology_dimensions:
        diagram = diagrams.get(dimension)
        if diagram is None:
            raise ValueError(f"Missing H{dimension} diagram for {manifold}")
        validate_persistence_diagram(diagram, dimension, f"{manifold} H{dimension}")
        stats = diagram_statistics(diagram, config.top_k_lifetimes)
        per_dimension[f"h{dimension}"] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in stats.items()
        }
        scalar_fields = (
            "finite_count",
            "essential_count",
            "total_persistence",
            "persistence_entropy",
            "max_lifetime",
            "mean_lifetime",
        )
        for field in scalar_fields:
            names.append(f"h{dimension}_{field}")
            values.append(float(stats[field]))
        for index, lifetime in enumerate(stats["top_lifetimes"], start=1):
            names.append(f"h{dimension}_top_lifetime_{index}")
            values.append(float(lifetime))
        curve = betti_curve(diagram, filtration_grid)
        curves[dimension] = curve
        for index, value in enumerate(curve):
            names.append(f"h{dimension}_betti_grid_{index}")
            values.append(float(value))

    scores = _audit_scores(diagrams, point_count, config.max_filtration)
    for name in ("collapse_score", "loop_score", "late_merge_bridge_score"):
        names.append(name)
        values.append(scores[name])
    feature_values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("Persistence feature vector contains non-finite values")
    return PersistenceSummary(
        manifold=manifold,
        diagrams={dimension: diagram.copy() for dimension, diagram in diagrams.items()},
        betti_curves=curves,
        feature_names=_unicode_array(names),
        feature_values=feature_values,
        metadata={
            **dict(metadata or {}),
            "point_count": point_count,
            "homology_dimensions": list(config.homology_dimensions),
            "top_k_lifetimes": config.top_k_lifetimes,
            "betti_grid_size": config.betti_grid_size,
            "audit_scores": scores,
            "per_dimension": per_dimension,
            "bridge_score_interpretation": (
                "largest normalized finite H0 merge scale; it is a late-merge audit "
                "heuristic, not proof of an unwanted bridge"
            ),
        },
    )


__all__ = [
    "build_persistence_summary",
    "diagram_statistics",
    "finite_lifetimes",
    "persistence_entropy",
]
