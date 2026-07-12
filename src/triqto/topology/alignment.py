"""Cross-manifold persistence-diagram alignment features."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TopologyAuditConfig
from .constants import MANIFOLD_ORDER
from .models import PersistenceSummary
from .persistent_homology import validate_persistence_diagram


def _unicode_array(values: list[str]) -> np.ndarray:
    width = max([1, *[len(value) for value in values]])
    return np.asarray(values, dtype=f"<U{width}")


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    validate_persistence_diagram(diagram, 0, "persistence diagram")
    finite = np.isfinite(diagram[:, 1])
    return diagram[finite].astype(np.float64, copy=True)


def _distance_to_diagonal(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.asarray([], dtype=np.float64)
    return 0.5 * (diagram[:, 1] - diagram[:, 0])


def wasserstein_distance_1(diagram_a: np.ndarray, diagram_b: np.ndarray) -> float:
    """Return finite-diagram 1-Wasserstein distance with L-infinity ground metric."""
    a = finite_diagram(diagram_a)
    b = finite_diagram(diagram_b)
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(np.sum(_distance_to_diagonal(b)))
    if m == 0:
        return float(np.sum(_distance_to_diagonal(a)))

    size = n + m
    maximum = max(
        1.0,
        float(np.max(a[:, 1])) if a.size else 0.0,
        float(np.max(b[:, 1])) if b.size else 0.0,
    )
    large = maximum * (size + 1) * 10.0
    cost = np.full((size, size), large, dtype=np.float64)
    pairwise = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    cost[:n, :m] = pairwise
    diagonal_a = _distance_to_diagonal(a)
    diagonal_b = _distance_to_diagonal(b)
    for index, value in enumerate(diagonal_a):
        cost[index, m + index] = value
    for index, value in enumerate(diagonal_b):
        cost[n + index, index] = value
    cost[n:, m:] = 0.0
    rows, columns = linear_sum_assignment(cost)
    value = float(cost[rows, columns].sum())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Wasserstein diagram distance must be finite and nonnegative")
    return value


def bottleneck_distance(diagram_a: np.ndarray, diagram_b: np.ndarray) -> float:
    """Return Gudhi bottleneck distance over finite diagram points."""
    a = finite_diagram(diagram_a)
    b = finite_diagram(diagram_b)
    if a.size == 0 and b.size == 0:
        return 0.0
    try:
        import gudhi
    except Exception as exc:  # pragma: no cover - dependency environment
        raise RuntimeError("gudhi is required for Phase 11 bottleneck distance") from exc
    value = float(gudhi.bottleneck_distance(a, b, e=0.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Bottleneck distance must be finite and nonnegative")
    return value


def build_alignment_features(
    persistence: dict[str, PersistenceSummary],
    config: TopologyAuditConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compare diagrams for every available manifold pair and homology dimension."""
    available = [name for name in MANIFOLD_ORDER if name in persistence]
    names: list[str] = []
    values: list[float] = []
    pair_metadata: dict[str, Any] = {}
    bottleneck_values: list[float] = []
    wasserstein_values: list[float] = []
    for left_index, left in enumerate(available):
        for right in available[left_index + 1 :]:
            pair_name = f"{left}_to_{right}"
            pair_metadata[pair_name] = {}
            for dimension in config.homology_dimensions:
                left_diagram = persistence[left].diagrams[dimension]
                right_diagram = persistence[right].diagrams[dimension]
                bottleneck = bottleneck_distance(left_diagram, right_diagram)
                wasserstein = wasserstein_distance_1(left_diagram, right_diagram)
                essential_gap = abs(
                    int(np.count_nonzero(np.isposinf(left_diagram[:, 1])))
                    - int(np.count_nonzero(np.isposinf(right_diagram[:, 1])))
                )
                similarity = math.exp(-bottleneck)
                entries = {
                    "bottleneck": bottleneck,
                    "wasserstein_1": wasserstein,
                    "essential_count_gap": float(essential_gap),
                    "alignment_similarity": similarity,
                }
                pair_metadata[pair_name][f"h{dimension}"] = entries
                for metric_name, metric_value in entries.items():
                    names.append(f"{pair_name}_h{dimension}_{metric_name}")
                    values.append(float(metric_value))
                bottleneck_values.append(bottleneck)
                wasserstein_values.append(wasserstein)

    aggregate_bottleneck = (
        float(np.mean(bottleneck_values)) if bottleneck_values else 0.0
    )
    aggregate_wasserstein = (
        float(np.mean(wasserstein_values)) if wasserstein_values else 0.0
    )
    topology_preservation_score = math.exp(-aggregate_bottleneck)
    names.extend(
        (
            "aggregate_mean_bottleneck",
            "aggregate_mean_wasserstein_1",
            "topology_preservation_score",
        )
    )
    values.extend(
        (
            aggregate_bottleneck,
            aggregate_wasserstein,
            topology_preservation_score,
        )
    )
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Alignment feature vector contains non-finite values")
    return _unicode_array(names), array, {
        "available_manifolds": available,
        "diagram_policy": "finite persistence points compared; essential counts audited separately",
        "bottleneck_ground_metric": "linf",
        "wasserstein_order": 1,
        "wasserstein_ground_metric": "linf",
        "pair_features": pair_metadata,
        "aggregate_mean_bottleneck": aggregate_bottleneck,
        "aggregate_mean_wasserstein_1": aggregate_wasserstein,
        "topology_preservation_score": topology_preservation_score,
        "alignment_is_diagnostic_not_training_loss": True,
    }


__all__ = [
    "bottleneck_distance",
    "build_alignment_features",
    "finite_diagram",
    "wasserstein_distance_1",
]
