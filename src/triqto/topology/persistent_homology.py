"""Vietoris-Rips persistent homology over validated precomputed distances."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import TopologyAuditConfig
from .distances import validate_distance_matrix


def validate_persistence_diagram(
    diagram: np.ndarray,
    dimension: int,
    name: str,
) -> None:
    if not isinstance(diagram, np.ndarray) or diagram.dtype != np.float64:
        raise TypeError(f"{name} must be a float64 NumPy array")
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError(f"{name} must have shape (features, 2)")
    if not np.isfinite(diagram[:, 0]).all():
        raise ValueError(f"{name} births must be finite")
    deaths = diagram[:, 1]
    if np.isnan(deaths).any() or np.isneginf(deaths).any():
        raise ValueError(f"{name} deaths cannot contain NaN or negative infinity")
    if np.any(diagram[:, 0] < 0.0):
        raise ValueError(f"{name} births must be nonnegative")
    finite = np.isfinite(deaths)
    if np.any(deaths[finite] < diagram[finite, 0]):
        raise ValueError(f"{name} finite deaths must not precede births")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
        raise TypeError("homology dimension must be a nonnegative integer")


def _sort_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    rows = [(float(birth), float(death)) for birth, death in diagram.tolist()]
    rows.sort(
        key=lambda item: (
            item[0],
            math.inf if math.isinf(item[1]) else item[1],
        )
    )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def compute_persistence_diagrams(
    distance_matrix: np.ndarray,
    config: TopologyAuditConfig,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Compute deterministic H0/H1 and optional H2 diagrams with ripser."""
    validate_distance_matrix(distance_matrix, "distance_matrix")
    if distance_matrix.shape[0] < config.min_points:
        raise ValueError(
            f"Persistent homology requires at least {config.min_points} points"
        )
    try:
        from ripser import ripser
    except Exception as exc:  # pragma: no cover - dependency environment
        raise RuntimeError("ripser is required for Phase 11 persistent homology") from exc

    max_dimension = max(config.homology_dimensions)
    result = ripser(
        distance_matrix,
        distance_matrix=True,
        maxdim=max_dimension,
        thresh=config.max_filtration,
    )
    raw_diagrams = result.get("dgms")
    if not isinstance(raw_diagrams, list):
        raise RuntimeError("ripser did not return a diagram list")
    diagrams: dict[int, np.ndarray] = {}
    for dimension in config.homology_dimensions:
        if dimension < len(raw_diagrams):
            diagram = np.asarray(raw_diagrams[dimension], dtype=np.float64)
        else:
            diagram = np.empty((0, 2), dtype=np.float64)
        diagram = _sort_diagram(diagram)
        validate_persistence_diagram(
            diagram,
            dimension,
            f"H{dimension} persistence diagram",
        )
        diagrams[dimension] = diagram
    return diagrams, {
        "engine": "ripser",
        "filtration": "vietoris_rips",
        "distance_matrix_precomputed": True,
        "max_homology_dimension": max_dimension,
        "max_filtration": config.max_filtration,
        "point_count": int(distance_matrix.shape[0]),
        "cocycle_output_used": False,
    }


def betti_curve(
    diagram: np.ndarray,
    filtration_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate one persistence diagram's Betti number on a fixed grid."""
    validate_persistence_diagram(diagram, 0, "diagram")
    if not isinstance(filtration_grid, np.ndarray):
        raise TypeError("filtration_grid must be a NumPy array")
    if filtration_grid.dtype != np.float64 or filtration_grid.ndim != 1:
        raise TypeError("filtration_grid must be one-dimensional float64")
    if filtration_grid.size < 2:
        raise ValueError("filtration_grid must contain at least two values")
    if not np.isfinite(filtration_grid).all() or np.any(filtration_grid < 0.0):
        raise ValueError("filtration_grid must be finite and nonnegative")
    if np.any(np.diff(filtration_grid) < 0.0):
        raise ValueError("filtration_grid must be sorted")
    values = np.zeros(filtration_grid.shape, dtype=np.float64)
    for index, epsilon in enumerate(filtration_grid):
        alive = (diagram[:, 0] <= epsilon) & (diagram[:, 1] > epsilon)
        values[index] = float(np.count_nonzero(alive))
    return values


def make_filtration_grid(config: TopologyAuditConfig) -> np.ndarray:
    return np.linspace(
        0.0,
        config.max_filtration,
        config.betti_grid_size,
        dtype=np.float64,
    )


__all__ = [
    "betti_curve",
    "compute_persistence_diagrams",
    "make_filtration_grid",
    "validate_persistence_diagram",
]
