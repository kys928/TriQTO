"""Public topology audit metrics derived from persistence diagrams."""
from __future__ import annotations

from typing import Any

import numpy as np

from triqto.topology.alignment import bottleneck_distance, wasserstein_distance_1
from triqto.topology.features import diagram_statistics, persistence_entropy


def summarize_persistence_diagram(
    diagram: np.ndarray,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    """Return finite/essential counts, persistence mass, entropy, and top lifetimes."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise TypeError("top_k must be a positive integer and not bool")
    summary = diagram_statistics(diagram, top_k)
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in summary.items()
    }


def persistence_diagram_distances(
    diagram_a: np.ndarray,
    diagram_b: np.ndarray,
) -> dict[str, float]:
    """Return bottleneck and 1-Wasserstein distances over finite diagram points."""
    return {
        "bottleneck": bottleneck_distance(diagram_a, diagram_b),
        "wasserstein_1": wasserstein_distance_1(diagram_a, diagram_b),
    }


__all__ = [
    "persistence_diagram_distances",
    "persistence_entropy",
    "summarize_persistence_diagram",
]
