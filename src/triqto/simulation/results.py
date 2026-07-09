"""In-memory result dataclasses for ideal TriQTO simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IdealStatevectorResult:
    """Result of an ideal statevector simulation."""

    simulation_mode: str
    n_qubits: int
    statevector: Any
    probabilities: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IdealShotResult:
    """Result of deterministic ideal shot sampling from Born probabilities."""

    simulation_mode: str
    n_qubits: int
    shots: int
    counts: dict[str, int]
    probabilities: dict[str, float]
    source_probabilities: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
