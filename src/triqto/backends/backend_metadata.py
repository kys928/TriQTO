"""Stable backend evidence snapshots with availability masks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from triqto.core.ids import make_deterministic_id


@dataclass(frozen=True, slots=True)
class BackendSnapshot:
    backend_id: str
    backend_name: str
    backend_source: str
    backend_class: str
    n_qubits: int
    basis_gates: tuple[str, ...]
    coupling_map: tuple[tuple[int, int], ...]
    calibration_timestamp: str | None
    feature_values: dict[str, Any] = field(default_factory=dict)
    feature_available: dict[str, bool] = field(default_factory=dict)
    missing_reasons: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend_class not in {"fake", "simulator", "physical"}:
            raise ValueError("backend_class must be fake, simulator, or physical")
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if not self.basis_gates:
            raise ValueError("basis_gates must be non-empty")
        for left, right in self.coupling_map:
            if left < 0 or right < 0 or left >= self.n_qubits or right >= self.n_qubits:
                raise ValueError("coupling_map contains out-of-range qubit")
        for name, available in self.feature_available.items():
            if not available and name not in self.missing_reasons:
                raise ValueError(f"missing feature {name!r} requires missing reason")


def backend_snapshot_id(payload: dict[str, Any]) -> str:
    return make_deterministic_id("backend", {"schema": "triqto.backend.snapshot.v1", **payload})


def summarize_coupling_map(n_qubits: int, coupling_map: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    neighbors = {index: set() for index in range(n_qubits)}
    for left, right in coupling_map:
        neighbors[left].add(right)
    degrees = [len(neighbors[index]) for index in range(n_qubits)]
    # Directed unweighted diameter over reachable pairs; None if disconnected.
    diameter = 0
    for start in range(n_qubits):
        seen = {start}
        frontier = {start}
        distance = 0
        while frontier:
            nxt = set()
            for node in frontier:
                nxt |= neighbors[node] - seen
            if nxt:
                distance += 1
                diameter = max(diameter, distance)
            seen |= nxt
            frontier = nxt
        if len(seen) != n_qubits:
            diameter = None
            break
    return {
        "directed_edge_count": len(coupling_map),
        "degree_min": min(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_mean": sum(degrees) / len(degrees) if degrees else 0.0,
        "directed_diameter": diameter,
    }


def describe_contract() -> str:
    return "Backend snapshots use explicit availability masks and missing reasons; missing is never encoded as zero."


__all__ = ["BackendSnapshot", "backend_snapshot_id", "summarize_coupling_map"]
