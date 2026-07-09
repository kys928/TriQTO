"""Core records and helpers for deterministic TriQTO circuit distortions."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class DistortedCircuit:
    """A clean circuit, distorted copy, and JSON-friendly distortion report."""

    clean_circuit: QuantumCircuit
    distorted_circuit: QuantumCircuit
    distortion_type: str
    distortion_family: str
    strength: float | None
    affected_qubits: list[int]
    affected_gates: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_circuit(circuit_or_generated: Any) -> QuantumCircuit:
    """Extract a Qiskit circuit from a QuantumCircuit or GeneratedCircuit-like object."""
    if isinstance(circuit_or_generated, QuantumCircuit):
        return circuit_or_generated
    circuit = getattr(circuit_or_generated, "circuit", None)
    if isinstance(circuit, QuantumCircuit):
        return circuit
    raise TypeError("Expected a qiskit.QuantumCircuit or TriQTO GeneratedCircuit with a .circuit field.")


def copy_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return an independent copy of a Qiskit circuit."""
    return circuit.copy()


def validate_finite_strength(strength: float, *, name: str = "strength") -> float:
    """Validate and return a finite floating-point strength value."""
    value = float(strength)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def validate_qubits(n_qubits: int, qubits: Sequence[int] | None) -> list[int]:
    """Validate selected qubits, defaulting to all qubits."""
    selected = list(range(n_qubits)) if qubits is None else [int(q) for q in qubits]
    seen: set[int] = set()
    for qubit in selected:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(f"Qubit index {qubit} is out of range for {n_qubits} qubits.")
        if qubit in seen:
            raise ValueError(f"Duplicate qubit index {qubit} is not allowed.")
        seen.add(qubit)
    return selected


def validate_probability(probability: float) -> float:
    """Validate a probability in the closed interval [0, 1]."""
    value = validate_finite_strength(probability, name="probability")
    if value < 0.0 or value > 1.0:
        raise ValueError("probability must be between 0 and 1 inclusive.")
    return value


def has_measurements(circuit: QuantumCircuit) -> bool:
    """Return whether a circuit contains measurement instructions."""
    return any(inst.operation.name == "measure" for inst in circuit.data)


def append_metadata_summary(
    metadata: dict[str, Any] | None,
    *,
    clean_circuit: QuantumCircuit,
    distorted_circuit: QuantumCircuit,
    distortion_type: str,
    distortion_family: str,
    strength: float | None,
    affected_qubits: Sequence[int],
    affected_gates: Sequence[str],
    ideal_only_distortion_model: bool = True,
) -> dict[str, Any]:
    """Merge caller metadata with a standard JSON-friendly distortion summary."""
    original_depth = clean_circuit.depth()
    distorted_depth = distorted_circuit.depth()
    summary: dict[str, Any] = {
        "distortion_type": distortion_type,
        "distortion_family": distortion_family,
        "strength": strength,
        "affected_qubits": list(affected_qubits),
        "affected_gates": list(affected_gates),
        "n_qubits": clean_circuit.num_qubits,
        "n_clbits": clean_circuit.num_clbits,
        "original_depth": original_depth,
        "distorted_depth": distorted_depth,
        "depth_delta": distorted_depth - original_depth,
        "has_measurements": has_measurements(clean_circuit),
        "ideal_only_distortion_model": ideal_only_distortion_model,
    }
    if metadata:
        summary.update(metadata)
    return summary


def make_distorted_circuit(
    circuit_or_generated: Any,
    *,
    distorted_circuit: QuantumCircuit,
    distortion_type: str,
    distortion_family: str,
    strength: float | None,
    affected_qubits: Sequence[int],
    affected_gates: Sequence[str],
    metadata: dict[str, Any] | None = None,
    ideal_only_distortion_model: bool = True,
) -> DistortedCircuit:
    """Create a structured distortion result without mutating the input circuit."""
    clean = copy_circuit(extract_circuit(circuit_or_generated))
    report = append_metadata_summary(
        metadata,
        clean_circuit=clean,
        distorted_circuit=distorted_circuit,
        distortion_type=distortion_type,
        distortion_family=distortion_family,
        strength=strength,
        affected_qubits=affected_qubits,
        affected_gates=affected_gates,
        ideal_only_distortion_model=ideal_only_distortion_model,
    )
    return DistortedCircuit(
        clean_circuit=clean,
        distorted_circuit=distorted_circuit,
        distortion_type=distortion_type,
        distortion_family=distortion_family,
        strength=strength,
        affected_qubits=list(affected_qubits),
        affected_gates=list(affected_gates),
        metadata=report,
    )
