"""Core records and helpers for deterministic TriQTO circuit distortions."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from qiskit import ClassicalRegister, QuantumCircuit


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


def has_classical_conditions(circuit: QuantumCircuit) -> bool:
    """Return whether any instruction has a classical condition."""
    return any(getattr(inst.operation, "condition", None) is not None for inst in circuit.data)


def has_nonmeasurement_after_measurement(circuit: QuantumCircuit) -> bool:
    """Return whether any non-measurement operation follows a measurement."""
    seen_measurement = False
    for inst in circuit.data:
        if inst.operation.name == "measure":
            seen_measurement = True
        elif seen_measurement:
            return True
    return False


def _final_measurement_pairs(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    """Return final measurement qubit/classical-bit index pairs from a circuit."""
    return [
        (circuit.find_bit(inst.qubits[0]).index, circuit.find_bit(inst.clbits[0]).index)
        for inst in circuit.data
        if inst.operation.name == "measure"
    ]


def _restore_classical_bits(source: QuantumCircuit, target: QuantumCircuit) -> None:
    """Restore enough classical bits on ``target`` to preserve measurement mapping."""
    if target.num_clbits >= source.num_clbits:
        return
    for creg in source.cregs:
        if target.num_clbits >= source.num_clbits:
            break
        target.add_register(ClassicalRegister(creg.size, creg.name))
    remaining = source.num_clbits - target.num_clbits
    if remaining > 0:
        target.add_register(ClassicalRegister(remaining, "distortion_c"))


def copy_for_unitary_distortion(circuit: QuantumCircuit) -> tuple[QuantumCircuit, Callable[[QuantumCircuit], QuantumCircuit], dict[str, Any]]:
    """Copy a circuit for unitary distortion before safely restoring final measurements.

    Unitary distortions are inserted before final measurements so that the final
    measurements remain removable by ideal statevector simulation. Circuits with
    mid-circuit measurements or classical conditions are rejected because inserting
    unitary drift around measurement/control flow would be ambiguous.
    """
    copied = copy_circuit(circuit)
    if has_classical_conditions(copied):
        raise ValueError("Unitary distortions do not support classical conditions.")
    if has_nonmeasurement_after_measurement(copied):
        raise ValueError("Unitary distortions require measurements to be final; mid-circuit measurements are unsupported.")
    measurement_pairs = _final_measurement_pairs(copied)
    if not measurement_pairs:
        return copied, lambda distorted: distorted, {"final_measurements_removed": False, "final_measurement_count": 0}

    try:
        stripped = copied.remove_final_measurements(inplace=False)
    except TypeError:  # pragma: no cover - compatibility with older Qiskit
        stripped = copied.copy()
        stripped.remove_final_measurements(inplace=True)

    if has_measurements(stripped):
        raise ValueError("Unitary distortions require measurements to be final; mid-circuit measurements are unsupported.")

    def restore_measurements(distorted: QuantumCircuit) -> QuantumCircuit:
        restored = distorted.copy()
        _restore_classical_bits(copied, restored)
        for qubit_index, clbit_index in measurement_pairs:
            restored.measure(qubit_index, clbit_index)
        return restored

    return stripped, restore_measurements, {
        "final_measurements_removed": True,
        "final_measurement_count": len(measurement_pairs),
        "final_measurement_map": [[q, c] for q, c in measurement_pairs],
    }


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
