"""Structured metadata for generated TriQTO circuits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from qiskit import QuantumCircuit
from .parameters import parameter_names


@dataclass(frozen=True)
class GeneratedCircuit:
    """A Qiskit circuit plus JSON-friendly generation metadata."""

    circuit: QuantumCircuit
    family: str
    n_qubits: int
    n_clbits: int
    parameters: list[str]
    target_type: str
    phase_sensitive: bool
    entanglement_expected: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Count two-qubit operations in a circuit."""
    return sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)


def has_measurements(circuit: QuantumCircuit) -> bool:
    """Return whether a circuit contains any measurement operations."""
    return any(instruction.operation.name == "measure" for instruction in circuit.data)


def summarize_circuit(circuit: QuantumCircuit) -> dict[str, Any]:
    """Summarize structural properties of a Qiskit circuit."""
    return {
        "n_qubits": circuit.num_qubits,
        "n_clbits": circuit.num_clbits,
        "depth": circuit.depth(),
        "two_qubit_gate_count": count_two_qubit_gates(circuit),
        "parameter_count": len(circuit.parameters),
        "has_measurements": has_measurements(circuit),
    }


def make_generated_circuit(
    circuit: QuantumCircuit,
    family: str,
    target_type: str,
    phase_sensitive: bool,
    entanglement_expected: bool,
    metadata: dict[str, Any] | None = None,
) -> GeneratedCircuit:
    """Build a GeneratedCircuit with consistent summary metadata."""
    summary = summarize_circuit(circuit)
    merged: dict[str, Any] = {
        "family": family,
        "phase_sensitive": phase_sensitive,
        "entanglement_expected": entanglement_expected,
        "target_type": target_type,
        **summary,
    }
    if metadata:
        merged.update(metadata)
    params = parameter_names(circuit)
    return GeneratedCircuit(
        circuit=circuit,
        family=family,
        n_qubits=circuit.num_qubits,
        n_clbits=circuit.num_clbits,
        parameters=params,
        target_type=target_type,
        phase_sensitive=phase_sensitive,
        entanglement_expected=entanglement_expected,
        metadata=merged,
    )
