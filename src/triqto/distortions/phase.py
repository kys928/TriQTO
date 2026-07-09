"""Phase distortion operators for TriQTO circuits."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import DistortedCircuit, copy_circuit, extract_circuit, make_distorted_circuit, validate_finite_strength, validate_qubits


def apply_phase_rz_drift(circuit_or_generated: Any, strength: float, qubits: Sequence[int] | None = None) -> DistortedCircuit:
    """Append deterministic RZ drift rotations to selected qubits."""
    value = validate_finite_strength(strength)
    clean = extract_circuit(circuit_or_generated)
    selected = validate_qubits(clean.num_qubits, qubits)
    distorted = copy_circuit(clean)
    for qubit in selected:
        distorted.rz(value, qubit)
    return make_distorted_circuit(
        circuit_or_generated,
        distorted_circuit=distorted,
        distortion_type="phase_rz_drift",
        distortion_family="phase",
        strength=value,
        affected_qubits=selected,
        affected_gates=["rz" for _ in selected],
        metadata={"selected_qubits": selected, "note": "Ideal circuit-level phase drift via appended RZ rotations."},
    )
