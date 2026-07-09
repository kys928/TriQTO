"""Amplitude-axis overrotation distortions for TriQTO circuits."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import DistortedCircuit, copy_for_unitary_distortion, extract_circuit, make_distorted_circuit, validate_finite_strength, validate_qubits


def _apply_axis_overrotation(circuit_or_generated: Any, *, strength: float, qubits: Sequence[int] | None, axis: str) -> DistortedCircuit:
    value = validate_finite_strength(strength)
    clean = extract_circuit(circuit_or_generated)
    selected = validate_qubits(clean.num_qubits, qubits)
    distorted, restore_measurements, measurement_metadata = copy_for_unitary_distortion(clean)
    gate_name = f"r{axis}"
    for qubit in selected:
        if axis == "x":
            distorted.rx(value, qubit)
        elif axis == "y":
            distorted.ry(value, qubit)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported overrotation axis {axis!r}.")
    distorted = restore_measurements(distorted)
    return make_distorted_circuit(
        circuit_or_generated,
        distorted_circuit=distorted,
        distortion_type=f"{gate_name}_overrotation",
        distortion_family="amplitude",
        strength=value,
        affected_qubits=selected,
        affected_gates=[gate_name for _ in selected],
        metadata={**measurement_metadata, "axis": axis, "selected_qubits": selected, "note": f"Ideal circuit-level R{axis.upper()} overrotation before final measurements."},
    )


def apply_rx_overrotation(circuit_or_generated: Any, strength: float, qubits: Sequence[int] | None = None) -> DistortedCircuit:
    """Append deterministic RX overrotations to selected qubits."""
    return _apply_axis_overrotation(circuit_or_generated, strength=strength, qubits=qubits, axis="x")


def apply_ry_overrotation(circuit_or_generated: Any, strength: float, qubits: Sequence[int] | None = None) -> DistortedCircuit:
    """Append deterministic RY overrotations to selected qubits."""
    return _apply_axis_overrotation(circuit_or_generated, strength=strength, qubits=qubits, axis="y")
