"""Marker-only readout distortion records for Phase 5."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import DistortedCircuit, copy_circuit, extract_circuit, make_distorted_circuit, validate_probability, validate_qubits


def apply_readout_bitflip_marker(circuit_or_generated: Any, probability: float, qubits: Sequence[int] | None = None) -> DistortedCircuit:
    """Record a readout bit-flip marker without simulating readout noise."""
    value = validate_probability(probability)
    clean = extract_circuit(circuit_or_generated)
    selected = validate_qubits(clean.num_qubits, qubits)
    distorted = copy_circuit(clean)
    return make_distorted_circuit(
        circuit_or_generated,
        distorted_circuit=distorted,
        distortion_type="readout_bitflip_marker",
        distortion_family="readout",
        strength=value,
        affected_qubits=selected,
        affected_gates=[],
        metadata={"marker_only": True, "not_a_noisy_simulator": True, "selected_qubits": selected, "probability": value},
        ideal_only_distortion_model=False,
    )
