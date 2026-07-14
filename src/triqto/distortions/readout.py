"""Observable symmetric readout-confusion distortion specification."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import (
    DistortedCircuit,
    copy_circuit,
    extract_circuit,
    make_distorted_circuit,
    validate_probability,
    validate_qubits,
)


def apply_readout_bitflip(
    circuit_or_generated: Any,
    probability: float,
    qubits: Sequence[int] | None = None,
) -> DistortedCircuit:
    """Attach a readout channel that the measurement simulator must execute.

    Readout error is not a unitary circuit edit, so the circuit remains unchanged.
    Unlike the old marker, the Phase 7 measurement path applies the declared
    classical channel to ``p(y | M)`` and records its provenance.
    """
    value = validate_probability(probability)
    clean = extract_circuit(circuit_or_generated)
    selected = validate_qubits(clean.num_qubits, qubits)
    distorted = copy_circuit(clean)
    return make_distorted_circuit(
        circuit_or_generated,
        distorted_circuit=distorted,
        distortion_type="readout_bitflip",
        distortion_family="readout",
        strength=value,
        affected_qubits=selected,
        affected_gates=[],
        metadata={
            "marker_only": False,
            "observable_measurement_channel": True,
            "measurement_channel": "independent_symmetric_readout_bitflip",
            "selected_qubits": selected,
            "probability": value,
        },
        ideal_only_distortion_model=False,
    )


def apply_readout_bitflip_marker(
    circuit_or_generated: Any,
    probability: float,
    qubits: Sequence[int] | None = None,
) -> DistortedCircuit:
    """Backward-compatible function name for the now-observable channel."""
    return apply_readout_bitflip(circuit_or_generated, probability, qubits)


__all__ = ["apply_readout_bitflip", "apply_readout_bitflip_marker"]
