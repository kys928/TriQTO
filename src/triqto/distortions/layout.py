"""Marker-only layout distortion records for Phase 5."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import DistortedCircuit, copy_circuit, extract_circuit, make_distorted_circuit


def validate_permutation(n_qubits: int, permutation: Sequence[int]) -> list[int]:
    """Validate a logical-to-physical qubit permutation marker."""
    selected = [int(q) for q in permutation]
    if sorted(selected) != list(range(n_qubits)):
        raise ValueError(f"permutation must contain each index in range({n_qubits}) exactly once.")
    return selected


def apply_layout_permutation_marker(circuit_or_generated: Any, permutation: Sequence[int]) -> DistortedCircuit:
    """Record a layout permutation marker without transpiling or remapping gates."""
    clean = extract_circuit(circuit_or_generated)
    selected = validate_permutation(clean.num_qubits, permutation)
    distorted = copy_circuit(clean)
    return make_distorted_circuit(
        circuit_or_generated,
        distorted_circuit=distorted,
        distortion_type="layout_permutation_marker",
        distortion_family="layout",
        strength=None,
        affected_qubits=selected,
        affected_gates=[],
        metadata={"marker_only": True, "not_transpiled": True, "permutation": selected},
        ideal_only_distortion_model=False,
    )
