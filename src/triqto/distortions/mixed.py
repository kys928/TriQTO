"""Mixed deterministic unitary drift distortions for TriQTO circuits."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import DistortedCircuit, copy_for_unitary_distortion, extract_circuit, make_distorted_circuit, validate_finite_strength, validate_qubits
from .entangling import append_rzz_or_decomposition


def apply_mixed_unitary_drift(
    circuit_or_generated: Any,
    strength: float,
    qubits: Sequence[int] | None = None,
    include_entangling: bool = True,
) -> DistortedCircuit:
    """Append deterministic RZ, RX, and optional neighboring RZZ drift components."""
    value = validate_finite_strength(strength)
    clean = extract_circuit(circuit_or_generated)
    selected = validate_qubits(clean.num_qubits, qubits)
    distorted, restore_measurements, measurement_metadata = copy_for_unitary_distortion(clean)
    affected_gates: list[str] = []
    for qubit in selected:
        distorted.rz(value, qubit)
        affected_gates.append("rz")
    half = value / 2.0
    for qubit in selected:
        distorted.rx(half, qubit)
        affected_gates.append("rx")
    edges = list(zip(selected, selected[1:])) if include_entangling and len(selected) >= 2 else []
    decompositions = []
    for a, b in edges:
        decomposition = append_rzz_or_decomposition(distorted, half, a, b)
        decompositions.append(decomposition)
        affected_gates.append("rzz" if decomposition == "native_rzz" else "cx_rz_cx")
    components = [
        {"type": "phase_rz_drift", "gate": "rz", "strength": value, "qubits": selected},
        {"type": "rx_overrotation", "gate": "rx", "strength": half, "qubits": selected},
    ]
    if edges:
        components.append({"type": "entangling_rzz_drift", "gate": "rzz", "strength": half, "edges": [[a, b] for a, b in edges]})
    distorted = restore_measurements(distorted)
    return make_distorted_circuit(
        circuit_or_generated,
        distorted_circuit=distorted,
        distortion_type="mixed_unitary_drift",
        distortion_family="mixed",
        strength=value,
        affected_qubits=selected,
        affected_gates=affected_gates,
        metadata={
            **measurement_metadata,
            "selected_qubits": selected,
            "include_entangling": include_entangling,
            "edges": [[a, b] for a, b in edges],
            "component_distortions": components,
            "rzz_decomposition": decompositions[0] if decompositions else "none",
        },
    )
