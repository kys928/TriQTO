"""Entangling drift distortions for TriQTO circuits."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import DistortedCircuit, copy_circuit, extract_circuit, make_distorted_circuit, validate_finite_strength


def validate_edges(n_qubits: int, edges: Sequence[tuple[int, int]] | None) -> list[tuple[int, int]]:
    """Validate entangling edges, defaulting to nearest-neighbor line edges."""
    if n_qubits < 2:
        raise ValueError("entangling_rzz_drift requires at least 2 qubits.")
    selected = [(i, i + 1) for i in range(n_qubits - 1)] if edges is None else [(int(a), int(b)) for a, b in edges]
    for a, b in selected:
        if a == b:
            raise ValueError(f"Invalid self-loop edge ({a}, {b}).")
        if a < 0 or b < 0 or a >= n_qubits or b >= n_qubits:
            raise ValueError(f"Edge ({a}, {b}) is out of range for {n_qubits} qubits.")
    return selected


def append_rzz_or_decomposition(circuit: Any, strength: float, a: int, b: int) -> str:
    """Append an RZZ interaction, falling back to CX-RZ-CX if unavailable."""
    if hasattr(circuit, "rzz"):
        circuit.rzz(strength, a, b)
        return "native_rzz"
    circuit.cx(a, b)
    circuit.rz(strength, b)
    circuit.cx(a, b)
    return "cx_rz_cx"


def apply_entangling_rzz_drift(circuit_or_generated: Any, strength: float, edges: Sequence[tuple[int, int]] | None = None) -> DistortedCircuit:
    """Append deterministic RZZ drift interactions on selected edges."""
    value = validate_finite_strength(strength)
    clean = extract_circuit(circuit_or_generated)
    selected_edges = validate_edges(clean.num_qubits, edges)
    distorted = copy_circuit(clean)
    decompositions = [append_rzz_or_decomposition(distorted, value, a, b) for a, b in selected_edges]
    edge_lists = [[a, b] for a, b in selected_edges]
    affected_qubits = sorted({q for edge in selected_edges for q in edge})
    metadata = {"edges": edge_lists, "selected_edges": edge_lists, "rzz_decomposition": decompositions[0] if decompositions else "none"}
    if len(set(decompositions)) > 1:
        metadata["rzz_decompositions"] = decompositions
    return make_distorted_circuit(
        circuit_or_generated,
        distorted_circuit=distorted,
        distortion_type="entangling_rzz_drift",
        distortion_family="entangling",
        strength=value,
        affected_qubits=affected_qubits,
        affected_gates=["rzz" if d == "native_rzz" else "cx_rz_cx" for d in decompositions],
        metadata=metadata,
    )
