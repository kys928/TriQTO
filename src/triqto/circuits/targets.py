"""Lightweight target metadata for circuit families."""
from __future__ import annotations
from typing import Any


def target_for_family(family: str, n_qubits: int) -> dict[str, Any]:
    if n_qubits < 1: raise ValueError("n_qubits must be >= 1")
    target_types = {
        "bell": "bell_pair", "ghz": "state_preparation", "phase_interference": "phase_interference",
        "qft_like": "phase_structure", "hardware_efficient_ansatz": "ansatz",
        "random_shallow": "random_shallow", "lattice_entangled": "lattice_entanglement", "qaoa_like": "qaoa_like",
    }
    if family not in target_types:
        raise ValueError(f"Unknown target family {family!r}. Available families: {', '.join(sorted(target_types))}")
    expected = 2 if family in {"bell", "ghz"} else None
    return {"family": family, "n_qubits": n_qubits, "target_type": target_types[family], "expected_support_size": expected, "statevector_target_available": False}
