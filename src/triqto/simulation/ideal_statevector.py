"""Ideal statevector simulation for TriQTO circuits."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qiskit.quantum_info import Statevector

from .result_normalization import (
    bind_parameter_values,
    copy_without_final_measurements,
    extract_quantum_circuit,
    normalize_probabilities,
)
from .results import IdealStatevectorResult


def statevector_probabilities(statevector: Any, n_qubits: int | None = None) -> dict[str, float]:
    """Extract normalized Born probabilities from a Qiskit Statevector-like object."""
    if hasattr(statevector, "probabilities_dict"):
        probabilities = statevector.probabilities_dict()
    else:
        sv = Statevector(statevector)
        probabilities = sv.probabilities_dict()
    if n_qubits is not None:
        probabilities = {str(key).zfill(n_qubits): value for key, value in probabilities.items()}
    return normalize_probabilities(probabilities)


def simulate_ideal_statevector(
    circuit_or_generated: Any,
    parameter_values: Mapping[str, float] | Mapping[Any, float] | None = None,
) -> IdealStatevectorResult:
    """Simulate a circuit exactly with Qiskit's quantum_info Statevector."""
    original = extract_quantum_circuit(circuit_or_generated)
    had_measurements = any(instruction.operation.name == "measure" for instruction in original.data)
    parameter_binding_requested = parameter_values is not None

    bound = bind_parameter_values(original, parameter_values)
    prepared = copy_without_final_measurements(bound)
    statevector = Statevector.from_instruction(prepared)
    probabilities = statevector_probabilities(statevector, prepared.num_qubits)

    metadata = {
        "original_circuit_name": original.name,
        "prepared_circuit_name": prepared.name,
        "measurements_removed": had_measurements,
        "parameter_binding_requested": parameter_binding_requested,
        "simulation_mode": "ideal_statevector",
    }
    return IdealStatevectorResult(
        simulation_mode="ideal_statevector",
        n_qubits=prepared.num_qubits,
        statevector=statevector,
        probabilities=probabilities,
        metadata=metadata,
    )
