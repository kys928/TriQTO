"""Normalization and circuit-preparation helpers for ideal simulation."""
from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from qiskit import QuantumCircuit


def extract_quantum_circuit(circuit_or_generated: Any) -> QuantumCircuit:
    """Return a Qiskit circuit from a QuantumCircuit or GeneratedCircuit-like object."""
    if isinstance(circuit_or_generated, QuantumCircuit):
        return circuit_or_generated
    circuit = getattr(circuit_or_generated, "circuit", None)
    if isinstance(circuit, QuantumCircuit):
        return circuit
    raise TypeError("Expected a qiskit.QuantumCircuit or an object with a QuantumCircuit 'circuit' attribute.")


def _has_measurements(circuit: QuantumCircuit) -> bool:
    return any(instruction.operation.name == "measure" for instruction in circuit.data)


def _has_classical_conditions(circuit: QuantumCircuit) -> bool:
    for instruction in circuit.data:
        operation = instruction.operation
        if getattr(operation, "condition", None) is not None:
            return True
    return False


def copy_without_final_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Copy ``circuit`` and remove only safely removable final measurements.

    Mid-circuit measurements or classically conditioned operations are rejected because
    they cannot be represented as a single ideal unitary statevector instruction.
    """
    if _has_classical_conditions(circuit):
        raise ValueError("Ideal statevector simulation does not support classical conditions.")

    copied = circuit.copy()
    if not _has_measurements(copied):
        return copied

    try:
        stripped = copied.remove_final_measurements(inplace=False)
    except TypeError:  # pragma: no cover - compatibility with older Qiskit
        stripped = copied.copy()
        stripped.remove_final_measurements(inplace=True)

    if _has_measurements(stripped):
        raise ValueError("Circuit contains mid-circuit measurements that cannot be safely removed.")
    return stripped


def validate_no_unbound_parameters(circuit: QuantumCircuit) -> None:
    """Raise if ``circuit`` still contains symbolic parameters."""
    if circuit.parameters:
        names = ", ".join(sorted(parameter.name for parameter in circuit.parameters))
        raise ValueError(f"Circuit has unbound parameters; provide parameter_values for: {names}")


def bind_parameter_values(
    circuit: QuantumCircuit,
    parameter_values: Mapping[str, float] | Mapping[Any, float] | None,
) -> QuantumCircuit:
    """Bind provided parameter values on a circuit copy without mutating the original."""
    copied = circuit.copy()
    if parameter_values is None:
        validate_no_unbound_parameters(copied)
        return copied

    by_name = {parameter.name: parameter for parameter in copied.parameters}
    assignments: dict[Any, float] = {}
    for key, value in parameter_values.items():
        if isinstance(key, str):
            if key not in by_name:
                raise ValueError(f"Unknown circuit parameter name: {key}")
            assignments[by_name[key]] = float(value)
        else:
            assignments[key] = float(value)

    bound = copied.assign_parameters(assignments, inplace=False)
    validate_no_unbound_parameters(bound)
    return bound


def normalize_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """Return integer, nonnegative counts with stable sorted keys."""
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError("Counts must be nonnegative.")
        if ivalue:
            normalized[str(key)] = ivalue
    return dict(sorted(normalized.items()))


def counts_to_probabilities(counts: Mapping[str, int]) -> dict[str, float]:
    """Convert counts to empirical probabilities."""
    normalized_counts = normalize_counts(counts)
    total = sum(normalized_counts.values())
    if total <= 0:
        raise ValueError("Total shot count must be positive.")
    return {key: value / total for key, value in normalized_counts.items()}


def normalize_probabilities(probabilities: Mapping[str, float], *, atol: float = 1e-12) -> dict[str, float]:
    """Validate and normalize a sparse probability distribution."""
    cleaned: dict[str, float] = {}
    for key, value in probabilities.items():
        fvalue = float(value)
        if fvalue < 0:
            if abs(fvalue) <= atol:
                fvalue = 0.0
            else:
                raise ValueError("Probabilities must be nonnegative.")
        if fvalue > atol:
            cleaned[str(key)] = fvalue
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("Probability distribution must have positive total mass.")
    return dict(sorted((key, value / total) for key, value in cleaned.items()))


def sample_counts_from_probabilities(
    probabilities: Mapping[str, float], shots: int, seed: int | None = None
) -> dict[str, int]:
    """Sample counts from probabilities using a deterministic RNG seed."""
    if shots <= 0:
        raise ValueError("shots must be positive.")
    normalized = normalize_probabilities(probabilities)
    keys = list(normalized)
    weights = [normalized[key] for key in keys]
    rng = random.Random(seed)
    sampled = rng.choices(keys, weights=weights, k=shots)
    counts = {key: 0 for key in keys}
    for key in sampled:
        counts[key] += 1
    return normalize_counts(counts)
