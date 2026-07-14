"""Ideal shot sampling from exact Born probabilities."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .ideal_statevector import simulate_ideal_statevector
from .result_normalization import counts_to_probabilities, sample_counts_from_probabilities
from .measurement import MeasurementSetting
from .results import IdealShotResult


def simulate_ideal_shots(
    circuit_or_generated: Any,
    shots: int = 1024,
    seed: int | None = None,
    parameter_values: Mapping[str, float] | Mapping[Any, float] | None = None,
    measurement_basis: str | tuple[str, ...] | MeasurementSetting | None = None,
) -> IdealShotResult:
    """Sample ideal counts from statevector Born probabilities."""
    if shots <= 0:
        raise ValueError("shots must be positive.")
    source = simulate_ideal_statevector(circuit_or_generated, parameter_values=parameter_values, measurement_basis=measurement_basis)
    counts = sample_counts_from_probabilities(source.probabilities, shots=shots, seed=seed)
    probabilities = counts_to_probabilities(counts)
    metadata = {
        "seed": seed,
        "shots": shots,
        "simulation_mode": "ideal_shot",
        "source_simulation_mode": source.simulation_mode,
        "measurement_setting": source.metadata["measurement_setting"],
        "probability_domain": "p(y|M)",
    }
    return IdealShotResult(
        simulation_mode="ideal_shot",
        n_qubits=source.n_qubits,
        shots=shots,
        counts=counts,
        probabilities=probabilities,
        source_probabilities=source.probabilities,
        metadata=metadata,
    )
