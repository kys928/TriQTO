"""Density-matrix execution helpers for simulator-only open-system evidence."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from triqto.metrics.hilbert import purity

from .noisy_shot import NoiseSpec, _build_noise_model
from .result_normalization import bind_parameter_values


@dataclass(frozen=True, slots=True)
class DensityMatrixResult:
    simulation_mode: str
    n_qubits: int
    density_matrix: np.ndarray
    probabilities: dict[str, float]
    metadata: dict[str, Any]


def _validate_density(matrix: np.ndarray, n_qubits: int) -> None:
    expected = 2 ** n_qubits
    if matrix.shape != (expected, expected):
        raise ValueError("density matrix dimension mismatch")
    purity(matrix)


def simulate_density_matrix(circuit_or_generated: Any, *, noise: NoiseSpec | None = None, parameter_values: Mapping[str, float] | Mapping[Any, float] | None = None, seed: int = 0) -> DensityMatrixResult:
    from qiskit_aer import AerSimulator

    circuit = bind_parameter_values(getattr(circuit_or_generated, "circuit", circuit_or_generated), parameter_values)
    prepared = circuit.remove_final_measurements(inplace=False)
    prepared.save_density_matrix()
    simulator = AerSimulator(method="density_matrix", noise_model=_build_noise_model(noise) if noise is not None else None, seed_simulator=seed)
    result = simulator.run(prepared, seed_simulator=seed).result()
    matrix = np.asarray(result.data(0)["density_matrix"], dtype=np.complex128)
    _validate_density(matrix, prepared.num_qubits)
    diag = np.real(np.diag(matrix))
    probabilities = {format(i, f"0{prepared.num_qubits}b"): float(p) for i, p in enumerate(diag) if p > 1e-12}
    total = sum(probabilities.values())
    probabilities = {k: v / total for k, v in sorted(probabilities.items())}
    return DensityMatrixResult(
        simulation_mode="density_matrix" if noise is None else "noisy_density_matrix",
        n_qubits=prepared.num_qubits,
        density_matrix=matrix,
        probabilities=probabilities,
        metadata={"seed": seed, "noise_model_id": noise.noise_model_id if noise else None, "evidence_tier": "density_simulator"},
    )


__all__ = ["DensityMatrixResult", "simulate_density_matrix"]
