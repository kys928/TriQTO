from __future__ import annotations

from qiskit import QuantumCircuit
import pytest

from triqto.simulation import NoiseSpec, simulate_density_matrix, simulate_noisy_aer_shots
from triqto.metrics.hilbert import purity, trace_distance


def test_noise_spec_validation_and_identity() -> None:
    first = NoiseSpec(({"type": "depolarizing", "probability": 0.1, "qubits": 1, "gates": ["x"]},))
    second = NoiseSpec(({"gates": ["x"], "qubits": 1, "probability": 0.1, "type": "depolarizing"},))
    assert first.noise_model_id == second.noise_model_id
    with pytest.raises(ValueError, match="unsupported"):
        NoiseSpec(({"type": "pretend", "probability": 0.1},))


def test_noisy_aer_shots_are_seeded_and_normalized() -> None:
    circuit = QuantumCircuit(1)
    circuit.x(0)
    noise = NoiseSpec(({"type": "depolarizing", "probability": 0.0, "qubits": 1, "gates": ["x"]},))
    result = simulate_noisy_aer_shots(circuit, noise=noise, shots=128, seed=11)
    assert result.simulation_mode == "noisy_aer_shot"
    assert sum(result.counts.values()) == 128
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.metadata["noise_model_id"] == noise.noise_model_id


def test_density_matrix_execution_validates_physical_matrix() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    result = simulate_density_matrix(circuit)
    assert result.simulation_mode == "density_matrix"
    assert purity(result.density_matrix) == pytest.approx(1.0)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_density_noise_changes_amplitude_damping_state() -> None:
    circuit = QuantumCircuit(1)
    circuit.x(0)
    clean = simulate_density_matrix(circuit)
    damped = simulate_density_matrix(circuit, noise=NoiseSpec(({"type": "amplitude_damping", "probability": 0.5, "gates": ["x"]},)))
    assert trace_distance(clean.density_matrix, damped.density_matrix) > 0.1
