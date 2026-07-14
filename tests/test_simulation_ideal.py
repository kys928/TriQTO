"""Tests for Phase 4 ideal simulation."""
from __future__ import annotations

import sys
import subprocess

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

import triqto.simulation as simulation
from triqto.circuits.ghz import make_ghz_circuit
from triqto.simulation import (
    bind_parameter_values,
    counts_to_probabilities,
    normalize_probabilities,
    sample_counts_from_probabilities,
    simulate_ideal_shots,
    simulate_ideal_statevector,
)


def test_ghz_ideal_statevector_probabilities_are_expected():
    result = simulate_ideal_statevector(make_ghz_circuit(4, measure=True))
    assert set(result.probabilities) == {"0000", "1111"}
    assert result.probabilities["0000"] == pytest.approx(0.5)
    assert result.probabilities["1111"] == pytest.approx(0.5)


def test_final_measurements_removed_without_mutating_original():
    generated = make_ghz_circuit(4, measure=True)
    assert generated.circuit.count_ops().get("measure", 0) == 4
    result = simulate_ideal_statevector(generated.circuit)
    assert result.metadata["measurements_removed"] is True
    assert generated.circuit.count_ops().get("measure", 0) == 4


def test_mid_circuit_measurements_are_rejected():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(0)
    with pytest.raises(ValueError, match="mid-circuit measurements"):
        simulate_ideal_statevector(circuit)


def test_simulate_ideal_shots_sums_and_is_deterministic():
    generated = make_ghz_circuit(4, measure=True)
    first = simulate_ideal_shots(generated, shots=257, seed=99)
    second = simulate_ideal_shots(generated, shots=257, seed=99)
    assert sum(first.counts.values()) == 257
    assert first.counts == second.counts


def test_counts_to_probabilities_sums_to_one():
    probabilities = counts_to_probabilities({"0": 1, "1": 3})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["1"] == pytest.approx(0.75)


def test_normalize_probabilities_accepts_valid_and_rejects_invalid():
    normalized = normalize_probabilities({"0": 2.0, "1": 2.0, "tiny": -1e-13})
    assert normalized == {"0": 0.5, "1": 0.5}
    with pytest.raises(ValueError):
        normalize_probabilities({"0": -0.1, "1": 1.1})
    with pytest.raises(ValueError):
        normalize_probabilities({"0": 0.0})


def test_sample_counts_from_probabilities_is_deterministic():
    first = sample_counts_from_probabilities({"0": 0.25, "1": 0.75}, shots=32, seed=7)
    second = sample_counts_from_probabilities({"0": 0.25, "1": 0.75}, shots=32, seed=7)
    assert first == second
    assert sum(first.values()) == 32


def test_parameterized_circuit_without_values_raises():
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    with pytest.raises(ValueError, match="unbound parameters"):
        simulate_ideal_statevector(circuit)


def test_parameterized_circuit_with_values_can_be_simulated():
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    result = simulate_ideal_statevector(circuit, parameter_values={"theta": 0.0})
    assert result.probabilities == {"0": pytest.approx(1.0)}


def test_bind_parameter_values_does_not_mutate_original():
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.rx(theta, 0)
    bound = bind_parameter_values(circuit, {"theta": 0.0})
    assert circuit.parameters
    assert not bound.parameters


def test_simulation_package_exports_expected_functions():
    for name in [
        "IdealStatevectorResult",
        "IdealShotResult",
        "simulate_ideal_statevector",
        "statevector_probabilities",
        "simulate_ideal_shots",
        "run_ideal_sampler",
        "counts_to_probabilities",
    ]:
        assert hasattr(simulation, name)


def test_qiskit_aer_import_is_not_required():
    code = "import sys; import triqto.simulation; assert 'qiskit_aer' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
