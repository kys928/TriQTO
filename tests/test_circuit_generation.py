"""Tests for Phase 3 circuit family generation."""
from __future__ import annotations

import pytest

from triqto.circuits import count_two_qubit_gates, generate_circuit_family, list_circuit_families
from triqto.circuits.bell import make_bell_circuit
from triqto.circuits.ghz import make_ghz_circuit
from triqto.circuits.hardware_efficient import make_hardware_efficient_ansatz
from triqto.circuits.lattice_entangled import make_lattice_entangled_circuit
from triqto.circuits.parameters import bind_random_parameters, parameter_names
from triqto.circuits.phase_interference import make_phase_interference_circuit
from triqto.circuits.qaoa_like import make_qaoa_like_circuit
from triqto.circuits.random_shallow import make_random_shallow_circuit

EXPECTED = {"bell", "ghz", "phase_interference", "qft_like", "hardware_efficient_ansatz", "random_shallow", "lattice_entangled", "qaoa_like"}


def test_list_circuit_families_includes_expected_families():
    assert EXPECTED.issubset(set(list_circuit_families()))


def test_generate_circuit_family_works_for_ghz():
    generated = generate_circuit_family("ghz", 4)
    assert generated.family == "ghz"
    assert generated.circuit.num_qubits == 4


def test_ghz_four_qubits_has_three_cnots():
    generated = make_ghz_circuit(4, measure=False)
    assert generated.n_qubits == 4
    assert count_two_qubit_gates(generated.circuit) == 3


def test_bell_rejects_too_few_qubits():
    with pytest.raises(ValueError, match="n_qubits >= 2"):
        make_bell_circuit(1)


def test_bell_four_qubits_marks_extra_idle_qubits():
    generated = make_bell_circuit(4)
    assert generated.metadata["has_idle_extra_qubits"] is True
    assert generated.metadata["idle_qubits"] == [2, 3]


def test_phase_interference_has_parameters_and_is_phase_sensitive():
    generated = make_phase_interference_circuit(3, layers=2)
    assert generated.parameters
    assert generated.phase_sensitive is True
    assert generated.metadata["parameter_count"] == 12


def test_hardware_efficient_ansatz_has_parameters():
    generated = make_hardware_efficient_ansatz(4, layers=2)
    assert generated.parameters
    assert "theta_0_0" in generated.parameters


def test_random_shallow_is_reproducible_with_same_seed():
    first = make_random_shallow_circuit(4, depth=3, seed=123)
    second = make_random_shallow_circuit(4, depth=3, seed=123)
    assert str(first.circuit) == str(second.circuit)
    assert first.metadata["random_gate_log"] == second.metadata["random_gate_log"]


def test_lattice_entangled_stores_edges():
    generated = make_lattice_entangled_circuit(4, layout="line")
    assert generated.metadata["lattice_edges"] == [[0, 1], [1, 2], [2, 3]]


def test_qaoa_like_exposes_gamma_beta_parameters():
    generated = make_qaoa_like_circuit(4, layers=2)
    assert {"gamma_0", "beta_0", "gamma_1", "beta_1"}.issubset(set(generated.parameters))


def test_generated_metadata_includes_required_fields():
    generated = make_ghz_circuit(4)
    for key in ["depth", "two_qubit_gate_count", "parameter_count", "family", "n_qubits", "has_measurements"]:
        assert key in generated.metadata
    assert generated.metadata["family"] == "ghz"
    assert generated.metadata["n_qubits"] == 4
    assert generated.metadata["has_measurements"] is True


@pytest.mark.parametrize("n_qubits", [4, 6])
def test_variable_size_behavior_for_supported_families(n_qubits):
    assert make_ghz_circuit(n_qubits).n_qubits == n_qubits
    assert make_phase_interference_circuit(n_qubits).n_qubits == n_qubits
    assert make_hardware_efficient_ansatz(n_qubits).n_qubits == n_qubits
    assert make_lattice_entangled_circuit(n_qubits).n_qubits == n_qubits
    assert make_qaoa_like_circuit(n_qubits).n_qubits == n_qubits


def test_parameter_names_returns_sorted_stable_names():
    generated = make_hardware_efficient_ansatz(2, layers=1, measure=False)
    assert parameter_names(generated.circuit) == sorted(generated.parameters)


def test_bind_random_parameters_does_not_mutate_original():
    generated = make_phase_interference_circuit(2, layers=1, measure=False)
    original_names = parameter_names(generated.circuit)
    bound = bind_random_parameters(generated.circuit, seed=7)
    assert parameter_names(generated.circuit) == original_names
    assert parameter_names(bound) == []
