from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit

from triqto.circuits.families import generate_circuit_family, list_circuit_families
from triqto.graph import (
    GraphConversionConfig,
    circuit_to_graph,
    graph_content_hash,
    validate_graph_data,
)


def one_outcome(n_qubits: int):
    return {"0" * n_qubits: 1.0}


def test_variable_size_nodes_multiedges_measurements_and_no_mutation():
    circuit = QuantumCircuit(4, 2)
    circuit.h(0)
    circuit.x(2)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    circuit.swap(1, 2)
    circuit.reset(3)
    circuit.barrier()
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    before = circuit.copy()
    graph = circuit_to_graph(
        circuit,
        circuit_id="circuit_test",
        source_run_id="run_test",
        role="clean",
        family="unit",
        parameter_bindings={"b": 2.0, "a": 1.0},
        exact_probabilities=one_outcome(4),
        source_sample_ids=["sample_a", "sample_b"],
    )
    validate_graph_data(graph)
    assert graph.node_features.shape[0] == 4
    assert graph.edge_index.shape == (2, 6)
    assert graph.edge_event_index.tolist() == [2, 2, 3, 3, 4, 4]
    assert graph.parameter_names.tolist() == ["a", "b"]
    assert graph.source_sample_ids == ("sample_a", "sample_b")
    assert circuit == before


def test_cx_direction_and_swap_symmetry_are_explicit():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.swap(0, 1)
    graph = circuit_to_graph(
        circuit,
        circuit_id="c",
        source_run_id="r",
        role="clean",
        family="unit",
        parameter_bindings={},
        exact_probabilities=one_outcome(2),
    )
    cx_forward = graph.edge_features[0]
    cx_reverse = graph.edge_features[1]
    assert cx_forward[5] == 1.0 and cx_forward[6] == 1.0
    assert cx_reverse[5] == 0.0 and cx_reverse[6] == 0.0
    assert graph.edge_features[2, 7] == 1.0
    assert graph.edge_features[3, 7] == 1.0


def test_three_qubit_gate_is_event_without_fake_clique():
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    graph = circuit_to_graph(
        circuit,
        circuit_id="c",
        source_run_id="r",
        role="clean",
        family="unit",
        parameter_bindings={},
        exact_probabilities=one_outcome(3),
    )
    assert graph.gate_features.shape[0] == 1
    assert graph.edge_index.shape == (2, 0)
    assert graph.scientific_metadata["multi_qubit_event_count"] == 1


def test_layers_allow_parallel_disjoint_gates_and_order_same_qubit():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.x(1)
    circuit.z(0)
    graph = circuit_to_graph(
        circuit,
        circuit_id="c",
        source_run_id="r",
        role="clean",
        family="unit",
        parameter_bindings={},
        exact_probabilities=one_outcome(2),
    )
    assert graph.gate_features[:, 4].tolist() == [0.0, 0.0, 1.0]


def test_angular_parameters_and_global_phase_exclusion():
    circuit = QuantumCircuit(1)
    circuit.global_phase = 0.5
    circuit.rz(math.pi / 7, 0)
    graph = circuit_to_graph(
        circuit,
        circuit_id="c",
        source_run_id="r",
        role="distorted",
        family="unit",
        parameter_bindings={"theta": math.pi / 7},
        exact_probabilities={"0": 1.0},
    )
    assert graph.gate_parameter_angle_mask.tolist() == [True]
    assert graph.gate_parameter_sin[0] == pytest.approx(math.sin(math.pi / 7))
    assert graph.global_features.shape == (0,)
    assert graph.provenance_metadata["global_phase"] == pytest.approx(0.5)
    assert graph.provenance_metadata["global_phase_excluded_from_features"] is True


def test_gate_and_probability_guardrails_are_enforced():
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.z(0)
    with pytest.raises(ValueError, match="max_gate_events"):
        circuit_to_graph(
            circuit,
            circuit_id="c",
            source_run_id="r",
            role="clean",
            family="unit",
            parameter_bindings={},
            exact_probabilities={"0": 1.0},
            config=GraphConversionConfig(max_gate_events=1),
        )
    with pytest.raises(ValueError, match="max_probability_outcomes"):
        circuit_to_graph(
            QuantumCircuit(1),
            circuit_id="c",
            source_run_id="r",
            role="clean",
            family="unit",
            parameter_bindings={},
            exact_probabilities={"0": 0.5, "1": 0.5},
            config=GraphConversionConfig(max_probability_outcomes=1),
        )


def test_supplemental_counts_do_not_change_structural_hash():
    circuit = QuantumCircuit(1)
    without_counts = circuit_to_graph(
        circuit,
        circuit_id="c",
        source_run_id="r",
        role="clean",
        family="unit",
        parameter_bindings={},
        exact_probabilities={"0": 1.0},
    )
    with_counts = circuit_to_graph(
        circuit,
        circuit_id="c",
        source_run_id="r",
        role="clean",
        family="unit",
        parameter_bindings={},
        exact_probabilities={"0": 1.0},
        supplemental_counts={"0": 4},
        supplemental_shots=4,
    )
    assert without_counts.graph_id == with_counts.graph_id
    assert graph_content_hash(without_counts) == graph_content_hash(with_counts)


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
def test_graph_shapes_scale_without_padding(n_qubits):
    graph = circuit_to_graph(
        QuantumCircuit(n_qubits),
        circuit_id=f"c{n_qubits}",
        source_run_id=f"r{n_qubits}",
        role="clean",
        family="unit",
        parameter_bindings={},
        exact_probabilities=one_outcome(n_qubits),
    )
    assert graph.node_features.shape[0] == n_qubits
    assert graph.node_index.shape == (n_qubits,)


@pytest.mark.parametrize("family", list_circuit_families())
def test_all_circuit_families_convert(family):
    generated = generate_circuit_family(family, n_qubits=4)
    circuit = generated.circuit
    if circuit.parameters:
        circuit = circuit.assign_parameters(
            {parameter: 0.1 for parameter in circuit.parameters},
            inplace=False,
        )
    graph = circuit_to_graph(
        circuit,
        circuit_id=f"c_{family}",
        source_run_id=f"r_{family}",
        role="clean",
        family=family,
        parameter_bindings={parameter: 0.1 for parameter in generated.parameters},
        exact_probabilities=one_outcome(4),
    )
    validate_graph_data(graph)
    assert graph.n_qubits == 4


def test_conditioned_control_flow_rejected_when_supported():
    circuit = QuantumCircuit(1, 1)
    if not hasattr(circuit, "if_test"):
        pytest.skip("Installed Qiskit does not expose QuantumCircuit.if_test")
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)
    with pytest.raises(NotImplementedError, match="event 0"):
        circuit_to_graph(
            circuit,
            circuit_id="c",
            source_run_id="r",
            role="clean",
            family="unit",
            parameter_bindings={},
            exact_probabilities={"0": 1.0},
        )
