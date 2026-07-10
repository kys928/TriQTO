from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from triqto.actions import (
    ActionCandidate,
    ActionEdit,
    ActionEngineConfig,
    action_content_hash,
    action_risk_score,
    apply_action,
    candidate_action_id,
    circuit_semantic_hash,
)


def make_candidate(edits: tuple[ActionEdit, ...]) -> ActionCandidate:
    config = ActionEngineConfig()
    action_id = candidate_action_id(
        sample_id="sample_1",
        graph_pair_id="graphpair_1",
        source_circuit_id="circuit_distorted",
        source_run_id="run_distorted",
        edits=edits,
    )
    candidate = ActionCandidate(
        action_id=action_id,
        sample_id="sample_1",
        graph_pair_id="graphpair_1",
        source_circuit_id="circuit_distorted",
        source_run_id="run_distorted",
        distortion_id="distortion_1",
        edits=edits,
        generation_sources=("blind_physics_prior",) if edits else ("no_op",),
        risk_score=action_risk_score(edits, config),
        metadata={},
    )
    candidate.content_hash = action_content_hash(candidate)
    return candidate


def measurement_map(circuit: QuantumCircuit):
    return [
        (
            circuit.find_bit(item.qubits[0]).index,
            circuit.find_bit(item.clbits[0]).index,
        )
        for item in circuit.data
        if item.operation.name == "measure"
    ]


def test_apply_action_preserves_source_and_final_measurements():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    before = circuit.copy()
    candidate = make_candidate((ActionEdit("append_rx", (0,), -0.2),))

    applied = apply_action(circuit, candidate, ActionEngineConfig())

    assert circuit == before
    assert applied.circuit is not circuit
    assert measurement_map(applied.circuit) == measurement_map(circuit)
    assert [item.operation.name for item in applied.circuit.data][-3:] == [
        "rx",
        "measure",
        "measure",
    ]
    assert applied.candidate_gate_count == applied.source_gate_count + 1
    assert applied.circuit_hash == circuit_semantic_hash(applied.circuit)


def test_no_op_returns_independent_semantically_equal_circuit():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    candidate = make_candidate(())
    applied = apply_action(circuit, candidate, ActionEngineConfig())
    assert applied.circuit == circuit
    assert applied.circuit is not circuit
    assert applied.source_gate_count == applied.candidate_gate_count


def test_rzz_action_is_applied_on_requested_edge():
    circuit = QuantumCircuit(2)
    candidate = make_candidate(
        (ActionEdit("append_rzz", (0, 1), math.pi / 7),)
    )
    applied = apply_action(circuit, candidate, ActionEngineConfig())
    names = [item.operation.name for item in applied.circuit.data]
    assert "rzz" in names or names == ["cx", "rz", "cx"]


def test_action_application_rejects_mid_circuit_measurement():
    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0)
    circuit.x(1)
    candidate = make_candidate((ActionEdit("append_rz", (1,), 0.1),))
    with pytest.raises(ValueError, match="measurements"):
        apply_action(circuit, candidate, ActionEngineConfig())


def test_action_application_rejects_unbound_parameters():
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("theta"), 0)
    candidate = make_candidate((ActionEdit("append_rz", (0,), 0.1),))
    with pytest.raises(ValueError, match="fully bound"):
        apply_action(circuit, candidate, ActionEngineConfig())


def test_action_validation_rejects_out_of_range_qubit():
    circuit = QuantumCircuit(1)
    candidate = make_candidate((ActionEdit("append_rx", (1,), 0.1),))
    with pytest.raises(ValueError, match="out of range"):
        apply_action(circuit, candidate, ActionEngineConfig())
