from __future__ import annotations

from pathlib import Path
import json

import pytest
from qiskit import QuantumCircuit

from triqto.actions.operational import basis_probe_action, layout_selection_action, routing_transpilation_action, semantics_verified_depth_reduction
from triqto.actions.operational_adapter import build_operational_action_tensor_batch, collate_operational_action_tensor_batches
from triqto.actions.operational_artifacts import load_operational_action_dataset, load_operational_action_result, save_operational_action_result, write_operational_action_dataset
from triqto.actions.identities import circuit_semantic_hash
from triqto.backends import local_line_backend


def test_basis_probe_acquires_new_evidence_without_mutating_source() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    before = circuit_semantic_hash(circuit)
    first = basis_probe_action(1, ("X",), circuit=circuit, shots=64, seed=17)
    second = basis_probe_action(1, ("X",), circuit=circuit, shots=64, seed=17)
    assert circuit_semantic_hash(circuit) == before
    assert first.action_id == second.action_id
    assert first.content_hash == second.content_hash
    assert first.acquires_evidence is True
    assert first.changes_circuit_semantics is False
    assert first.after_metadata["probability_domain"] == "p(y|M)"
    assert first.after_metadata["evidence_namespace"] == "diagnostic_probe"
    assert first.evidence["probe_evidence_id"] == second.evidence["probe_evidence_id"]
    assert sum(first.evidence["counts"].values()) == 64
    assert first.physical_hardware is False
    assert first.privileged_information is False


def test_basis_probe_rejects_invalid_preconditions() -> None:
    circuit = QuantumCircuit(2)
    rejected = basis_probe_action(2, ("X",), circuit=circuit)
    assert rejected.status == "rejected"
    assert rejected.available is False
    assert rejected.availability_mask is False
    assert rejected.rejection_reason


def test_compilation_actions_are_backend_bound_and_not_physical() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    backend = local_line_backend(2)
    _, layout = layout_selection_action(circuit, backend, seed=9)
    _, routing = routing_transpilation_action(circuit, backend, seed=9)
    for result in (layout, routing):
        assert result.source_backend_snapshot_id == backend.backend_id
        assert result.required_evidence_tier == "fake_backend_fixture"
        assert result.physical_hardware is False
        assert result.privileged_information is False
        assert result.evidence["backend_class"] == "fake"
        assert result.evidence["coupling_map"]


def test_depth_reduction_requires_semantics_and_objective_improvement() -> None:
    source = QuantumCircuit(1)
    source.h(0)
    source.h(0)
    source.x(0)
    candidate = QuantumCircuit(1)
    candidate.x(0)
    accepted = semantics_verified_depth_reduction(source, candidate)
    assert accepted.status == "accepted"
    assert accepted.objective_comparison["objective_improved"] is True
    assert accepted.evidence["state_fidelity"] > 1.0 - 1e-12
    equal = semantics_verified_depth_reduction(candidate, candidate)
    assert equal.status == "no_op"
    wrong = QuantumCircuit(1)
    wrong.z(0)
    rejected = semantics_verified_depth_reduction(source, wrong)
    assert rejected.status == "rejected"
    assert rejected.rejection_reason == "semantic distance exceeds tolerance"


def test_operational_action_artifact_roundtrip_and_tamper_rejection(tmp_path: Path) -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    result = basis_probe_action(1, ("Y",), circuit=circuit, shots=32, seed=3)
    artifact = tmp_path / "action.json"
    save_operational_action_result(artifact, result)
    assert load_operational_action_result(artifact) == result
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["after_metadata"]["probability_domain"] = "tampered"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        load_operational_action_result(artifact)


def test_operational_action_dataset_is_immutable_and_content_addressed(tmp_path: Path) -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    backend = local_line_backend(2)
    _, layout = layout_selection_action(circuit, backend, seed=5)
    _, routing = routing_transpilation_action(circuit, backend, seed=5)
    root = tmp_path / "operational"
    written = write_operational_action_dataset(root, (layout, routing), source_dataset_id="phase8_fixture", evidence_tier="fake_backend_fixture")
    loaded = load_operational_action_dataset(root)
    assert loaded["manifest"]["operational_action_dataset_id"] == written["manifest"]["operational_action_dataset_id"]
    assert loaded["phase12_arrays"]["action_candidate_target_mask"].sum() == 0
    assert loaded["phase12_arrays"]["action_privileged_oracle_mask"].sum() == 0
    with pytest.raises(FileExistsError):
        write_operational_action_dataset(root, (layout,), source_dataset_id="phase8_fixture", evidence_tier="fake_backend_fixture")


def test_operational_family_and_availability_masks_survive_batching() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    backend = local_line_backend(2)
    probe = basis_probe_action(2, ("X", "Y"), circuit=circuit, shots=16, seed=2)
    _, routing = routing_transpilation_action(circuit, backend, seed=2)
    invalid = basis_probe_action(2, ("X",), circuit=circuit)
    first = build_operational_action_tensor_batch((probe, invalid), graph_index=0)
    second = build_operational_action_tensor_batch((routing,), graph_index=0)
    combined = collate_operational_action_tensor_batches((first, second))
    assert combined.candidate_family_ids.tolist().count(1) == 2
    assert combined.candidate_family_ids.tolist().count(2) == 1
    assert combined.model_candidates.candidate_available_mask.tolist().count(False) == 1
    assert combined.model_candidates.candidate_batch.tolist() == [0, 0, 1]
    assert not bool(combined.candidate_target_mask.any())
    assert not bool(combined.privileged_information_mask.any())
    unavailable = ~combined.model_candidates.candidate_available_mask
    assert not bool((combined.model_candidates.candidate_features[unavailable] != 0).any())
