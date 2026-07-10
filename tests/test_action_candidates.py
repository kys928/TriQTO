from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from triqto.actions import (
    ActionEngineConfig,
    generate_action_candidates,
    observed_two_qubit_edges,
    oracle_inverse_edits,
)
from triqto.storage import DatasetSampleRecord, DistortionRecord, GraphPairRecord

_HASH = "sha256:" + "0" * 64


def sample_record() -> DatasetSampleRecord:
    return DatasetSampleRecord(
        sample_id="sample_1",
        dataset_name="tiny",
        schema_version="triqto.phase7.v1",
        clean_circuit_id="clean_circuit",
        distorted_circuit_id="distorted_circuit",
        clean_run_id="clean_run",
        distorted_run_id="distorted_run",
        distortion_id="distortion_1",
        metric_id="metric_1",
        family="bell",
        n_qubits=2,
        repetition_index=0,
    )


def pair_record() -> GraphPairRecord:
    return GraphPairRecord(
        graph_pair_id="graphpair_1",
        sample_id="sample_1",
        clean_graph_id="clean_graph",
        distorted_graph_id="distorted_graph",
        distortion_id="distortion_1",
        metric_id="metric_1",
        pair_ref="artifacts/pairs/graphpair_1.npz",
        content_hash=_HASH,
    )


def test_candidate_generation_is_deterministic_and_deduplicates_oracle():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    distortion = DistortionRecord(
        distortion_id="distortion_1",
        circuit_id="clean_circuit",
        distortion_type="rx_overrotation",
        strength=0.2,
        affected_qubits=[0],
        affected_gates=["rx"],
        metadata={},
    )
    config = ActionEngineConfig(
        candidate_magnitudes=(0.2,),
        max_candidates_per_sample=64,
    )
    first = generate_action_candidates(
        sample=sample_record(),
        graph_pair_record=pair_record(),
        distortion=distortion,
        distorted_circuit=circuit,
        config=config,
    )
    second = generate_action_candidates(
        sample=sample_record(),
        graph_pair_record=pair_record(),
        distortion=distortion,
        distorted_circuit=circuit,
        config=config,
    )
    assert [candidate.action_id for candidate in first] == [
        candidate.action_id for candidate in second
    ]
    inverse = next(
        candidate
        for candidate in first
        if len(candidate.edits) == 1
        and candidate.edits[0].edit_type == "append_rx"
        and candidate.edits[0].qubits == (0,)
        and candidate.edits[0].magnitude == pytest.approx(-0.2)
    )
    assert inverse.generation_sources == (
        "blind_physics_prior",
        "oracle_inverse",
    )
    assert sum(not candidate.edits for candidate in first) == 1


def test_observed_edges_are_unique_and_do_not_invent_connectivity():
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    circuit.swap(2, 3)
    circuit.barrier(0, 2)
    assert observed_two_qubit_edges(circuit) == ((0, 1), (2, 3))


def test_oracle_inverse_covers_supported_unitary_distortions():
    phase = DistortionRecord(
        "d1", "c", "phase_rz_drift", 0.3, [0, 1], ["rz", "rz"], metadata={}
    )
    phase_edits = oracle_inverse_edits(phase, n_qubits=2, max_abs_angle=3.2)
    assert phase_edits is not None
    assert [edit.edit_type for edit in phase_edits] == [
        "append_rz",
        "append_rz",
    ]

    entangling = DistortionRecord(
        "d2",
        "c",
        "entangling_rzz_drift",
        0.4,
        [0, 1],
        ["rzz"],
        metadata={"selected_edges": [[0, 1]]},
    )
    rzz = oracle_inverse_edits(entangling, n_qubits=2, max_abs_angle=3.2)
    assert rzz is not None
    assert rzz[0].edit_type == "append_rzz"
    assert rzz[0].magnitude == pytest.approx(-0.4)

    mixed = DistortionRecord(
        "d3",
        "c",
        "mixed_unitary_drift",
        0.6,
        [0, 1],
        ["rz", "rz", "rx", "rx", "rzz"],
        metadata={"edges": [[0, 1]]},
    )
    mixed_edits = oracle_inverse_edits(mixed, n_qubits=2, max_abs_angle=3.2)
    assert mixed_edits is not None
    assert [edit.edit_type for edit in mixed_edits] == [
        "append_rzz",
        "append_rx",
        "append_rx",
        "append_rz",
        "append_rz",
    ]


def test_marker_only_distortion_has_no_circuit_oracle():
    marker = DistortionRecord(
        "d",
        "c",
        "readout_bitflip_marker",
        0.1,
        [0],
        [],
        metadata={"marker_only": True},
    )
    assert oracle_inverse_edits(marker, n_qubits=1, max_abs_angle=3.2) is None


def test_candidate_guardrail_raises_instead_of_truncating():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    distortion = DistortionRecord(
        "distortion_1",
        "clean_circuit",
        "rx_overrotation",
        0.2,
        [0],
        ["rx"],
        metadata={},
    )
    config = ActionEngineConfig(
        candidate_magnitudes=(0.2,),
        max_candidates_per_sample=1,
    )
    with pytest.raises(ValueError, match="exceeding"):
        generate_action_candidates(
            sample=sample_record(),
            graph_pair_record=pair_record(),
            distortion=distortion,
            distorted_circuit=circuit,
            config=config,
        )
