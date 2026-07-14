from __future__ import annotations

import json
from pathlib import Path

import pytest

from triqto.actions import (
    ActionEngineConfig,
    action_config_to_dict,
    build_action_engine_result,
    load_action_artifact,
    load_candidate_circuit,
    load_rollout_artifact,
    write_action_dataset,
)
from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    generate_dataset,
    write_dataset,
)
from triqto.graph import (
    GraphConversionConfig,
    convert_completed_dataset_to_graphs,
    snapshot_managed_files,
    write_graph_dataset,
)
from triqto.storage import (
    ActionCandidateRecordV1,
    ActionRolloutRecord,
    ManifestReader,
)


def build_sources(tmp_path: Path) -> tuple[Path, Path]:
    phase7_root = tmp_path / "phase7"
    graph_root = tmp_path / "phase8"
    generation_config = DatasetGenerationConfig(
        dataset_name="phase9-source",
        base_seed=29,
        circuit_specs=[
            CircuitGenerationSpec(
                family="bell",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=1,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            ),
            DistortionSpec(
                name="readout_bitflip",
                kwargs={"probability": 0.1, "qubits": [0]},
            ),
        ],
        store_statevectors=False,
        max_samples=4,
    )
    write_dataset(generate_dataset(generation_config), phase7_root)
    graph_result = convert_completed_dataset_to_graphs(
        phase7_root,
        GraphConversionConfig(include_supplemental_counts=False),
    )
    write_graph_dataset(graph_result, graph_root)
    return phase7_root, graph_root


def managed_snapshot(root: Path, marker_name: str):
    marker = json.loads((root / marker_name).read_text())
    return snapshot_managed_files(root, tuple(marker["managed_files"]))


def action_config() -> ActionEngineConfig:
    return ActionEngineConfig(
        candidate_magnitudes=(0.2,),
        max_candidates_per_sample=64,
        max_edits_per_action=16,
        improvement_atol=1e-7,
    )


def test_end_to_end_action_engine_and_source_immutability(tmp_path):
    phase7_root, graph_root = build_sources(tmp_path)
    phase7_before = managed_snapshot(phase7_root, "dataset_complete.json")
    graph_before = managed_snapshot(graph_root, "graph_complete.json")

    result = build_action_engine_result(
        phase7_root,
        graph_root,
        action_config(),
    )

    assert result.summary["source_sample_count"] == 2
    assert result.summary["selected_action_count"] == 2
    assert result.summary["learned_policy_present"] is False
    assert result.summary["source_immutability_verified"] is True
    assert sum(rollout.selected for rollout in result.rollouts) == 2

    oracle_candidates = [
        candidate
        for candidate in result.candidates
        if "oracle_inverse" in candidate.generation_sources
    ]
    assert oracle_candidates
    oracle_action_ids = {candidate.action_id for candidate in oracle_candidates}
    oracle_rollouts = [
        rollout
        for rollout in result.rollouts
        if rollout.action_id in oracle_action_ids
    ]
    assert oracle_rollouts
    assert min(
        rollout.candidate_metric_values[0]
        for rollout in oracle_rollouts
    ) <= action_config().improvement_atol
    assert any(
        rollout.selected
        and rollout.candidate_metric_values[0]
        <= action_config().improvement_atol
        for rollout in result.rollouts
    )
    assert result.summary["selected_no_op_count"] >= 1

    assert managed_snapshot(phase7_root, "dataset_complete.json") == phase7_before
    assert managed_snapshot(graph_root, "graph_complete.json") == graph_before


def test_action_dataset_persistence_typed_roundtrip_and_immutable_root(tmp_path):
    phase7_root, graph_root = build_sources(tmp_path)
    config = action_config()
    result = build_action_engine_result(phase7_root, graph_root, config)
    output = tmp_path / "phase9"
    written = write_action_dataset(result, output)

    assert json.loads((output / "action_config.json").read_text()) == action_config_to_dict(config)
    marker = json.loads((output / "action_complete.json").read_text())
    assert marker["complete"] is True
    assert marker["candidate_count"] == len(result.candidates)
    assert marker["rollout_count"] == len(result.rollouts)
    assert marker["managed_files"] == sorted(marker["managed_files"])
    assert set(marker["managed_files"]) == {
        path.relative_to(output).as_posix() for path in written.written_paths
    }

    reader = ManifestReader(output / "manifests")
    candidate_records = reader.read_typed_records(
        "action_candidate_manifest", ActionCandidateRecordV1
    )
    rollout_records = reader.read_typed_records(
        "action_rollout_manifest", ActionRolloutRecord
    )
    assert len(candidate_records) == len(result.candidates)
    assert len(rollout_records) == len(result.rollouts)

    circuits = {}
    for record in candidate_records:
        record.validate()
        candidate = load_action_artifact(
            output / record.action_ref,
            config,
            record.content_hash,
        )
        circuit = load_candidate_circuit(
            output / record.circuit_ref,
            record.circuit_hash,
        )
        assert candidate.action_id == record.action_id
        circuits[record.candidate_circuit_id] = circuit
    for record in rollout_records:
        record.validate()
        rollout = load_rollout_artifact(
            output / record.rollout_ref,
            circuits[record.candidate_circuit_id],
            record.content_hash,
        )
        assert rollout.rollout_id == record.rollout_id

    with pytest.raises(FileExistsError):
        write_action_dataset(result, output)


def test_logical_reproduction_across_output_roots(tmp_path):
    phase7_root, graph_root = build_sources(tmp_path)
    first = build_action_engine_result(phase7_root, graph_root, action_config())
    second = build_action_engine_result(phase7_root, graph_root, action_config())
    assert [item.action_id for item in first.candidates] == [
        item.action_id for item in second.candidates
    ]
    assert [item.content_hash for item in first.rollouts] == [
        item.content_hash for item in second.rollouts
    ]
    write_action_dataset(first, tmp_path / "out-a")
    write_action_dataset(second, tmp_path / "out-b")
    first_marker = json.loads((tmp_path / "out-a" / "action_complete.json").read_text())
    second_marker = json.loads((tmp_path / "out-b" / "action_complete.json").read_text())
    assert first_marker == second_marker


def test_failed_publication_removes_staging_and_final_root(tmp_path, monkeypatch):
    phase7_root, graph_root = build_sources(tmp_path)
    result = build_action_engine_result(phase7_root, graph_root, action_config())
    output = tmp_path / "phase9"
    import triqto.actions.artifacts as artifacts

    def fail_rollout(*args, **kwargs):
        raise RuntimeError("injected rollout persistence failure")

    monkeypatch.setattr(artifacts, "save_rollout_artifact", fail_rollout)
    with pytest.raises(RuntimeError, match="injected"):
        write_action_dataset(result, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".phase9.staging-*"))


def test_output_must_not_be_nested_inside_source_roots(tmp_path):
    phase7_root, graph_root = build_sources(tmp_path)
    result = build_action_engine_result(phase7_root, graph_root, action_config())
    with pytest.raises(ValueError, match="Phase 7"):
        write_action_dataset(result, phase7_root / "phase9")
    with pytest.raises(ValueError, match="Phase 8"):
        write_action_dataset(result, graph_root / "phase9")
