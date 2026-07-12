from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triqto.actions import (
    ActionEngineConfig,
    build_action_engine_result,
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
    ManifestReader,
    TrainingViewDefinitionRecordV1,
    TrainingViewItemRecordV1,
)
from triqto.topology import (
    TopologyAuditConfig,
    build_topology_audit_result,
    write_topology_dataset,
)
from triqto.training_views import (
    TrainingViewConfig,
    build_training_view_result,
    load_training_view_item_artifact,
    validate_training_view_item,
    write_training_view_dataset,
)


def build_sources(
    tmp_path: Path,
    *,
    store_statevectors: bool,
) -> tuple[Path, Path, Path, Path]:
    phase7_root = tmp_path / "phase7"
    graph_root = tmp_path / "phase8"
    action_root = tmp_path / "phase9"
    topology_root = tmp_path / "phase11"
    generation_config = DatasetGenerationConfig(
        dataset_name="phase12-source",
        base_seed=1212,
        circuit_specs=[
            CircuitGenerationSpec(
                family="bell",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=4,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            )
        ],
        store_statevectors=store_statevectors,
        max_samples=8,
    )
    write_dataset(generate_dataset(generation_config), phase7_root)
    graph_result = convert_completed_dataset_to_graphs(
        phase7_root,
        GraphConversionConfig(include_supplemental_counts=False),
    )
    write_graph_dataset(graph_result, graph_root)
    action_result = build_action_engine_result(
        phase7_root,
        graph_root,
        ActionEngineConfig(
            candidate_magnitudes=(0.2,),
            max_candidates_per_sample=64,
            max_edits_per_action=16,
        ),
    )
    write_action_dataset(action_result, action_root)
    topology_result = build_topology_audit_result(
        phase7_root,
        graph_root,
        action_root,
        TopologyAuditConfig(
            min_points=3,
            betti_grid_size=8,
            top_k_lifetimes=2,
            max_points_per_group=128,
            max_groups=64,
            max_statevector_amplitudes=64,
        ),
    )
    write_topology_dataset(topology_result, topology_root)
    return phase7_root, graph_root, action_root, topology_root


def managed_snapshot(root: Path, marker_name: str):
    marker = json.loads((root / marker_name).read_text())
    return snapshot_managed_files(root, tuple(marker["managed_files"]))


def small_config(**overrides) -> TrainingViewConfig:
    values = {
        "split_seed": 19,
        "max_items": 1000,
        "max_candidates_per_item": 128,
        "max_source_refs_per_item": 1024,
    }
    values.update(overrides)
    return TrainingViewConfig(**values)


def test_full_training_view_pipeline_is_deterministic_and_leakage_safe(
    tmp_path: Path,
) -> None:
    roots = build_sources(tmp_path, store_statevectors=True)
    phase7_root, graph_root, action_root, topology_root = roots
    before = (
        managed_snapshot(phase7_root, "dataset_complete.json"),
        managed_snapshot(graph_root, "graph_complete.json"),
        managed_snapshot(action_root, "action_complete.json"),
        managed_snapshot(topology_root, "topology_complete.json"),
    )
    config = small_config()
    first = build_training_view_result(*roots, config)
    second = build_training_view_result(*roots, config)

    assert first.summary["training_executed"] is False
    assert first.summary["model_present"] is False
    assert first.summary["hardware_data_present"] is False
    assert first.summary["topology_loss_weight"] == 0.0
    assert first.summary["clean_circuit_grouped_split"] is True
    assert first.summary["born_prediction_input_target_leakage_blocked"] is True
    assert first.summary["task_item_counts"] == {
        "diagnosis": 4,
        "action_ranking": 4,
        "born_prediction": 4,
        "hilbert_to_born": 4,
        "topology_audit": 6,
        "joint_multitask": 4,
        "hardware_masked": 4,
    }
    assert len(first.items) == 30
    assert [item.view_item_id for item in first.items] == [
        item.view_item_id for item in second.items
    ]
    assert [item.content_hash for item in first.items] == [
        item.content_hash for item in second.items
    ]

    split_by_group: dict[str, str] = {}
    for item in first.items:
        validate_training_view_item(item, config)
        if item.split != "audit_only":
            assert split_by_group.setdefault(item.split_group_id, item.split) == item.split
        if item.task == "born_prediction":
            assert "born_target_probabilities" in item.arrays
            assert not any(name.startswith("born_input_") for name in item.arrays)
        if item.task == "action_ranking":
            assert item.metadata["rollout_artifacts_are_target_provenance_only"] is True
            assert item.arrays["action_candidate_ids"].tolist() == sorted(
                item.arrays["action_candidate_ids"].tolist()
            )
        if item.task == "topology_audit" and item.metadata["cross_split_group"]:
            assert item.split == "audit_only"
        if item.task == "hardware_masked":
            assert item.hilbert_available_mask is False
            assert item.metadata["hardware_data"] is False
            rows = list(
                zip(
                    item.arrays["source_dataset_names"].tolist(),
                    item.arrays["source_usage_names"].tolist(),
                    strict=True,
                )
            )
            assert ("phase7", "input") not in rows
            assert not any(name.startswith("topology_") for name in item.arrays)

    after = (
        managed_snapshot(phase7_root, "dataset_complete.json"),
        managed_snapshot(graph_root, "graph_complete.json"),
        managed_snapshot(action_root, "action_complete.json"),
        managed_snapshot(topology_root, "topology_complete.json"),
    )
    assert after == before


def test_hilbert_view_can_be_empty_and_joint_masks_it(tmp_path: Path) -> None:
    roots = build_sources(tmp_path, store_statevectors=False)
    config = small_config()
    result = build_training_view_result(*roots, config)
    assert result.summary["task_item_counts"]["hilbert_to_born"] == 0
    joint = [item for item in result.items if item.task == "joint_multitask"]
    hardware = [item for item in result.items if item.task == "hardware_masked"]
    assert joint and hardware
    assert all(item.hilbert_available_mask is False for item in joint)
    assert all(item.hilbert_available_mask is False for item in hardware)
    assert all(
        not any(
            dataset == "phase7" and usage == "input"
            for dataset, usage in zip(
                item.arrays["source_dataset_names"].tolist(),
                item.arrays["source_usage_names"].tolist(),
                strict=True,
            )
        )
        for item in hardware
    )


def test_training_view_dataset_roundtrip_and_immutable_publication(
    tmp_path: Path,
) -> None:
    roots = build_sources(tmp_path, store_statevectors=True)
    config = small_config(
        tasks=(
            "diagnosis",
            "action_ranking",
            "born_prediction",
            "hilbert_to_born",
            "topology_audit",
            "hardware_masked",
        )
    )
    result = build_training_view_result(*roots, config)
    output = tmp_path / "phase12"
    written = write_training_view_dataset(result, output)
    marker = json.loads((output / "training_view_complete.json").read_text())
    assert marker["complete"] is True
    assert marker["view_count"] == 6
    assert marker["item_count"] == len(result.items)
    assert marker["topology_loss_weight"] == 0.0
    assert marker["managed_files"] == sorted(marker["managed_files"])
    assert set(marker["managed_files"]) == {
        path.relative_to(output).as_posix() for path in written.written_paths
    }

    reader = ManifestReader(output / "manifests")
    definitions = reader.read_typed_records(
        "training_view_manifest",
        TrainingViewDefinitionRecordV1,
    )
    records = reader.read_typed_records(
        "training_item_manifest",
        TrainingViewItemRecordV1,
    )
    assert len(definitions) == 6
    assert len(records) == len(result.items)
    loaded_ids = set()
    for record in records:
        record.validate()
        item = load_training_view_item_artifact(
            output / record.artifact_ref,
            config,
            record.content_hash,
        )
        loaded_ids.add(item.view_item_id)
        with np.load(output / record.artifact_ref, allow_pickle=False) as payload:
            assert all(payload[name].dtype.kind != "O" for name in payload.files)
    assert loaded_ids == {item.view_item_id for item in result.items}
    with pytest.raises(ValueError, match="content_hash does not match manifest"):
        load_training_view_item_artifact(
            output / records[0].artifact_ref,
            config,
            "sha256:" + "0" * 64,
        )
    with pytest.raises(FileExistsError):
        write_training_view_dataset(result, output)


def test_item_guardrail_fails_instead_of_truncating(tmp_path: Path) -> None:
    roots = build_sources(tmp_path, store_statevectors=False)
    config = small_config(max_items=3)
    with pytest.raises(RuntimeError, match="max_items"):
        build_training_view_result(*roots, config)
