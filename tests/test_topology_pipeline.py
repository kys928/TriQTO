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
from triqto.storage import ManifestReader, TopologyGroupRecordV1
from triqto.topology import (
    TopologyAuditConfig,
    build_topology_audit_result,
    load_topology_group_artifact,
    validate_topology_group_result,
    write_topology_dataset,
)


def build_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    phase7_root = tmp_path / "phase7"
    graph_root = tmp_path / "phase8"
    action_root = tmp_path / "phase9"
    generation_config = DatasetGenerationConfig(
        dataset_name="phase11-source",
        base_seed=111,
        circuit_specs=[
            CircuitGenerationSpec(
                family="bell",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=3,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            )
        ],
        store_statevectors=False,
        max_samples=6,
    )
    write_dataset(generate_dataset(generation_config), phase7_root)
    graph_result = convert_completed_dataset_to_graphs(
        phase7_root,
        GraphConversionConfig(include_supplemental_counts=False),
    )
    write_graph_dataset(graph_result, graph_root)
    action_config = ActionEngineConfig(
        candidate_magnitudes=(0.2,),
        max_candidates_per_sample=64,
        max_edits_per_action=16,
    )
    action_result = build_action_engine_result(
        phase7_root,
        graph_root,
        action_config,
    )
    write_action_dataset(action_result, action_root)
    return phase7_root, graph_root, action_root


def managed_snapshot(root: Path, marker_name: str):
    marker = json.loads((root / marker_name).read_text())
    return snapshot_managed_files(root, tuple(marker["managed_files"]))


def small_config(**overrides) -> TopologyAuditConfig:
    values = {
        "min_points": 3,
        "betti_grid_size": 12,
        "top_k_lifetimes": 3,
        "max_points_per_group": 128,
        "max_groups": 64,
        "max_statevector_amplitudes": 64,
    }
    values.update(overrides)
    return TopologyAuditConfig(**values)


def test_topology_pipeline_builds_aligned_audits_deterministically(
    tmp_path: Path,
) -> None:
    phase7_root, graph_root, action_root = build_sources(tmp_path)
    before = (
        managed_snapshot(phase7_root, "dataset_complete.json"),
        managed_snapshot(graph_root, "graph_complete.json"),
        managed_snapshot(action_root, "action_complete.json"),
    )
    config = small_config()
    first = build_topology_audit_result(
        phase7_root,
        graph_root,
        action_root,
        config,
    )
    second = build_topology_audit_result(
        phase7_root,
        graph_root,
        action_root,
        config,
    )

    assert first.summary["topology_loss_weight"] == 0.0
    assert first.summary["topology_mode"] == "audit_and_feature_only"
    assert first.summary["latent_topology_available"] is False
    assert first.summary["source_immutability_verified"] is True
    assert first.summary["group_kind_counts"] == {
        "action_neighborhood": 3,
        "family_qubit_cohort": 1,
        "family_qubit_distortion_cohort": 1,
    }
    assert len(first.groups) == 5
    assert [group.topology_group_id for group in first.groups] == [
        group.topology_group_id for group in second.groups
    ]
    assert [group.content_hash for group in first.groups] == [
        group.content_hash for group in second.groups
    ]

    for group in first.groups:
        validate_topology_group_result(group, config)
        assert group.manifold_available_mask.tolist() == [True, True, True]
        assert set(group.persistence) == {"parameter", "hilbert", "born"}
        assert group.metadata["raw_statevectors_persisted"] is False
        assert group.topology_feature_values.size > 0
        assert group.alignment_feature_values.size > 0
        assert np.isfinite(group.topology_feature_values).all()
        assert np.isfinite(group.alignment_feature_values).all()
        assert group.parameter_distance_matrix.shape == (
            group.point_ids.size,
            group.point_ids.size,
        )
        assert np.allclose(group.parameter_distance_matrix, group.parameter_distance_matrix.T)

    after = (
        managed_snapshot(phase7_root, "dataset_complete.json"),
        managed_snapshot(graph_root, "graph_complete.json"),
        managed_snapshot(action_root, "action_complete.json"),
    )
    assert after == before


def test_hardware_masked_topology_omits_hilbert_without_fabrication(
    tmp_path: Path,
) -> None:
    phase7_root, graph_root, action_root = build_sources(tmp_path)
    config = small_config(
        group_kinds=("action_neighborhood",),
        include_hilbert=False,
    )
    result = build_topology_audit_result(
        phase7_root,
        graph_root,
        action_root,
        config,
    )
    assert result.summary["hilbert_group_count"] == 0
    for group in result.groups:
        assert group.manifold_available_mask.tolist() == [True, False, True]
        assert set(group.persistence) == {"parameter", "born"}
        assert np.array_equal(
            group.hilbert_distance_matrix,
            np.zeros_like(group.hilbert_distance_matrix),
        )
        assert group.metadata["hilbert_available"] is False
        assert group.metadata["latent_available"] is False


def test_topology_dataset_roundtrip_is_strict_and_immutable(tmp_path: Path) -> None:
    phase7_root, graph_root, action_root = build_sources(tmp_path)
    config = small_config(group_kinds=("action_neighborhood",))
    result = build_topology_audit_result(
        phase7_root,
        graph_root,
        action_root,
        config,
    )
    output = tmp_path / "phase11"
    written = write_topology_dataset(result, output)

    marker = json.loads((output / "topology_complete.json").read_text())
    assert marker["complete"] is True
    assert marker["group_count"] == len(result.groups)
    assert marker["topology_loss_weight"] == 0.0
    assert marker["managed_files"] == sorted(marker["managed_files"])
    assert set(marker["managed_files"]) == {
        path.relative_to(output).as_posix() for path in written.written_paths
    }

    reader = ManifestReader(output / "manifests")
    records = reader.read_typed_records(
        "topology_group_manifest",
        TopologyGroupRecordV1,
    )
    assert len(records) == len(result.groups)
    loaded_ids = set()
    for record in records:
        record.validate()
        group = load_topology_group_artifact(
            output / record.artifact_ref,
            config,
            record.content_hash,
        )
        loaded_ids.add(group.topology_group_id)
        with np.load(output / record.artifact_ref, allow_pickle=False) as payload:
            assert all("statevector" not in name for name in payload.files)
            assert all(payload[name].dtype.kind != "O" for name in payload.files)
    assert loaded_ids == {group.topology_group_id for group in result.groups}

    with pytest.raises(ValueError, match="content_hash does not match manifest"):
        load_topology_group_artifact(
            output / records[0].artifact_ref,
            config,
            "sha256:" + "0" * 64,
        )
    with pytest.raises(FileExistsError):
        write_topology_dataset(result, output)


def test_point_guardrail_fails_instead_of_subsampling(tmp_path: Path) -> None:
    phase7_root, graph_root, action_root = build_sources(tmp_path)
    config = small_config(
        group_kinds=("action_neighborhood",),
        max_points_per_group=3,
    )
    with pytest.raises(RuntimeError, match="max_points_per_group"):
        build_topology_audit_result(
            phase7_root,
            graph_root,
            action_root,
            config,
        )
