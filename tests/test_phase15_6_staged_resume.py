from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

from triqto.phase15_6.resumable import (
    commit_checkpoint_artifact,
    load_checkpoint_artifact,
    prepare_checkpoint_root,
)
from triqto.phase15_6.resumable_phase11 import (
    _distance_arrays,
    _load_distances,
    _load_persistence,
    _load_point_cloud,
    _persistence_arrays,
    _point_cloud_arrays,
)
from triqto.phase15_6.resumable_phase12 import _shard_index
from triqto.topology.config import TopologyAuditConfig
from triqto.topology.models import PersistenceSummary, TopologyPointCloudGroup
from triqto.topology.point_clouds import build_point_cloud_group
from triqto.topology.topology_groups import TopologyGroupSpec


def test_checkpoint_strict_reuse_and_repair_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    artifact = root / "units" / "u1" / "stage.txt"
    marker = root / "units" / "u1" / "stage.json"
    identity = {"source": "stable", "config": 1}

    committed = commit_checkpoint_artifact(
        phase="phase-test",
        unit_id="u1",
        stage="stage",
        artifact_path=artifact,
        marker_path=marker,
        identity=identity,
        writer=lambda path: path.write_text("payload", encoding="utf-8"),
        validator=lambda path: path.read_text(encoding="utf-8"),
    )
    assert committed == "payload"

    loaded = load_checkpoint_artifact(
        root=root,
        phase="phase-test",
        unit_id="u1",
        stage="stage",
        artifact_path=artifact,
        marker_path=marker,
        identity=identity,
        resume_mode="strict",
        loader=lambda path, _marker: path.read_text(encoding="utf-8"),
    )
    assert loaded == "payload"

    artifact.write_text("corrupt", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact SHA-256 mismatch"):
        load_checkpoint_artifact(
            root=root,
            phase="phase-test",
            unit_id="u1",
            stage="stage",
            artifact_path=artifact,
            marker_path=marker,
            identity=identity,
            resume_mode="strict",
            loader=lambda path, _marker: path.read_text(encoding="utf-8"),
        )

    repaired = load_checkpoint_artifact(
        root=root,
        phase="phase-test",
        unit_id="u1",
        stage="stage",
        artifact_path=artifact,
        marker_path=marker,
        identity=identity,
        resume_mode="repair",
        loader=lambda path, _marker: path.read_text(encoding="utf-8"),
    )
    assert repaired is None
    assert not artifact.exists()
    assert not marker.exists()
    assert any((root / "quarantine").iterdir())


def test_resume_off_clears_checkpoint_root(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    prepared = prepare_checkpoint_root(root, "off")
    assert prepared == root
    assert prepared.is_dir()
    assert list(prepared.iterdir()) == []


def test_phase11_point_cloud_distance_and_persistence_roundtrip(
    tmp_path: Path,
) -> None:
    point_cloud = TopologyPointCloudGroup(
        group_kind="family_qubit_cohort",
        group_key="family=bell|n_qubits=2",
        point_ids=np.asarray(["s1", "s2"], dtype="<U2"),
        parameter_coordinate_names=np.asarray(["theta"], dtype="<U5"),
        parameter_coordinates=np.asarray([[0.0], [1.0]], dtype=np.float64),
        parameter_coordinate_mask=np.ones((2, 1), dtype=np.bool_),
        born_outcome_bitstrings=np.asarray(["00", "11"], dtype="<U2"),
        born_coordinates=np.asarray([[0.5, 0.5], [0.4, 0.6]], dtype=np.float64),
        statevectors=np.asarray(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            dtype=np.complex128,
        ),
        metadata={"sample_id": "s1"},
    )
    point_path = tmp_path / "point.npz"
    with point_path.open("wb") as handle:
        np.savez_compressed(handle, **_point_cloud_arrays(point_cloud))
    loaded_point = _load_point_cloud(point_path)
    assert loaded_point.group_key == point_cloud.group_key
    assert np.array_equal(loaded_point.statevectors, point_cloud.statevectors)

    matrices = {
        "parameter": np.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
        "hilbert": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        "born": np.asarray([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64),
    }
    distance_path = tmp_path / "distance.npz"
    with distance_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            **_distance_arrays(
                matrices,
                {
                    "normalization_scales": {
                        "parameter": 1.0,
                        "hilbert": 1.0,
                        "born": 1.0,
                    }
                },
            ),
        )
    loaded_matrices, loaded_metadata = _load_distances(distance_path)
    assert np.array_equal(loaded_matrices["born"], matrices["born"])
    assert loaded_metadata["normalization_scales"]["born"] == 1.0

    config = TopologyAuditConfig(betti_grid_size=4)
    summary = PersistenceSummary(
        manifold="born",
        diagrams={
            0: np.asarray([[0.0, 0.5]], dtype=np.float64),
            1: np.empty((0, 2), dtype=np.float64),
        },
        betti_curves={
            0: np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
            1: np.zeros(4, dtype=np.float64),
        },
        feature_names=np.asarray(["x"], dtype="<U1"),
        feature_values=np.asarray([1.0], dtype=np.float64),
        metadata={"engine": {"name": "test"}},
    )
    persistence_path = tmp_path / "persistence.npz"
    with persistence_path.open("wb") as handle:
        np.savez_compressed(handle, **_persistence_arrays(summary))
    loaded_summary = _load_persistence(persistence_path, "born", config)
    assert np.array_equal(loaded_summary.diagrams[0], summary.diagrams[0])
    assert loaded_summary.metadata == summary.metadata


def test_action_point_cloud_uses_sample_local_rollout_index() -> None:
    class FailOnGlobalScan(dict[str, object]):
        def values(self):  # type: ignore[override]
            raise AssertionError("global rollout scan must not be used")

    candidate = SimpleNamespace(action_id="a1", edits=())
    rollout = SimpleNamespace(
        action_id="a1",
        candidate_circuit_id="c1",
        outcome_bitstrings=np.asarray(["0"], dtype="<U1"),
        exact_probabilities=np.asarray([1.0], dtype=np.float64),
    )
    action = SimpleNamespace(
        candidates_by_id={"a1": candidate},
        circuits_by_id={"c1": QuantumCircuit(1)},
        rollouts_by_id=FailOnGlobalScan(),
        rollouts_by_sample_id={"s1": (rollout,)},
    )
    sources = SimpleNamespace(action=action)
    spec = TopologyGroupSpec(
        group_kind="action_neighborhood",
        group_key="sample=s1",
        point_ids=("a1",),
        metadata={"sample_id": "s1"},
    )

    group = build_point_cloud_group(
        spec,
        sources,
        TopologyAuditConfig(include_hilbert=False),
    )

    assert group.point_ids.tolist() == ["a1"]
    assert group.born_coordinates.tolist() == [[1.0]]
    assert group.metadata["rollout_index_scope"] == "sample_neighborhood_only"


def test_phase12_shard_assignment_is_deterministic() -> None:
    first = [_shard_index(f"sample-{index}", 256) for index in range(100)]
    second = [_shard_index(f"sample-{index}", 256) for index in range(100)]
    assert first == second
    assert all(0 <= value < 256 for value in first)
