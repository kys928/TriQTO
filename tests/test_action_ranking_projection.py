from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triqto.actions.artifacts import _ROLLOUT_ARRAY_NAMES, _ROLLOUT_METADATA_NAME
from triqto.actions.constants import ROLLOUT_ARTIFACT_SCHEMA_VERSION
from triqto.phase15_6 import action_ranking_projection as projection


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = BytesIO()
    np.save(handle, array, allow_pickle=False)
    return handle.getvalue()


def _metadata_payload() -> dict[str, object]:
    return {
        "artifact_schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
        "rollout_id": "rollout-1",
        "action_id": "action-1",
        "sample_id": "sample-1",
        "graph_pair_id": "pair-1",
        "candidate_circuit_id": "candidate-circuit-1",
        "clean_target_run_id": "clean-run-1",
        "scientific_config_id": "science-1",
        "rank": 1,
        "reward": 0.75,
        "risk_score": 0.25,
        "dominates_baseline": True,
        "primary_metric_nonworsening": True,
        "selected": True,
        "depth_delta": -1,
        "gate_delta": -2,
        "metadata": {},
        "content_hash": "sha256:" + "3" * 64,
    }


def test_metadata_projection_does_not_decode_unused_rollout_arrays() -> None:
    metadata_bytes = json.dumps(
        _metadata_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    metadata_array = np.frombuffer(metadata_bytes, dtype=np.uint8).copy()
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        for name in _ROLLOUT_ARRAY_NAMES:
            # These deliberately are not valid NPY payloads.  The metadata-only reader
            # must succeed without attempting to decode them.
            archive.writestr(f"{name}.npy", b"unused-array-payload")
        archive.writestr(
            f"{_ROLLOUT_METADATA_NAME}.npy",
            _npy_bytes(metadata_array),
        )
    archive_bytes.seek(0)
    with ZipFile(archive_bytes, "r") as archive:
        loaded = projection._decode_rollout_metadata_member(archive)
    assert loaded == _metadata_payload()


def test_projected_builder_emits_canonical_action_ranking_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edit = SimpleNamespace(edit_type="rz", magnitude=0.125, qubits=(1,))
    projected = projection.ActionRankingProjection(
        action_id="action-1",
        edits=(edit,),
        generation_sources=("oracle_inverse",),
        risk_score=0.25,
        rank=1,
        reward=0.75,
        selected=True,
        dominates_baseline=True,
        primary_metric_nonworsening=True,
        depth_delta=-1,
        gate_delta=-2,
        circuit_ref="artifacts/shards/a.zip#circuits/c.qpy",
        action_ref="artifacts/shards/a.zip#actions/a.json",
        rollout_ref="artifacts/shards/a.zip#rollouts/r.npz",
    )
    monkeypatch.setattr(
        projection,
        "_projection_rows",
        lambda _action, sample_id: [projected] if sample_id == "sample-1" else [],
    )
    monkeypatch.setattr(
        projection,
        "graph_structure_arrays",
        lambda _graph: {"graph_sentinel": np.asarray([7], dtype=np.int64)},
    )

    captured: dict[str, object] = {}

    def fake_make_training_item(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(projection, "make_training_item", fake_make_training_item)
    sample = SimpleNamespace(
        sample_id="sample-1",
        metadata={
            "identifiability_status": "identified",
            "identifiability_reason": "test",
            "diagnosis_supervision_mask": True,
            "action_supervision_mask": True,
        },
    )
    pair = SimpleNamespace(
        graph_pair_id="pair-1",
        distorted_graph_id="graph-1",
        pair_ref="pairs/pair-1.json",
    )
    context = SimpleNamespace(
        sources=SimpleNamespace(
            phase7=SimpleNamespace(samples=[sample]),
            graph=SimpleNamespace(graphs_by_id={"graph-1": object()}),
            action=SimpleNamespace(is_lazy=True, db_path="index.sqlite3", root="phase9"),
        ),
        pair_records_by_sample_id={"sample-1": pair},
        graph_records_by_id={
            "graph-1": SimpleNamespace(graph_ref="graphs/graph-1.npz")
        },
        config=SimpleNamespace(
            max_candidates_per_item=8,
            max_source_refs_per_item=32,
        ),
        view_ids={"action_ranking": "view-1"},
        dataset_id="dataset-1",
        sample_splits={"sample-1": "train"},
        sample_split_groups={"sample-1": "split-group-1"},
    )

    items = projection.build_action_ranking_items_projected(context)
    assert len(items) == 1
    arrays = captured["arrays"]
    assert isinstance(arrays, dict)
    np.testing.assert_array_equal(arrays["action_candidate_ids"], np.asarray(["action-1"]))
    np.testing.assert_array_equal(
        arrays["action_candidate_features"],
        np.asarray([[1.0, 0.25, -1.0, -2.0, 0.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(arrays["action_edit_ptr"], np.asarray([0, 1]))
    np.testing.assert_array_equal(arrays["action_edit_types"], np.asarray(["rz"]))
    np.testing.assert_array_equal(arrays["action_edit_qubits"], np.asarray([1]))
    np.testing.assert_array_equal(arrays["action_target_rank"], np.asarray([1]))
    np.testing.assert_array_equal(
        arrays["action_privileged_oracle_mask"], np.asarray([True])
    )
    assert captured["source_refs"] == [
        ("phase8", "provenance", "graphs/graph-1.npz"),
        ("phase8", "provenance", "pairs/pair-1.json"),
        ("phase9", "input", projected.circuit_ref),
        ("phase9", "provenance", projected.action_ref),
        ("phase9", "target_provenance", projected.rollout_ref),
    ]
    assert captured["privileged_target_available"] is True


def test_projection_falls_back_for_nonlazy_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = [object()]
    monkeypatch.setattr(
        projection,
        "_canonical_action_ranking_builder",
        lambda context: sentinel if context == "context" else [],
    )
    context = SimpleNamespace(sources=SimpleNamespace(action=SimpleNamespace(is_lazy=False)))
    assert projection.build_action_ranking_items_projected(context) is sentinel


def test_installer_replaces_resumable_dispatcher() -> None:
    from triqto.phase15_6 import resumable_phase12

    original = resumable_phase12.build_action_ranking_items
    try:
        projection.install_action_ranking_projection()
        assert (
            resumable_phase12.build_action_ranking_items
            is projection.build_action_ranking_items_projected
        )
    finally:
        resumable_phase12.build_action_ranking_items = original
