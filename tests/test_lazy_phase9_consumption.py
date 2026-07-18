from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from triqto.actions.lazy_dataset import _connect, _create_schema, _finalize_index
from triqto.topology import TopologyAuditConfig
from triqto.topology.topology_groups import build_topology_group_specs
from triqto.training_views.config import TrainingViewConfig
from triqto.training_views.context import build_view_context


class _BombMapping:
    def get(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("full Phase 9 rollout mapping must not be touched")


class _BombSequence:
    def __iter__(self):
        raise AssertionError("global Phase 9 record sequence must not be iterated")


class _LazyActionForGroups:
    is_lazy = True
    rollouts_by_sample_id = _BombMapping()

    def action_ids_for_sample(self, sample_id: str) -> tuple[str, ...]:
        assert sample_id == "sample-1"
        return ("action-b", "action-a")


def test_topology_group_planning_uses_lazy_action_ids_without_hydration() -> None:
    sample = SimpleNamespace(
        sample_id="sample-1",
        family="bell",
        n_qubits=2,
        distortion_id="distortion-1",
    )
    sources = SimpleNamespace(
        phase7=SimpleNamespace(samples=[sample], distortions=[]),
        action=_LazyActionForGroups(),
    )
    config = TopologyAuditConfig(
        group_kinds=("action_neighborhood",),
        min_points=2,
        max_points_per_group=8,
    )
    specs, skipped = build_topology_group_specs(sources, config)
    assert skipped == {}
    assert len(specs) == 1
    assert specs[0].point_ids == ("action-a", "action-b")
    assert specs[0].metadata["action_source_mode"] == "lazy_per_sample"


def test_phase12_context_preserves_lazy_record_mappings(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = SimpleNamespace(sample_id="sample-1")
    candidate_map = {"action-1": SimpleNamespace(action_id="action-1")}
    rollout_map = {"action-1": SimpleNamespace(action_id="action-1")}
    action_to_sample = {"action-1": "sample-1"}
    action = SimpleNamespace(
        is_lazy=True,
        candidate_records=_BombSequence(),
        rollout_records=_BombSequence(),
        candidate_records_by_action_id=candidate_map,
        rollout_records_by_action_id=rollout_map,
        action_to_sample=action_to_sample,
    )
    sources = SimpleNamespace(
        phase7=SimpleNamespace(
            samples=[sample],
            distortions=[],
            simulations=[],
            metrics=[],
        ),
        graph=SimpleNamespace(graph_records=[], pair_records_by_sample_id={}),
        action=action,
        topology=SimpleNamespace(groups_by_id={}, records_by_id={}),
    )
    monkeypatch.setattr(
        "triqto.training_views.context.build_sample_split_maps",
        lambda _phase7, _config: (
            {"sample-1": "train"},
            {"sample-1": "split-group-1"},
        ),
    )
    context = build_view_context(
        sources,
        TrainingViewConfig(),
        "dataset-1",
        {"diagnosis": "view-1"},
    )
    assert context.candidate_records_by_action_id is candidate_map
    assert context.rollout_records_by_action_id is rollout_map
    assert context.action_to_sample is action_to_sample


def test_lazy_sql_index_validates_sample_rank_and_selection(tmp_path: Path) -> None:
    root = tmp_path / "index"
    root.mkdir()
    db = root / "action_index.sqlite3"
    with _connect(db) as connection:
        _create_schema(connection)
        connection.execute(
            "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "action-1",
                "sample-1",
                "pair-1",
                "source-circuit-1",
                "source-run-1",
                "distortion-1",
                "candidate-circuit-1",
                '["no_op"]',
                "artifacts/shards/action-shard-000.zip#actions/action-1.json",
                "artifacts/shards/action-shard-000.zip#circuits/candidate-circuit-1.qpy",
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
                0,
                1,
                0.0,
            ),
        )
        connection.execute(
            "INSERT INTO rollouts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "rollout-1",
                "action-1",
                "sample-1",
                "pair-1",
                "candidate-circuit-1",
                "clean-run-1",
                "scientific-config-1",
                "artifacts/shards/action-shard-000.zip#rollouts/rollout-1.npz",
                "sha256:" + "3" * 64,
                1,
                1.0,
                0.0,
                1,
                1,
                1,
            ),
        )
        connection.commit()
        _finalize_index(
            "Test",
            root,
            connection,
            candidate_count=1,
            rollout_count=1,
            source_sample_ids={"sample-1"},
        )
        assert connection.execute(
            "SELECT value FROM index_state WHERE key='ready'"
        ).fetchone()[0] == "1"
