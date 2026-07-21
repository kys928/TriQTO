from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from triqto.graph.utils import write_strict_json
from triqto.phase15_6 import resumable_phase12
from triqto.phase15_6.fast_phase12 import (
    _fast_existing_item_checkpoint,
    _validate_committed_shard_manifest,
    build_training_view_result_fast,
)
from triqto.phase15_6.resumable import commit_checkpoint_artifact


def test_fast_existing_item_reuse_verifies_marker_and_sha_without_npz_decode(
    tmp_path: Path,
) -> None:
    item = SimpleNamespace(
        view_item_id="viewitem_test",
        task="diagnosis",
        content_hash="sha256:" + "a" * 64,
    )
    dataset_id = "viewdataset_test"
    artifact, marker = resumable_phase12._item_paths(tmp_path, item.view_item_id)
    commit_checkpoint_artifact(
        phase="phase12",
        unit_id=item.view_item_id,
        stage="item_artifact",
        artifact_path=artifact,
        marker_path=marker,
        identity=resumable_phase12._item_identity(dataset_id, item),
        writer=lambda path: path.write_bytes(b"compressed-npz-placeholder"),
        validator=lambda path: path.read_bytes(),
        marker_metadata={
            "task": item.task,
            "entity_id": "sample-1",
            "content_hash": item.content_hash,
        },
    )

    assert _fast_existing_item_checkpoint(
        tmp_path,
        dataset_id,
        item,
        "strict",
    ) is True

    artifact.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="artifact SHA-256 mismatch"):
        _fast_existing_item_checkpoint(
            tmp_path,
            dataset_id,
            item,
            "strict",
        )


def test_shard_manifest_fast_validator_returns_in_memory_items(tmp_path: Path) -> None:
    path = tmp_path / "shard.json"
    payload = {"schema": "test", "items": [{"view_item_id": "v1"}]}
    write_strict_json(path, payload)
    items = [SimpleNamespace(view_item_id="v1")]

    assert _validate_committed_shard_manifest(path, payload, items) == items


def test_fast_phase12_worker_guard_runs_before_source_loading() -> None:
    with pytest.raises(ValueError, match="workers must not exceed 32"):
        build_training_view_result_fast(
            "phase7",
            "phase8",
            "phase9",
            "phase11",
            "checkpoints",
            workers=33,
        )
