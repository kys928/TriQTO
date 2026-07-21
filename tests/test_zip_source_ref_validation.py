from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triqto.graph.utils import resolve_safe_file
from triqto.training_views.artifacts import _validate_source_refs


def _item(reference: str) -> SimpleNamespace:
    return SimpleNamespace(
        view_item_id="viewitem-test",
        arrays={
            "source_dataset_names": np.asarray(["phase9"]),
            "source_refs": np.asarray([reference]),
        },
    )


def test_training_view_source_ref_accepts_zip_member(tmp_path: Path) -> None:
    phase9 = tmp_path / "phase9"
    archive = phase9 / "artifacts" / "shards" / "action-shard-119.zip"
    archive.parent.mkdir(parents=True)
    member = "circuits/candidatecircuit-test.qpy"
    with ZipFile(archive, "w") as payload:
        payload.writestr(member, b"qpy-bytes")

    reference = f"artifacts/shards/action-shard-119.zip#{member}"
    _validate_source_refs(_item(reference), {"phase9": phase9})
    assert resolve_safe_file(phase9, reference, "source_ref") == archive.resolve()


def test_training_view_source_ref_rejects_missing_zip_member(tmp_path: Path) -> None:
    phase9 = tmp_path / "phase9"
    archive = phase9 / "artifacts" / "shards" / "action-shard-119.zip"
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, "w") as payload:
        payload.writestr("circuits/present.qpy", b"qpy-bytes")

    reference = "artifacts/shards/action-shard-119.zip#circuits/missing.qpy"
    with pytest.raises(FileNotFoundError, match="missing ZIP member"):
        _validate_source_refs(_item(reference), {"phase9": phase9})


def test_training_view_source_ref_rejects_archive_member_traversal(
    tmp_path: Path,
) -> None:
    phase9 = tmp_path / "phase9"
    archive = phase9 / "action.zip"
    phase9.mkdir()
    with ZipFile(archive, "w") as payload:
        payload.writestr("circuits/present.qpy", b"qpy-bytes")

    with pytest.raises(ValueError, match="traversal"):
        resolve_safe_file(
            phase9,
            "action.zip#../circuits/present.qpy",
            "source_ref",
        )


def test_training_view_source_ref_rejects_invalid_zip(tmp_path: Path) -> None:
    phase9 = tmp_path / "phase9"
    phase9.mkdir()
    (phase9 / "action.zip").write_bytes(b"not-a-zip")

    with pytest.raises(ValueError, match="invalid ZIP archive"):
        resolve_safe_file(
            phase9,
            "action.zip#circuits/present.qpy",
            "source_ref",
        )


def test_plain_source_ref_behavior_is_unchanged(tmp_path: Path) -> None:
    phase9 = tmp_path / "phase9"
    phase9.mkdir()
    target = phase9 / "manifest.json"
    target.write_text("{}")

    assert resolve_safe_file(phase9, "manifest.json", "source_ref") == target.resolve()
