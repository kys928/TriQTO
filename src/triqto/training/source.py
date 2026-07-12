"""Strict read-only loading of completed Phase 12 training-view datasets."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

from triqto.graph.utils import (
    ensure_sorted_unique_strings,
    require_mapping,
    require_nonblank,
    resolve_safe_file,
    strict_json_load,
)
from triqto.training_views.config import load_training_view_config

from .models import (
    CompletedTrainingViewDataset,
    ManagedFileEntry,
    ManagedFileSnapshot,
)

_MARKER_KEYS = {
    "complete",
    "source_scientific_generation_id",
    "graph_conversion_id",
    "action_engine_id",
    "topology_audit_id",
    "training_view_dataset_id",
    "operational_config_id",
    "training_view_schema_id",
    "view_count",
    "item_count",
    "phase7_snapshot_hash",
    "graph_snapshot_hash",
    "action_snapshot_hash",
    "topology_snapshot_hash",
    "topology_loss_weight",
    "managed_files",
}
_REQUIRED_MANAGED = {
    "training_view_config.json",
    "training_view_summary.json",
    "training_view_complete.json",
    "manifests/training_view_manifest.parquet",
    "manifests/training_item_manifest.parquet",
}
_ALLOWED_SPLITS = {"train", "validation", "test", "audit_only"}


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, f"sha256:{digest.hexdigest()}"


def snapshot_managed_files(root: str | Path, references: Sequence[str]) -> ManagedFileSnapshot:
    """Hash a sorted managed inventory without trusting file timestamps."""
    base = Path(root)
    normalized = ensure_sorted_unique_strings(references, "managed file references")
    entries: list[ManagedFileEntry] = []
    aggregate = hashlib.sha256()
    for reference in normalized:
        path = resolve_safe_file(base, reference, f"managed file {reference!r}")
        size, digest = _file_sha256(path)
        entry = ManagedFileEntry(reference=reference, size_bytes=size, sha256=digest)
        entries.append(entry)
        aggregate.update(reference.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return ManagedFileSnapshot(
        entries=tuple(entries),
        aggregate_sha256=f"sha256:{aggregate.hexdigest()}",
    )


def _actual_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def load_completed_training_view_dataset(
    training_view_root: str | Path,
) -> CompletedTrainingViewDataset:
    """Load, typed-read, artifact-read, and join-validate a Phase 12 root.

    Imports requiring PyArrow are intentionally delayed until this function is called,
    so pure model/trainer unit tests remain usable without the data-lake dependency.
    """
    root = Path(training_view_root)
    if not root.exists():
        raise FileNotFoundError(f"Phase 12 training-view root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Phase 12 training-view root is not a directory: {root}")

    marker_raw = strict_json_load(root / "training_view_complete.json")
    marker = dict(require_mapping(marker_raw, "training_view_complete.json"))
    if set(marker) != _MARKER_KEYS:
        raise ValueError(
            "training_view_complete.json key mismatch; "
            f"missing={sorted(_MARKER_KEYS - set(marker))}, "
            f"unexpected={sorted(set(marker) - _MARKER_KEYS)}"
        )
    if marker.get("complete") is not True:
        raise ValueError("training_view_complete.json complete must be exactly true")
    if marker.get("topology_loss_weight") != 0.0:
        raise ValueError("Phase 12 topology_loss_weight must remain exactly zero")
    for name in (
        "source_scientific_generation_id",
        "graph_conversion_id",
        "action_engine_id",
        "topology_audit_id",
        "training_view_dataset_id",
        "operational_config_id",
        "training_view_schema_id",
        "phase7_snapshot_hash",
        "graph_snapshot_hash",
        "action_snapshot_hash",
        "topology_snapshot_hash",
    ):
        require_nonblank(marker.get(name), f"training_view_complete.json {name}")

    managed_raw = marker.get("managed_files")
    if not isinstance(managed_raw, list):
        raise TypeError("training_view_complete.json managed_files must be a list")
    managed_files = ensure_sorted_unique_strings(managed_raw, "managed_files")
    missing = _REQUIRED_MANAGED - set(managed_files)
    if missing:
        raise ValueError(f"Phase 12 managed inventory misses {sorted(missing)}")
    actual = _actual_files(root)
    if actual != set(managed_files):
        raise ValueError(
            "Phase 12 managed inventory mismatch; "
            f"missing={sorted(set(managed_files) - actual)}, "
            f"unmanaged={sorted(actual - set(managed_files))}"
        )
    snapshot = snapshot_managed_files(root, managed_files)

    config = load_training_view_config(root / "training_view_config.json")
    summary_raw = strict_json_load(root / "training_view_summary.json")
    summary = dict(require_mapping(summary_raw, "training_view_summary.json"))
    if summary.get("training_view_dataset_id") != marker["training_view_dataset_id"]:
        raise ValueError("training_view_summary.json training_view_dataset_id mismatch")
    if summary.get("topology_loss_weight") != 0.0:
        raise ValueError("training_view_summary.json topology_loss_weight must be zero")
    if summary.get("training_executed") is not False:
        raise ValueError("Phase 12 summary must record training_executed=false")
    if summary.get("model_present") is not False:
        raise ValueError("Phase 12 summary must record model_present=false")

    from triqto.storage.manifest import ManifestReader
    from triqto.storage.training_view_schema import (
        TrainingViewDefinitionRecordV1,
        TrainingViewItemRecordV1,
    )
    from triqto.training_views.artifacts import load_training_view_item_artifact
    from triqto.training_views.validators import validate_training_view_dataset_joins

    reader = ManifestReader(root / "manifests")
    definitions = reader.read_typed_records(
        "training_view_manifest", TrainingViewDefinitionRecordV1
    )
    records = reader.read_typed_records(
        "training_item_manifest", TrainingViewItemRecordV1
    )
    if _strict_nonnegative_int(marker.get("view_count"), "view_count") != len(definitions):
        raise ValueError("Phase 12 view_count mismatch")
    if _strict_nonnegative_int(marker.get("item_count"), "item_count") != len(records):
        raise ValueError("Phase 12 item_count mismatch")

    records_by_id: dict[str, Any] = {}
    records_by_task_split_lists: dict[tuple[str, str], list[Any]] = {}
    items_by_id: dict[str, Any] = {}
    for definition in definitions:
        definition.validate()
        if definition.training_view_dataset_id != marker["training_view_dataset_id"]:
            raise ValueError("Definition training_view_dataset_id mismatch")
    for record in records:
        record.validate()
        if record.training_view_dataset_id != marker["training_view_dataset_id"]:
            raise ValueError("Item training_view_dataset_id mismatch")
        if record.split not in _ALLOWED_SPLITS:
            raise ValueError(f"Unsupported Phase 12 split {record.split!r}")
        if record.view_item_id in records_by_id:
            raise ValueError(f"Duplicate Phase 12 view_item_id {record.view_item_id}")
        artifact_path = resolve_safe_file(
            root, record.artifact_ref, f"Phase 12 item {record.view_item_id}.artifact_ref"
        )
        item = load_training_view_item_artifact(
            artifact_path,
            config,
            expected_content_hash=record.content_hash,
        )
        if (
            item.task != record.task
            or item.split != record.split
            or item.split_group_id != record.split_group_id
            or item.entity_id != record.entity_id
        ):
            raise ValueError(f"Phase 12 item/manifest semantic mismatch for {record.view_item_id}")
        records_by_id[record.view_item_id] = record
        items_by_id[record.view_item_id] = item
        records_by_task_split_lists.setdefault((record.task, record.split), []).append(record)

    validate_training_view_dataset_joins(
        definitions,
        records,
        items_by_id=items_by_id,
        config=config,
    )
    records_by_task_split = {
        key: tuple(sorted(values, key=lambda row: row.view_item_id))
        for key, values in records_by_task_split_lists.items()
    }

    graph_anchor_record_by_entity_id: dict[str, Any] = {}
    for record in sorted(records, key=lambda row: row.view_item_id):
        if record.task != "born_prediction":
            continue
        earlier = graph_anchor_record_by_entity_id.setdefault(record.entity_id, record)
        if earlier.split != record.split or earlier.split_group_id != record.split_group_id:
            raise ValueError(
                f"Entity {record.entity_id} has inconsistent born-prediction graph anchors"
            )

    return CompletedTrainingViewDataset(
        root=root,
        config=config,
        completion_marker=marker,
        summary=summary,
        definition_records=list(definitions),
        item_records=list(records),
        records_by_id=records_by_id,
        records_by_task_split=records_by_task_split,
        graph_anchor_record_by_entity_id=graph_anchor_record_by_entity_id,
        managed_files=managed_files,
        snapshot=snapshot,
    )


def verify_training_view_snapshot(dataset: CompletedTrainingViewDataset) -> None:
    actual = snapshot_managed_files(dataset.root, dataset.managed_files)
    if actual != dataset.snapshot:
        raise RuntimeError("Managed Phase 12 files changed during Phase 14")


def load_phase7_managed_snapshot(phase7_root: str | Path) -> ManagedFileSnapshot:
    root = Path(phase7_root)
    marker_raw = strict_json_load(root / "dataset_complete.json")
    marker = require_mapping(marker_raw, "dataset_complete.json")
    if marker.get("complete") is not True:
        raise ValueError("Phase 7 dataset_complete.json complete must be exactly true")
    managed = marker.get("managed_files")
    if not isinstance(managed, list):
        raise TypeError("Phase 7 dataset_complete.json managed_files must be a list")
    references = ensure_sorted_unique_strings(managed, "Phase 7 managed_files")
    if _actual_files(root) != set(references):
        raise ValueError("Phase 7 managed file inventory mismatch")
    return snapshot_managed_files(root, references)


__all__ = [
    "load_completed_training_view_dataset",
    "load_phase7_managed_snapshot",
    "snapshot_managed_files",
    "verify_training_view_snapshot",
]
