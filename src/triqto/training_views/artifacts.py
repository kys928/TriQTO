"""Strict NPZ item artifacts and immutable publication for Phase 12 views."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np

from triqto.graph import snapshot_managed_files
from triqto.graph.utils import (
    json_copy,
    resolve_safe_file,
    strict_json_load,
    strict_json_loads,
    write_strict_json,
)
from triqto.storage import ManifestReader, ManifestWriter
from triqto.storage.training_view_schema import (
    TrainingViewDefinitionRecordV1,
    TrainingViewItemRecordV1,
)

from .config import (
    TrainingViewConfig,
    load_training_view_config,
    save_training_view_config,
)
from .constants import (
    TRAINING_ITEM_METADATA_ARRAY_NAME,
    TRAINING_VIEW_ARTIFACT_VERSION,
)
from .identities import training_view_item_content_hash
from .models import (
    TrainingViewBuildResult,
    TrainingViewItem,
    TrainingViewWriteResult,
)
from .validators import (
    validate_training_view_dataset_joins,
    validate_training_view_item,
)


def _json_bytes(payload: Mapping[str, Any]) -> np.ndarray:
    text = json.dumps(
        json_copy(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def _decode_json_bytes(array: np.ndarray, name: str) -> dict[str, Any]:
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional uint8 array")
    try:
        text = array.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    payload = strict_json_loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must decode to a JSON object")
    return dict(payload)


def _artifact_metadata(item: TrainingViewItem) -> dict[str, Any]:
    return {
        "artifact_version": TRAINING_VIEW_ARTIFACT_VERSION,
        "view_item_id": item.view_item_id,
        "training_view_id": item.training_view_id,
        "training_view_dataset_id": item.training_view_dataset_id,
        "task": item.task,
        "split": item.split,
        "split_group_id": item.split_group_id,
        "entity_id": item.entity_id,
        "input_groups": list(item.input_groups),
        "target_groups": list(item.target_groups),
        "hilbert_available_mask": item.hilbert_available_mask,
        "topology_available_mask": item.topology_available_mask,
        "privileged_target_available_mask": item.privileged_target_available_mask,
        "array_names": sorted(item.arrays),
        "metadata": item.metadata,
        "content_hash": training_view_item_content_hash(item),
    }


def save_training_view_item_artifact(
    item: TrainingViewItem,
    config: TrainingViewConfig,
    path: str | Path,
) -> Path:
    validate_training_view_item(item, config, require_hash=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **item.arrays,
        **{TRAINING_ITEM_METADATA_ARRAY_NAME: _json_bytes(_artifact_metadata(item))},
    )
    return target


def load_training_view_item_artifact(
    path: str | Path,
    config: TrainingViewConfig,
    expected_content_hash: str | None = None,
) -> TrainingViewItem:
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        if TRAINING_ITEM_METADATA_ARRAY_NAME not in payload.files:
            raise ValueError("Training item metadata array is missing")
        metadata = _decode_json_bytes(
            payload[TRAINING_ITEM_METADATA_ARRAY_NAME],
            TRAINING_ITEM_METADATA_ARRAY_NAME,
        )
        expected_metadata_keys = {
            "artifact_version",
            "view_item_id",
            "training_view_id",
            "training_view_dataset_id",
            "task",
            "split",
            "split_group_id",
            "entity_id",
            "input_groups",
            "target_groups",
            "hilbert_available_mask",
            "topology_available_mask",
            "privileged_target_available_mask",
            "array_names",
            "metadata",
            "content_hash",
        }
        if set(metadata) != expected_metadata_keys:
            raise ValueError("Training item artifact metadata-key mismatch")
        if metadata["artifact_version"] != TRAINING_VIEW_ARTIFACT_VERSION:
            raise ValueError("Unsupported training item artifact version")
        array_names = metadata["array_names"]
        if not isinstance(array_names, list) or any(
            not isinstance(name, str) or not name for name in array_names
        ):
            raise TypeError("Training item array_names must be a list of strings")
        if array_names != sorted(set(array_names)):
            raise ValueError("Training item array_names must be sorted and unique")
        actual = set(payload.files)
        expected = set(array_names) | {TRAINING_ITEM_METADATA_ARRAY_NAME}
        if actual != expected:
            raise ValueError(
                "Training item artifact array-name mismatch; "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        arrays = {name: payload[name].copy() for name in array_names}
    item_metadata = metadata["metadata"]
    if not isinstance(item_metadata, Mapping):
        raise TypeError("Training item metadata.metadata must be a mapping")
    item = TrainingViewItem(
        view_item_id=metadata["view_item_id"],
        training_view_id=metadata["training_view_id"],
        training_view_dataset_id=metadata["training_view_dataset_id"],
        task=metadata["task"],
        split=metadata["split"],
        split_group_id=metadata["split_group_id"],
        entity_id=metadata["entity_id"],
        input_groups=tuple(metadata["input_groups"]),
        target_groups=tuple(metadata["target_groups"]),
        arrays=arrays,
        hilbert_available_mask=metadata["hilbert_available_mask"],
        topology_available_mask=metadata["topology_available_mask"],
        privileged_target_available_mask=metadata[
            "privileged_target_available_mask"
        ],
        metadata=dict(item_metadata),
        content_hash=metadata["content_hash"],
    )
    validate_training_view_item(item, config, require_hash=True)
    if expected_content_hash is not None and item.content_hash != expected_content_hash:
        raise ValueError("Training item content_hash does not match manifest")
    return item


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _verify_result_sources(result: TrainingViewBuildResult) -> None:
    checks = (
        ("Phase 7", result.phase7_source_root, result.phase7_snapshot),
        ("Phase 8", result.graph_source_root, result.graph_snapshot),
        ("Phase 9", result.action_source_root, result.action_snapshot),
        ("Phase 11", result.topology_source_root, result.topology_snapshot),
    )
    for name, root, expected in checks:
        actual = snapshot_managed_files(
            root,
            tuple(entry.reference for entry in expected.entries),
        )
        if actual != expected:
            raise RuntimeError(f"{name} managed source files changed during Phase 12")


def _validate_source_refs(
    item: TrainingViewItem,
    source_roots: Mapping[str, Path],
) -> None:
    for dataset, reference in zip(
        item.arrays["source_dataset_names"].tolist(),
        item.arrays["source_refs"].tolist(),
        strict=True,
    ):
        root = source_roots.get(str(dataset))
        if root is None:
            raise ValueError(f"No root configured for source dataset {dataset!r}")
        resolve_safe_file(root, str(reference), f"Training item {item.view_item_id} source_ref")


def write_training_view_dataset(
    result: TrainingViewBuildResult,
    output_root: str | Path,
) -> TrainingViewWriteResult:
    """Publish a fully validated Phase 12 dataset into a fresh immutable root."""
    if not isinstance(result, TrainingViewBuildResult):
        raise TypeError("result must be TrainingViewBuildResult")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Training-view output root already exists: {output}")
    resolved_output = output.resolve()
    source_roots = {
        "phase7": Path(result.phase7_source_root),
        "phase8": Path(result.graph_source_root),
        "phase9": Path(result.action_source_root),
        "phase11": Path(result.topology_source_root),
    }
    for source_name, source_root in source_roots.items():
        resolved_source = source_root.resolve()
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError(
                f"Training-view output root must not be inside {source_name} source root"
            )
    _verify_result_sources(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected existing staging directory: {staging}")
    try:
        (staging / "manifests").mkdir(parents=True)
        (staging / "artifacts" / "items").mkdir(parents=True)
        managed: list[str] = []
        save_training_view_config(result.config, staging / "training_view_config.json")
        managed.append("training_view_config.json")
        write_strict_json(staging / "training_view_summary.json", result.summary)
        managed.append("training_view_summary.json")
        for item in result.items:
            reference = f"artifacts/items/{item.view_item_id}.npz"
            _validate_source_refs(item, source_roots)
            save_training_view_item_artifact(item, result.config, staging / reference)
            managed.append(reference)
        writer = ManifestWriter(staging / "manifests")
        writer.write_records(
            "training_view_manifest",
            result.definition_records,
            overwrite=False,
        )
        managed.append("manifests/training_view_manifest.parquet")
        writer.write_records(
            "training_item_manifest",
            result.item_records,
            overwrite=False,
        )
        managed.append("manifests/training_item_manifest.parquet")

        persisted_config = load_training_view_config(
            staging / "training_view_config.json"
        )
        if persisted_config != result.config:
            raise ValueError("Persisted training-view config mismatch")
        reader = ManifestReader(staging / "manifests")
        definitions = reader.read_typed_records(
            "training_view_manifest",
            TrainingViewDefinitionRecordV1,
        )
        records = reader.read_typed_records(
            "training_item_manifest",
            TrainingViewItemRecordV1,
        )
        loaded: dict[str, TrainingViewItem] = {}
        for record in records:
            record.validate()
            item = load_training_view_item_artifact(
                resolve_safe_file(
                    staging,
                    record.artifact_ref,
                    f"TrainingViewItemRecordV1 {record.view_item_id}.artifact_ref",
                ),
                persisted_config,
                record.content_hash,
            )
            _validate_source_refs(item, source_roots)
            if item.view_item_id in loaded:
                raise ValueError(f"Duplicate persisted training item {item.view_item_id}")
            loaded[item.view_item_id] = item
        validate_training_view_dataset_joins(
            definitions,
            records,
            items_by_id=loaded,
            config=persisted_config,
        )

        if len(set(managed)) != len(managed):
            raise ValueError("Managed Phase 12 file inventory contains duplicates")
        expected_before_marker = set(managed)
        actual_before_marker = _relative_file_set(staging)
        if actual_before_marker != expected_before_marker:
            raise ValueError(
                "Staging training-view inventory mismatch; "
                f"missing={sorted(expected_before_marker - actual_before_marker)}, "
                f"unexpected={sorted(actual_before_marker - expected_before_marker)}"
            )
        managed_files = tuple(sorted([*managed, "training_view_complete.json"]))
        completion = {
            "complete": True,
            "source_scientific_generation_id": result.source_scientific_generation_id,
            "graph_conversion_id": result.graph_conversion_id,
            "action_engine_id": result.action_engine_id,
            "topology_audit_id": result.topology_audit_id,
            "training_view_dataset_id": result.training_view_dataset_id,
            "operational_config_id": result.operational_config_id,
            "training_view_schema_id": result.training_view_schema_id,
            "view_count": len(result.definitions),
            "item_count": len(result.items),
            "phase7_snapshot_hash": result.phase7_snapshot.aggregate_sha256,
            "graph_snapshot_hash": result.graph_snapshot.aggregate_sha256,
            "action_snapshot_hash": result.action_snapshot.aggregate_sha256,
            "topology_snapshot_hash": result.topology_snapshot.aggregate_sha256,
            "topology_loss_weight": 0.0,
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "training_view_complete.json", completion)
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed training-view inventory does not match staging")
        if strict_json_load(staging / "training_view_complete.json") != completion:
            raise ValueError("training_view_complete.json content mismatch")
        _verify_result_sources(result)
        if output.exists():
            raise FileExistsError(
                f"Training-view output root appeared during publication: {output}"
            )
        os.replace(staging, output)
        manifest_paths = (
            output / "manifests" / "training_view_manifest.parquet",
            output / "manifests" / "training_item_manifest.parquet",
        )
        artifact_paths = tuple(
            sorted(
                [
                    output / reference
                    for reference in managed_files
                    if reference.startswith("artifacts/")
                ],
                key=lambda path: path.as_posix(),
            )
        )
        written_paths = tuple(
            sorted(
                [output / reference for reference in managed_files],
                key=lambda path: path.as_posix(),
            )
        )
        return TrainingViewWriteResult(
            output_root=output,
            training_view_complete_path=output / "training_view_complete.json",
            manifest_paths=manifest_paths,
            artifact_paths=artifact_paths,
            written_paths=written_paths,
            managed_files=managed_files,
            view_count=len(result.definitions),
            item_count=len(result.items),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "load_training_view_item_artifact",
    "save_training_view_item_artifact",
    "write_training_view_dataset",
]
