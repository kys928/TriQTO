"""Restartable Phase 12 artifact publication.

The logical training-view build remains deterministic. Expensive per-item NPZ serialization is
checkpointed independently, validated on reuse, and copied into one atomic final publication.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Callable

from triqto.graph.utils import resolve_safe_file, strict_json_load, write_strict_json
from triqto.storage import ManifestReader, ManifestWriter
from triqto.storage.training_view_schema import TrainingViewDefinitionRecordV1, TrainingViewItemRecordV1
from triqto.training_views.artifacts import (
    load_training_view_item_artifact,
    save_training_view_item_artifact,
)
from triqto.training_views.config import load_training_view_config, save_training_view_config
from triqto.training_views.models import TrainingViewBuildResult, TrainingViewWriteResult
from triqto.training_views.validators import validate_training_view_dataset_joins

ProgressCallback = Callable[[dict[str, Any]], None]


def _relative_file_set(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def write_training_view_dataset_resumable(
    result: TrainingViewBuildResult,
    output_root: str | Path,
    checkpoint_root: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> TrainingViewWriteResult:
    """Checkpoint each validated item artifact and publish the completed dataset atomically."""
    if not isinstance(result, TrainingViewBuildResult):
        raise TypeError("result must be TrainingViewBuildResult")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Training-view output root already exists: {output}")
    checkpoints = Path(checkpoint_root)
    artifacts = checkpoints / "items"
    markers = checkpoints / "markers"
    artifacts.mkdir(parents=True, exist_ok=True)
    markers.mkdir(parents=True, exist_ok=True)
    resumed = 0
    for index, item in enumerate(result.items, start=1):
        artifact = artifacts / f"{item.view_item_id}.npz"
        marker = markers / f"{item.view_item_id}.json"
        if artifact.is_file() and marker.is_file():
            payload = strict_json_load(marker)
            if not isinstance(payload, dict) or payload.get("complete") is not True:
                raise ValueError(f"Invalid Phase 12 checkpoint marker {marker}")
            if payload.get("training_view_dataset_id") != result.training_view_dataset_id:
                raise ValueError(f"Stale Phase 12 checkpoint {marker}")
            if payload.get("view_item_id") != item.view_item_id or payload.get("content_hash") != item.content_hash:
                raise ValueError(f"Phase 12 checkpoint identity mismatch {marker}")
            load_training_view_item_artifact(artifact, result.config, item.content_hash)
            resumed += 1
        else:
            artifact.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            save_training_view_item_artifact(item, result.config, artifact)
            write_strict_json(
                marker,
                {
                    "complete": True,
                    "training_view_dataset_id": result.training_view_dataset_id,
                    "view_item_id": item.view_item_id,
                    "task": item.task,
                    "content_hash": item.content_hash,
                },
            )
        if progress_callback is not None:
            progress_callback({
                "completed_items": index,
                "total_items": len(result.items),
                "resumed_items": resumed,
                "task": item.task,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        (staging / "manifests").mkdir(parents=True)
        (staging / "artifacts" / "items").mkdir(parents=True)
        managed: list[str] = []
        save_training_view_config(result.config, staging / "training_view_config.json")
        managed.append("training_view_config.json")
        summary = {**result.summary, "checkpoint_resume": {"enabled": True, "resumed_item_count": resumed}}
        write_strict_json(staging / "training_view_summary.json", summary)
        managed.append("training_view_summary.json")
        for item in result.items:
            reference = f"artifacts/items/{item.view_item_id}.npz"
            source = artifacts / f"{item.view_item_id}.npz"
            target = staging / reference
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            managed.append(reference)
        writer = ManifestWriter(staging / "manifests")
        writer.write_records("training_view_manifest", result.definition_records, overwrite=False)
        managed.append("manifests/training_view_manifest.parquet")
        writer.write_records("training_item_manifest", result.item_records, overwrite=False)
        managed.append("manifests/training_item_manifest.parquet")
        persisted_config = load_training_view_config(staging / "training_view_config.json")
        if persisted_config != result.config:
            raise ValueError("Persisted training-view config mismatch")
        reader = ManifestReader(staging / "manifests")
        definitions = reader.read_typed_records("training_view_manifest", TrainingViewDefinitionRecordV1)
        records = reader.read_typed_records("training_item_manifest", TrainingViewItemRecordV1)
        loaded = {}
        for record in records:
            item = load_training_view_item_artifact(
                resolve_safe_file(staging, record.artifact_ref, f"Training item {record.view_item_id}"),
                persisted_config,
                record.content_hash,
            )
            loaded[item.view_item_id] = item
        validate_training_view_dataset_joins(definitions, records, items_by_id=loaded, config=persisted_config)
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
            "checkpoint_resume": {"enabled": True, "resumed_item_count": resumed},
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "training_view_complete.json", completion)
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed training-view inventory does not match staging")
        os.replace(staging, output)
        manifest_paths = (
            output / "manifests" / "training_view_manifest.parquet",
            output / "manifests" / "training_item_manifest.parquet",
        )
        artifact_paths = tuple(sorted((output / ref for ref in managed_files if ref.startswith("artifacts/")), key=lambda p: p.as_posix()))
        written_paths = tuple(sorted((output / ref for ref in managed_files), key=lambda p: p.as_posix()))
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


__all__ = ["write_training_view_dataset_resumable"]
