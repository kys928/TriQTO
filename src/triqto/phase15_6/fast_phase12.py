"""Fast, lossless Phase 12 construction with bounded parallel shard workers.

This module preserves the established scientific Phase 12 item builders while improving
only operational execution:

* existing content-addressed item checkpoints are verified by marker identity and
  compressed-artifact SHA-256 without reopening/decompressing their NPZ payloads;
* independent deterministic logical shards are built concurrently with a bounded thread
  pool; and
* newly committed shard manifests are validated structurally without reopening every item
  artifact a second time.

No candidates, items, source references, or targets are sampled or truncated.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any

from triqto.graph.utils import strict_json_load, write_strict_json
from triqto.training_views.config import TrainingViewConfig
from triqto.training_views.models import TrainingViewBuildResult, TrainingViewItem
from triqto.training_views.validators import validate_training_view_item

from . import resumable_phase12 as _base
from .resumable import (
    clear_checkpoint_failure,
    commit_checkpoint_artifact,
    load_checkpoint_artifact,
    record_checkpoint_failure,
)

ProgressCallback = _base.ProgressCallback
_PHASE = "phase12"


def _require_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Phase 12 workers must be an integer and not bool")
    if value < 1:
        raise ValueError("Phase 12 workers must be positive")
    if value > 32:
        raise ValueError("Phase 12 workers must not exceed 32")
    return value


def _fast_existing_item_checkpoint(
    root: Path,
    dataset_id: str,
    item: TrainingViewItem,
    resume_mode: str,
) -> bool:
    """Verify an existing item checkpoint without decoding its NPZ arrays.

    The item was freshly rebuilt and validated in memory before this function is called.
    The checkpoint marker binds the same dataset, item id, task, and content hash, while
    ``load_checkpoint_artifact`` verifies the persisted compressed file's SHA-256.  That
    is sufficient to reuse the existing immutable artifact without a second semantic NPZ
    decode during operational migration.
    """
    artifact, marker = _base._item_paths(root, item.view_item_id)
    loaded = load_checkpoint_artifact(
        root=root,
        phase=_PHASE,
        unit_id=item.view_item_id,
        stage="item_artifact",
        artifact_path=artifact,
        marker_path=marker,
        identity=_base._item_identity(dataset_id, item),
        resume_mode=resume_mode,
        loader=lambda _path, _payload: True,
    )
    return loaded is True


def _fast_ensure_item_checkpoint(
    root: Path,
    dataset_id: str,
    item: TrainingViewItem,
    config: TrainingViewConfig,
    resume_mode: str,
) -> TrainingViewItem:
    if _fast_existing_item_checkpoint(root, dataset_id, item, resume_mode):
        return item
    # New or repaired items retain the original full write-and-reload validation path.
    return _ORIGINAL_ENSURE_ITEM_CHECKPOINT(
        root,
        dataset_id,
        item,
        config,
        resume_mode,
    )


def _validate_committed_shard_manifest(
    path: Path,
    expected: Mapping[str, Any],
    validated_items: list[TrainingViewItem],
) -> list[TrainingViewItem]:
    payload = strict_json_load(path)
    if payload != dict(expected):
        raise ValueError("Committed Phase 12 shard manifest content mismatch")
    return list(validated_items)


def _process_shard(
    *,
    task: str,
    context: _base.ViewBuildContext,
    root: Path,
    dataset_id: str,
    config: TrainingViewConfig,
    built_by_task: Mapping[str, list[TrainingViewItem]],
    shard_count: int,
    resume_mode: str,
    source_identity: Mapping[str, Any],
    shard_index: int,
    ids: list[str],
) -> tuple[int, list[TrainingViewItem], str]:
    active_resume_mode = "strict" if resume_mode == "off" else resume_mode
    unit_id = f"{task}-shard-{shard_index:03d}"
    manifest_path, marker_path = _base._shard_paths(root, task, shard_index)
    identity = {
        "checkpoint_schema": "triqto.phase15_6.phase12_logical_shard.v2",
        "training_view_dataset_id": dataset_id,
        "task": task,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "entity_ids": ids,
        **dict(source_identity),
    }
    loaded = load_checkpoint_artifact(
        root=root,
        phase=_PHASE,
        unit_id=unit_id,
        stage="logical_shard",
        artifact_path=manifest_path,
        marker_path=marker_path,
        identity=identity,
        resume_mode=active_resume_mode,
        loader=lambda path, _payload: _base._load_shard_items(
            path,
            root=root,
            dataset_id=dataset_id,
            task=task,
            config=config,
            resume_mode=active_resume_mode,
        ),
    )
    if loaded is not None:
        return shard_index, loaded, "resumed"

    try:
        built = _base._build_task_subset(task, context, ids, built_by_task)
        allowed = set(ids)
        seen_entities: set[str] = set()
        validated: list[TrainingViewItem] = []
        for item in built:
            if item.entity_id not in allowed:
                raise ValueError(
                    f"Task {task} shard produced unexpected entity {item.entity_id}"
                )
            if item.entity_id in seen_entities:
                raise ValueError(
                    f"Task {task} shard produced duplicate entity {item.entity_id}"
                )
            seen_entities.add(item.entity_id)
            validate_training_view_item(item, config, require_hash=True)
            validated.append(
                _base._ensure_item_checkpoint(
                    root,
                    dataset_id,
                    item,
                    config,
                    active_resume_mode,
                )
            )
        validated.sort(key=lambda item: item.view_item_id)
        manifest = {
            "schema": "triqto.phase15_6.phase12_logical_shard_manifest.v1",
            "training_view_dataset_id": dataset_id,
            "task": task,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "entity_ids": ids,
            "items": [
                {
                    "view_item_id": item.view_item_id,
                    "entity_id": item.entity_id,
                    "content_hash": item.content_hash,
                }
                for item in validated
            ],
        }
        committed = commit_checkpoint_artifact(
            phase=_PHASE,
            unit_id=unit_id,
            stage="logical_shard",
            artifact_path=manifest_path,
            marker_path=marker_path,
            identity=identity,
            writer=lambda path: write_strict_json(path, manifest),
            validator=lambda path: _validate_committed_shard_manifest(
                path,
                manifest,
                validated,
            ),
            marker_metadata={
                "task": task,
                "shard_index": shard_index,
                "entity_count": len(ids),
                "item_count": len(validated),
                "fast_item_reuse": True,
            },
        )
        clear_checkpoint_failure(root, unit_id)
        return shard_index, committed, "completed"
    except Exception as exc:
        record_checkpoint_failure(
            root=root,
            phase=_PHASE,
            unit_id=unit_id,
            stage="logical_shard",
            error=exc,
            context={
                "task": task,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "entity_count": len(ids),
            },
        )
        raise


def _parallel_build_task_resumable(
    *,
    task: str,
    context: _base.ViewBuildContext,
    root: Path,
    dataset_id: str,
    config: TrainingViewConfig,
    built_by_task: Mapping[str, list[TrainingViewItem]],
    shard_count: int,
    resume_mode: str,
    progress_callback: ProgressCallback | None,
    source_identity: Mapping[str, Any],
    workers: int,
) -> tuple[list[TrainingViewItem], int]:
    entity_ids = _base._task_entity_ids(task, context)
    shards: dict[int, list[str]] = defaultdict(list)
    for entity_id in entity_ids:
        shards[_base._shard_index(entity_id, shard_count)].append(entity_id)
    nonempty = sorted(shards)
    if not nonempty:
        return [], 0

    task_started = time.monotonic()
    total_entities = sum(len(shards[index]) for index in nonempty)
    result: list[TrainingViewItem] = []
    resumed_shards = 0
    completed_shards = 0
    completed_entities = 0

    def emit(shard_index: int, ids: list[str], status: str) -> None:
        if progress_callback is None:
            return
        elapsed = max(time.monotonic() - task_started, 1e-9)
        rate = completed_shards / elapsed if completed_shards else 0.0
        eta = (
            (len(nonempty) - completed_shards) / rate
            if rate > 0.0
            else None
        )
        progress_callback(
            {
                "event": "logical_shard",
                "task": task,
                "shard_index": shard_index,
                "completed_shards": completed_shards,
                "total_shards": len(nonempty),
                "entity_count": len(ids),
                "completed_entities": completed_entities,
                "total_entities": total_entities,
                "status": status,
                "resumed_shards": resumed_shards,
                "workers": workers,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            }
        )

    max_workers = min(workers, len(nonempty))
    futures: dict[Future[tuple[int, list[TrainingViewItem], str]], tuple[int, list[str]]] = {}
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=f"triqto-phase12-{task}",
    ) as executor:
        for shard_index in nonempty:
            ids = sorted(shards[shard_index])
            future = executor.submit(
                _process_shard,
                task=task,
                context=context,
                root=root,
                dataset_id=dataset_id,
                config=config,
                built_by_task=built_by_task,
                shard_count=shard_count,
                resume_mode=resume_mode,
                source_identity=source_identity,
                shard_index=shard_index,
                ids=ids,
            )
            futures[future] = (shard_index, ids)

        try:
            for future in as_completed(futures):
                shard_index, ids = futures[future]
                _resolved_index, items, status = future.result()
                result.extend(items)
                completed_shards += 1
                completed_entities += len(ids)
                if status == "resumed":
                    resumed_shards += 1
                emit(shard_index, ids, status)
        except Exception:
            for pending in futures:
                pending.cancel()
            raise

    result.sort(key=lambda item: item.view_item_id)
    return result, resumed_shards


def build_training_view_result_fast(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    action_source_root: str | Path,
    topology_source_root: str | Path,
    checkpoint_root: str | Path,
    config: TrainingViewConfig | None = None,
    *,
    shard_count: int = 256,
    resume_mode: str = "strict",
    progress_callback: ProgressCallback | None = None,
    workers: int = 4,
) -> TrainingViewBuildResult:
    """Build Phase 12 using lossless fast reuse and bounded parallel shards."""
    resolved_workers = _require_workers(workers)
    original_ensure = _base._ensure_item_checkpoint
    original_task_builder = _base._build_task_resumable

    def task_builder(**kwargs: Any):
        return _parallel_build_task_resumable(
            **kwargs,
            workers=resolved_workers,
        )

    _base._ensure_item_checkpoint = _fast_ensure_item_checkpoint
    _base._build_task_resumable = task_builder
    try:
        return _base.build_training_view_result_resumable(
            phase7_source_root,
            graph_source_root,
            action_source_root,
            topology_source_root,
            checkpoint_root,
            config,
            shard_count=shard_count,
            resume_mode=resume_mode,
            progress_callback=progress_callback,
        )
    finally:
        _base._ensure_item_checkpoint = original_ensure
        _base._build_task_resumable = original_task_builder


_ORIGINAL_ENSURE_ITEM_CHECKPOINT = _base._ensure_item_checkpoint


__all__ = ["build_training_view_result_fast"]
