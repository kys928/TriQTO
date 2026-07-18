"""Deterministically sharded and restartable Phase 12 construction/publication."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import copy
import hashlib
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Callable

from triqto.graph.utils import resolve_safe_file, strict_json_load, write_strict_json
from triqto.storage import ManifestReader, ManifestWriter
from triqto.storage.training_view_schema import (
    TrainingViewDefinitionRecordV1,
    TrainingViewItemRecordV1,
)
from triqto.training_views.action_ranking_view import build_action_ranking_items
from triqto.training_views.artifacts import (
    _validate_source_refs,
    _verify_result_sources,
    load_training_view_item_artifact,
    save_training_view_item_artifact,
)
from triqto.training_views.born_prediction_view import build_born_prediction_items
from triqto.training_views.config import (
    TrainingViewConfig,
    load_training_view_config,
    save_training_view_config,
)
from triqto.training_views.constants import (
    SPLIT_ORDER,
    TASK_INPUT_GROUPS,
    TASK_ORDER,
    TASK_TARGET_GROUPS,
)
from triqto.training_views.context import ViewBuildContext, build_view_context
from triqto.training_views.diagnosis_view import build_diagnosis_items
from triqto.training_views.hardware_masked_view import build_hardware_masked_items
from triqto.training_views.hilbert_to_born_view import build_hilbert_to_born_items
from triqto.training_views.identities import (
    training_view_dataset_id,
    training_view_id,
    training_view_operational_config_id,
    training_view_schema_id,
)
from triqto.training_views.models import (
    TrainingViewBuildResult,
    TrainingViewDefinition,
    TrainingViewItem,
    TrainingViewWriteResult,
)
from triqto.training_views.multitask_view import build_joint_multitask_items
from triqto.training_views.source import (
    load_training_view_sources,
    verify_training_view_source_snapshots,
)
from triqto.training_views.topology_view import build_topology_audit_items
from triqto.training_views.validators import (
    validate_training_view_dataset_joins,
    validate_training_view_item,
)

from .resumable import (
    clear_checkpoint_failure,
    commit_checkpoint_artifact,
    load_checkpoint_artifact,
    prepare_checkpoint_root,
    record_checkpoint_failure,
)

ProgressCallback = Callable[[dict[str, Any]], None]
_PHASE = "phase12"
_SAMPLE_TASKS = {
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "hilbert_to_born",
    "joint_multitask",
    "hardware_masked",
}


def _by_sample(items: list[TrainingViewItem]) -> dict[str, TrainingViewItem]:
    result: dict[str, TrainingViewItem] = {}
    for item in items:
        sample_id = item.metadata.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        if sample_id in result:
            raise ValueError(f"Multiple {item.task} items resolve to sample {sample_id}")
        result[sample_id] = item
    return result


def _fixed_split_counts(items: list[TrainingViewItem]) -> dict[str, int]:
    observed = Counter(item.split for item in items)
    unknown = set(observed) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"Unknown Phase 12 splits: {sorted(unknown)}")
    return {split: int(observed.get(split, 0)) for split in SPLIT_ORDER}


def _shard_index(entity_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(entity_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % shard_count


def _sample_context(
    context: ViewBuildContext,
    sample_ids: list[str],
) -> ViewBuildContext:
    phase7 = copy.copy(context.sources.phase7)
    phase7.samples = [context.samples_by_id[sample_id] for sample_id in sample_ids]
    sources = copy.copy(context.sources)
    sources.phase7 = phase7
    filtered = copy.copy(context)
    filtered.sources = sources
    return filtered


def _topology_context(
    context: ViewBuildContext,
    group_ids: list[str],
) -> ViewBuildContext:
    topology = copy.copy(context.sources.topology)
    topology.groups_by_id = {
        group_id: context.sources.topology.groups_by_id[group_id]
        for group_id in group_ids
    }
    topology.records_by_id = {
        group_id: context.sources.topology.records_by_id[group_id]
        for group_id in group_ids
    }
    sources = copy.copy(context.sources)
    sources.topology = topology
    filtered = copy.copy(context)
    filtered.sources = sources
    return filtered


def _item_paths(root: Path, item_id: str) -> tuple[Path, Path]:
    return root / "items" / f"{item_id}.npz", root / "markers" / f"{item_id}.json"


def _item_identity(result_id: str, item: TrainingViewItem) -> dict[str, Any]:
    return {
        "checkpoint_schema": "triqto.phase15_6.phase12_item.v2",
        "training_view_dataset_id": result_id,
        "view_item_id": item.view_item_id,
        "task": item.task,
        "content_hash": item.content_hash,
    }


def _load_item_checkpoint(
    root: Path,
    result_id: str,
    item_id: str,
    content_hash: str,
    task: str,
    config: TrainingViewConfig,
    resume_mode: str,
) -> TrainingViewItem | None:
    artifact, marker = _item_paths(root, item_id)
    identity = {
        "checkpoint_schema": "triqto.phase15_6.phase12_item.v2",
        "training_view_dataset_id": result_id,
        "view_item_id": item_id,
        "task": task,
        "content_hash": content_hash,
    }
    return load_checkpoint_artifact(
        root=root,
        phase=_PHASE,
        unit_id=item_id,
        stage="item_artifact",
        artifact_path=artifact,
        marker_path=marker,
        identity=identity,
        resume_mode=resume_mode,
        loader=lambda path, _payload: load_training_view_item_artifact(
            path, config, content_hash
        ),
    )


def _ensure_item_checkpoint(
    root: Path,
    dataset_id: str,
    item: TrainingViewItem,
    config: TrainingViewConfig,
    resume_mode: str,
) -> TrainingViewItem:
    loaded = _load_item_checkpoint(
        root,
        dataset_id,
        item.view_item_id,
        item.content_hash,
        item.task,
        config,
        resume_mode,
    )
    if loaded is not None:
        return loaded
    artifact, marker = _item_paths(root, item.view_item_id)
    return commit_checkpoint_artifact(
        phase=_PHASE,
        unit_id=item.view_item_id,
        stage="item_artifact",
        artifact_path=artifact,
        marker_path=marker,
        identity=_item_identity(dataset_id, item),
        writer=lambda path: save_training_view_item_artifact(item, config, path),
        validator=lambda path: load_training_view_item_artifact(
            path, config, item.content_hash
        ),
        marker_metadata={
            "task": item.task,
            "entity_id": item.entity_id,
            "content_hash": item.content_hash,
        },
    )


def _shard_paths(root: Path, task: str, shard_index: int) -> tuple[Path, Path]:
    directory = root / "logical_shards" / task
    stem = f"shard-{shard_index:03d}"
    return directory / f"{stem}.json", directory / f"{stem}.complete.json"


def _load_shard_items(
    manifest_path: Path,
    *,
    root: Path,
    dataset_id: str,
    task: str,
    config: TrainingViewConfig,
    resume_mode: str,
) -> list[TrainingViewItem]:
    payload = strict_json_load(manifest_path)
    if not isinstance(payload, dict):
        raise TypeError("Phase 12 shard manifest must be a JSON object")
    if payload.get("training_view_dataset_id") != dataset_id:
        raise ValueError("Phase 12 shard dataset identity mismatch")
    if payload.get("task") != task:
        raise ValueError("Phase 12 shard task mismatch")
    records = payload.get("items")
    if not isinstance(records, list):
        raise TypeError("Phase 12 shard items must be a list")
    items: list[TrainingViewItem] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Phase 12 shard item record must be an object")
        item_id = record.get("view_item_id")
        content_hash = record.get("content_hash")
        if not isinstance(item_id, str) or not isinstance(content_hash, str):
            raise TypeError("Phase 12 shard item identity is invalid")
        item = _load_item_checkpoint(
            root,
            dataset_id,
            item_id,
            content_hash,
            task,
            config,
            resume_mode,
        )
        if item is None:
            raise ValueError(f"Phase 12 shard references missing item {item_id}")
        items.append(item)
    items.sort(key=lambda item: item.view_item_id)
    return items


def _build_task_subset(
    task: str,
    context: ViewBuildContext,
    entity_ids: list[str],
    built_by_task: Mapping[str, list[TrainingViewItem]],
) -> list[TrainingViewItem]:
    if task == "topology_audit":
        return build_topology_audit_items(_topology_context(context, entity_ids))
    filtered = _sample_context(context, entity_ids)
    if task == "diagnosis":
        return build_diagnosis_items(filtered)
    if task == "action_ranking":
        return build_action_ranking_items(filtered)
    if task == "born_prediction":
        return build_born_prediction_items(filtered)
    if task == "hilbert_to_born":
        return build_hilbert_to_born_items(filtered)
    if task == "joint_multitask":
        diagnosis = _by_sample(built_by_task["diagnosis"])
        actions = _by_sample(built_by_task["action_ranking"])
        born = _by_sample(built_by_task["born_prediction"])
        hilbert = _by_sample(built_by_task["hilbert_to_born"])
        topology = _by_sample(
            [
                item
                for item in built_by_task["topology_audit"]
                if item.metadata.get("group_kind") == "action_neighborhood"
            ]
        )
        selected = set(entity_ids)
        return build_joint_multitask_items(
            filtered,
            diagnosis_items={key: value for key, value in diagnosis.items() if key in selected},
            action_items={key: value for key, value in actions.items() if key in selected},
            born_items={key: value for key, value in born.items() if key in selected},
            hilbert_items={key: value for key, value in hilbert.items() if key in selected},
            topology_items={key: value for key, value in topology.items() if key in selected},
        )
    if task == "hardware_masked":
        joint = _by_sample(built_by_task["joint_multitask"])
        selected = set(entity_ids)
        return build_hardware_masked_items(
            filtered,
            {key: value for key, value in joint.items() if key in selected},
        )
    raise ValueError(f"Unsupported Phase 12 task {task!r}")


def _task_entity_ids(task: str, context: ViewBuildContext) -> list[str]:
    if task == "hilbert_to_born":
        if not context.config.include_hilbert:
            return []
        eligible = [
            sample_id
            for sample_id, sample in sorted(context.samples_by_id.items())
            if (
                (simulation := context.simulations_by_id.get(sample.distorted_run_id))
                is not None
                and bool(simulation.statevector_ref)
            )
        ]
        if not eligible and not context.config.allow_empty_hilbert_view:
            raise ValueError(
                "No Phase 7 distorted statevector artifacts are available for hilbert_to_born"
            )
        return eligible
    if task == "topology_audit":
        if not context.config.include_topology:
            return []
        return sorted(context.sources.topology.groups_by_id)
    if task in _SAMPLE_TASKS:
        return sorted(context.samples_by_id)
    raise ValueError(f"Unsupported Phase 12 task {task!r}")


def _build_task_resumable(
    *,
    task: str,
    context: ViewBuildContext,
    root: Path,
    dataset_id: str,
    config: TrainingViewConfig,
    built_by_task: Mapping[str, list[TrainingViewItem]],
    shard_count: int,
    resume_mode: str,
    progress_callback: ProgressCallback | None,
    source_identity: Mapping[str, Any],
) -> tuple[list[TrainingViewItem], int]:
    active_resume_mode = "strict" if resume_mode == "off" else resume_mode
    entity_ids = _task_entity_ids(task, context)
    shards: dict[int, list[str]] = defaultdict(list)
    for entity_id in entity_ids:
        shards[_shard_index(entity_id, shard_count)].append(entity_id)
    result: list[TrainingViewItem] = []
    resumed_shards = 0
    nonempty = sorted(shards)
    for ordinal, shard_index in enumerate(nonempty, start=1):
        ids = sorted(shards[shard_index])
        unit_id = f"{task}-shard-{shard_index:03d}"
        manifest_path, marker_path = _shard_paths(root, task, shard_index)
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
            loader=lambda path, _payload: _load_shard_items(
                path,
                root=root,
                dataset_id=dataset_id,
                task=task,
                config=config,
                resume_mode=active_resume_mode,
            ),
        )
        if loaded is not None:
            resumed_shards += 1
            result.extend(loaded)
            status = "resumed"
        else:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "logical_shard",
                        "task": task,
                        "shard_index": shard_index,
                        "completed_shards": ordinal - 1,
                        "total_shards": len(nonempty),
                        "entity_count": len(ids),
                        "status": "started",
                        "resumed_shards": resumed_shards,
                    }
                )
            try:
                built = _build_task_subset(task, context, ids, built_by_task)
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
                        _ensure_item_checkpoint(
                            root, dataset_id, item, config, active_resume_mode
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
                loaded = commit_checkpoint_artifact(
                    phase=_PHASE,
                    unit_id=unit_id,
                    stage="logical_shard",
                    artifact_path=manifest_path,
                    marker_path=marker_path,
                    identity=identity,
                    writer=lambda path: write_strict_json(path, manifest),
                    validator=lambda path: _load_shard_items(
                        path,
                        root=root,
                        dataset_id=dataset_id,
                        task=task,
                        config=config,
                        resume_mode=active_resume_mode,
                    ),
                    marker_metadata={
                        "task": task,
                        "shard_index": shard_index,
                        "entity_count": len(ids),
                        "item_count": len(validated),
                    },
                )
                clear_checkpoint_failure(root, unit_id)
                result.extend(loaded)
                status = "completed"
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
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "logical_shard",
                    "task": task,
                    "shard_index": shard_index,
                    "completed_shards": ordinal,
                    "total_shards": len(nonempty),
                    "entity_count": len(ids),
                    "status": status,
                    "resumed_shards": resumed_shards,
                }
            )
    result.sort(key=lambda item: item.view_item_id)
    return result, resumed_shards


def build_training_view_result_resumable(
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
) -> TrainingViewBuildResult:
    """Build Phase 12 in deterministic task/entity shards with durable checkpoints."""
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("Phase 12 shard_count must be a positive integer")
    view_config = config or TrainingViewConfig()
    if not isinstance(view_config, TrainingViewConfig):
        raise TypeError("config must be TrainingViewConfig or None")
    sources = load_training_view_sources(
        phase7_source_root,
        graph_source_root,
        action_source_root,
        topology_source_root,
    )
    dataset_id = training_view_dataset_id(
        sources.phase7.source_scientific_generation_id,
        sources.graph.completion_marker["graph_conversion_id"],
        sources.action.completion_marker["action_engine_id"],
        sources.topology.completion_marker["topology_audit_id"],
        view_config,
    )
    view_ids = {task: training_view_id(dataset_id, task) for task in TASK_ORDER}
    context = build_view_context(sources, view_config, dataset_id, view_ids)
    root = prepare_checkpoint_root(checkpoint_root, resume_mode)
    source_identity = {
        "operational_config_id": training_view_operational_config_id(view_config),
        "phase7_snapshot_hash": sources.phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": sources.graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": sources.action.snapshot.aggregate_sha256,
        "topology_snapshot_hash": sources.topology.snapshot.aggregate_sha256,
    }
    if progress_callback is not None:
        progress_callback(
            {
                "event": "plan",
                "phase": 12,
                "shard_count": shard_count,
                "task_count": len(TASK_ORDER),
            }
        )
    built_by_task: dict[str, list[TrainingViewItem]] = {}
    resumed_by_task: dict[str, int] = {}
    for task in TASK_ORDER:
        task_items, resumed = _build_task_resumable(
            task=task,
            context=context,
            root=root,
            dataset_id=dataset_id,
            config=view_config,
            built_by_task=built_by_task,
            shard_count=shard_count,
            resume_mode=resume_mode,
            progress_callback=progress_callback,
            source_identity=source_identity,
        )
        built_by_task[task] = task_items
        resumed_by_task[task] = resumed
    items = [
        item
        for task in view_config.tasks
        for item in built_by_task[task]
    ]
    items.sort(key=lambda value: (TASK_ORDER.index(value.task), value.view_item_id))
    if len(items) > view_config.max_items:
        raise RuntimeError(
            f"Training item count {len(items)} exceeds max_items={view_config.max_items}"
        )
    item_ids: set[str] = set()
    for item in items:
        if item.view_item_id in item_ids:
            raise ValueError(f"Duplicate training view item {item.view_item_id}")
        item_ids.add(item.view_item_id)
        validate_training_view_item(item, view_config, require_hash=True)
    definitions: list[TrainingViewDefinition] = []
    definition_records: list[TrainingViewDefinitionRecordV1] = []
    item_records: list[TrainingViewItemRecordV1] = []
    items_by_task: dict[str, list[TrainingViewItem]] = defaultdict(list)
    for item in items:
        items_by_task[item.task].append(item)
    for task in view_config.tasks:
        task_items = items_by_task.get(task, [])
        split_counts = _fixed_split_counts(task_items)
        definition = TrainingViewDefinition(
            training_view_id=view_ids[task],
            training_view_dataset_id=dataset_id,
            task=task,
            input_groups=TASK_INPUT_GROUPS[task],
            target_groups=TASK_TARGET_GROUPS[task],
            mask_policy="explicit_per_item_and_per_head_masks",
            split_policy="sha256_grouped_by_clean_circuit_id",
            item_count=len(task_items),
            split_counts=split_counts,
            metadata={
                "phase": 12,
                "topology_loss_weight": 0.0,
                "model_present": False,
                "training_executed": False,
                "empty_view": len(task_items) == 0,
                "hardware_data": False,
            },
        )
        definitions.append(definition)
        record = TrainingViewDefinitionRecordV1(
            training_view_id=definition.training_view_id,
            training_view_dataset_id=definition.training_view_dataset_id,
            task=definition.task,
            input_groups=list(definition.input_groups),
            target_groups=list(definition.target_groups),
            mask_policy=definition.mask_policy,
            split_policy=definition.split_policy,
            item_count=definition.item_count,
            split_counts=dict(definition.split_counts),
            metadata=dict(definition.metadata),
        )
        record.validate()
        definition_records.append(record)
    for item in items:
        record = TrainingViewItemRecordV1(
            view_item_id=item.view_item_id,
            training_view_id=item.training_view_id,
            training_view_dataset_id=item.training_view_dataset_id,
            task=item.task,
            split=item.split,
            split_group_id=item.split_group_id,
            entity_id=item.entity_id,
            input_groups=list(item.input_groups),
            target_groups=list(item.target_groups),
            artifact_ref=f"artifacts/items/{item.view_item_id}.npz",
            content_hash=item.content_hash,
            hilbert_available_mask=item.hilbert_available_mask,
            topology_available_mask=item.topology_available_mask,
            privileged_target_available_mask=item.privileged_target_available_mask,
            metadata={
                "phase": 12,
                "topology_loss_weight": 0.0,
                "hardware_data": False,
            },
        )
        record.validate()
        item_records.append(record)
    validate_training_view_dataset_joins(
        definition_records,
        item_records,
        items_by_id={item.view_item_id: item for item in items},
        config=view_config,
    )
    verify_training_view_source_snapshots(sources)
    task_counts = {task: len(items_by_task.get(task, [])) for task in view_config.tasks}
    split_counts = _fixed_split_counts(items)
    privileged_count = sum(item.privileged_target_available_mask for item in items)
    hilbert_count = sum(item.hilbert_available_mask for item in items)
    topology_count = sum(item.topology_available_mask for item in items)
    summary = {
        "source_scientific_generation_id": sources.phase7.source_scientific_generation_id,
        "graph_conversion_id": sources.graph.completion_marker["graph_conversion_id"],
        "action_engine_id": sources.action.completion_marker["action_engine_id"],
        "topology_audit_id": sources.topology.completion_marker["topology_audit_id"],
        "training_view_dataset_id": dataset_id,
        "operational_config_id": training_view_operational_config_id(view_config),
        "training_view_schema_id": training_view_schema_id(),
        "view_count": len(definitions),
        "item_count": len(items),
        "task_item_counts": task_counts,
        "split_item_counts": split_counts,
        "unique_split_group_count": len({item.split_group_id for item in items}),
        "hilbert_available_item_count": hilbert_count,
        "topology_available_item_count": topology_count,
        "privileged_target_available_item_count": privileged_count,
        "cross_split_topology_items_are_audit_only": True,
        "clean_circuit_grouped_split": True,
        "born_prediction_input_target_leakage_blocked": True,
        "action_rollout_targets_excluded_from_inputs": True,
        "hardware_masked_items_are_simulation": True,
        "hardware_data_present": False,
        "topology_loss_weight": 0.0,
        "training_executed": False,
        "model_present": False,
        "source_immutability_verified": True,
        "phase7_snapshot_hash": sources.phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": sources.graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": sources.action.snapshot.aggregate_sha256,
        "topology_snapshot_hash": sources.topology.snapshot.aggregate_sha256,
        "quantum_advantage_claimed": False,
        "checkpoint_resume": {
            "enabled": True,
            "granularity": "deterministic_task_entity_hash_shard_and_item_artifact",
            "shard_count": shard_count,
            "resume_mode": resume_mode,
            "resumed_shards_by_task": resumed_by_task,
        },
    }
    return TrainingViewBuildResult(
        phase7_source_root=sources.phase7.source_root,
        graph_source_root=sources.graph.root,
        action_source_root=sources.action.root,
        topology_source_root=sources.topology.root,
        config=view_config,
        source_scientific_generation_id=sources.phase7.source_scientific_generation_id,
        graph_conversion_id=sources.graph.completion_marker["graph_conversion_id"],
        action_engine_id=sources.action.completion_marker["action_engine_id"],
        topology_audit_id=sources.topology.completion_marker["topology_audit_id"],
        training_view_dataset_id=dataset_id,
        operational_config_id=training_view_operational_config_id(view_config),
        training_view_schema_id=training_view_schema_id(),
        items=items,
        definitions=definitions,
        item_records=item_records,
        definition_records=definition_records,
        phase7_snapshot=sources.phase7.source_snapshot,
        graph_snapshot=sources.graph.snapshot,
        action_snapshot=sources.action.snapshot,
        topology_snapshot=sources.topology.snapshot,
        summary=summary,
    )


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def write_training_view_dataset_resumable(
    result: TrainingViewBuildResult,
    output_root: str | Path,
    checkpoint_root: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
    resume_mode: str = "strict",
) -> TrainingViewWriteResult:
    """Publish from validated item checkpoints into one atomic immutable dataset."""
    if not isinstance(result, TrainingViewBuildResult):
        raise TypeError("result must be TrainingViewBuildResult")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Training-view output root already exists: {output}")
    source_roots = {
        "phase7": Path(result.phase7_source_root),
        "phase8": Path(result.graph_source_root),
        "phase9": Path(result.action_source_root),
        "phase11": Path(result.topology_source_root),
    }
    _verify_result_sources(result)
    active_resume_mode = "strict" if resume_mode == "off" else resume_mode
    checkpoints = prepare_checkpoint_root(checkpoint_root, active_resume_mode)
    resumed = 0
    for index, item in enumerate(result.items, start=1):
        _validate_source_refs(item, source_roots)
        existing = _load_item_checkpoint(
            checkpoints,
            result.training_view_dataset_id,
            item.view_item_id,
            item.content_hash,
            item.task,
            result.config,
            active_resume_mode,
        )
        if existing is None:
            _ensure_item_checkpoint(
                checkpoints,
                result.training_view_dataset_id,
                item,
                result.config,
                active_resume_mode,
            )
        else:
            resumed += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "item_publication",
                    "completed_items": index,
                    "total_items": len(result.items),
                    "resumed_items": resumed,
                    "task": item.task,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        (staging / "manifests").mkdir(parents=True)
        (staging / "artifacts" / "items").mkdir(parents=True)
        managed: list[str] = []
        save_training_view_config(result.config, staging / "training_view_config.json")
        managed.append("training_view_config.json")
        summary = {
            **result.summary,
            "checkpoint_resume": {
                **dict(result.summary.get("checkpoint_resume", {})),
                "resumed_item_count": resumed,
            },
        }
        write_strict_json(staging / "training_view_summary.json", summary)
        managed.append("training_view_summary.json")
        for item in result.items:
            reference = f"artifacts/items/{item.view_item_id}.npz"
            source = _item_paths(checkpoints, item.view_item_id)[0]
            target = staging / reference
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            managed.append(reference)
        writer = ManifestWriter(staging / "manifests")
        writer.write_records(
            "training_view_manifest", result.definition_records, overwrite=False
        )
        managed.append("manifests/training_view_manifest.parquet")
        writer.write_records(
            "training_item_manifest", result.item_records, overwrite=False
        )
        managed.append("manifests/training_item_manifest.parquet")
        persisted_config = load_training_view_config(
            staging / "training_view_config.json"
        )
        if persisted_config != result.config:
            raise ValueError("Persisted training-view config mismatch")
        reader = ManifestReader(staging / "manifests")
        definitions = reader.read_typed_records(
            "training_view_manifest", TrainingViewDefinitionRecordV1
        )
        records = reader.read_typed_records(
            "training_item_manifest", TrainingViewItemRecordV1
        )
        loaded: dict[str, TrainingViewItem] = {}
        for record in records:
            item = load_training_view_item_artifact(
                resolve_safe_file(
                    staging,
                    record.artifact_ref,
                    f"Training item {record.view_item_id}",
                ),
                persisted_config,
                record.content_hash,
            )
            _validate_source_refs(item, source_roots)
            loaded[item.view_item_id] = item
        validate_training_view_dataset_joins(
            definitions,
            records,
            items_by_id=loaded,
            config=persisted_config,
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
            "checkpoint_resume": summary["checkpoint_resume"],
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "training_view_complete.json", completion)
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed training-view inventory does not match staging")
        _verify_result_sources(result)
        os.replace(staging, output)
        manifest_paths = (
            output / "manifests" / "training_view_manifest.parquet",
            output / "manifests" / "training_item_manifest.parquet",
        )
        artifact_paths = tuple(
            sorted(
                (
                    output / reference
                    for reference in managed_files
                    if reference.startswith("artifacts/")
                ),
                key=lambda path: path.as_posix(),
            )
        )
        written_paths = tuple(
            sorted(
                (output / reference for reference in managed_files),
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
    "build_training_view_result_resumable",
    "write_training_view_dataset_resumable",
]
