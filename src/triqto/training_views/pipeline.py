"""Dataset-level orchestration for deterministic Phase 12 training views."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from triqto.storage.training_view_schema import (
    TrainingViewDefinitionRecordV1,
    TrainingViewItemRecordV1,
)

from .action_ranking_view import build_action_ranking_items
from .born_prediction_view import build_born_prediction_items
from .config import TrainingViewConfig
from .constants import TASK_INPUT_GROUPS, TASK_ORDER, TASK_TARGET_GROUPS
from .context import build_view_context
from .diagnosis_view import build_diagnosis_items
from .hardware_masked_view import build_hardware_masked_items
from .hilbert_to_born_view import build_hilbert_to_born_items
from .identities import (
    training_view_dataset_id,
    training_view_id,
    training_view_operational_config_id,
    training_view_schema_id,
)
from .models import (
    TrainingViewBuildResult,
    TrainingViewDefinition,
    TrainingViewItem,
)
from .multitask_view import build_joint_multitask_items
from .source import load_training_view_sources, verify_training_view_source_snapshots
from .topology_view import build_topology_audit_items
from .validators import (
    validate_training_view_dataset_joins,
    validate_training_view_item,
)


def _by_sample(items: list[TrainingViewItem]) -> dict[str, TrainingViewItem]:
    result: dict[str, TrainingViewItem] = {}
    for item in items:
        sample_id = item.metadata.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        if sample_id in result:
            raise ValueError(
                f"Multiple {item.task} items resolve to sample {sample_id}"
            )
        result[sample_id] = item
    return result


def build_training_view_result(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    action_source_root: str | Path,
    topology_source_root: str | Path,
    config: TrainingViewConfig | None = None,
) -> TrainingViewBuildResult:
    """Build leakage-safe task views without training a model or mutating sources."""
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

    diagnosis_items = build_diagnosis_items(context)
    action_items = build_action_ranking_items(context)
    born_items = build_born_prediction_items(context)
    hilbert_items = build_hilbert_to_born_items(context)
    topology_items = build_topology_audit_items(context)
    diagnosis_by_sample = _by_sample(diagnosis_items)
    action_by_sample = _by_sample(action_items)
    born_by_sample = _by_sample(born_items)
    hilbert_by_sample = _by_sample(hilbert_items)
    topology_by_sample = _by_sample(
        [
            item
            for item in topology_items
            if item.metadata.get("group_kind") == "action_neighborhood"
        ]
    )
    joint_items = build_joint_multitask_items(
        context,
        diagnosis_items=diagnosis_by_sample,
        action_items=action_by_sample,
        born_items=born_by_sample,
        hilbert_items=hilbert_by_sample,
        topology_items=topology_by_sample,
    )
    joint_by_sample = _by_sample(joint_items)
    hardware_items = build_hardware_masked_items(context, joint_by_sample)

    built_by_task = {
        "diagnosis": diagnosis_items,
        "action_ranking": action_items,
        "born_prediction": born_items,
        "hilbert_to_born": hilbert_items,
        "topology_audit": topology_items,
        "joint_multitask": joint_items,
        "hardware_masked": hardware_items,
    }
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
        split_counts = dict(sorted(Counter(item.split for item in task_items).items()))
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
    split_counts = dict(sorted(Counter(item.split for item in items).items()))
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


__all__ = ["build_training_view_result"]
