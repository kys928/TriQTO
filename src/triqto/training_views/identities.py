"""Deterministic scientific identities and logical item hashes for Phase 12."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.graph.utils import json_copy

from .config import TrainingViewConfig, training_view_config_to_dict
from .constants import (
    TASK_INPUT_GROUPS,
    TASK_ORDER,
    TASK_TARGET_GROUPS,
    TRAINING_VIEW_ARTIFACT_VERSION,
    TRAINING_VIEW_DEFINITION_MANIFEST_VERSION,
    TRAINING_VIEW_ITEM_MANIFEST_VERSION,
    TRAINING_VIEW_MASK_VERSION,
    TRAINING_VIEW_SCHEMA_VERSION,
    TRAINING_VIEW_SPLIT_VERSION,
)
from .models import TrainingViewItem


def training_view_schema_id() -> str:
    return make_deterministic_id(
        "viewschema",
        {
            "schema_version": TRAINING_VIEW_SCHEMA_VERSION,
            "artifact_version": TRAINING_VIEW_ARTIFACT_VERSION,
            "definition_manifest_version": TRAINING_VIEW_DEFINITION_MANIFEST_VERSION,
            "item_manifest_version": TRAINING_VIEW_ITEM_MANIFEST_VERSION,
            "split_version": TRAINING_VIEW_SPLIT_VERSION,
            "mask_version": TRAINING_VIEW_MASK_VERSION,
            "task_order": TASK_ORDER,
            "task_input_groups": TASK_INPUT_GROUPS,
            "task_target_groups": TASK_TARGET_GROUPS,
            "topology_loss_weight": 0.0,
            "hardware_masked_is_simulation_not_hardware": True,
        },
    )


def scientific_training_view_config_payload(
    config: TrainingViewConfig,
) -> dict[str, Any]:
    if not isinstance(config, TrainingViewConfig):
        raise TypeError("config must be TrainingViewConfig")
    return {
        "schema_version": config.schema_version,
        "tasks": list(config.tasks),
        "split_seed": config.split_seed,
        "split_strategy": config.split_strategy,
        "holdout_axis": config.holdout_axis,
        "holdout_values": list(config.holdout_values),
        "train_fraction": config.train_fraction,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "split_grouping": config.split_grouping,
        "topology_cross_split_policy": config.topology_cross_split_policy,
        "include_hilbert": config.include_hilbert,
        "include_topology": config.include_topology,
        "allow_empty_hilbert_view": config.allow_empty_hilbert_view,
        "topology_loss_weight": config.topology_loss_weight,
    }


def training_view_dataset_id(
    source_scientific_generation_id: str,
    graph_conversion_id: str,
    action_engine_id: str,
    topology_audit_id: str,
    config: TrainingViewConfig,
) -> str:
    return make_deterministic_id(
        "viewdataset",
        {
            "source_scientific_generation_id": source_scientific_generation_id,
            "graph_conversion_id": graph_conversion_id,
            "action_engine_id": action_engine_id,
            "topology_audit_id": topology_audit_id,
            "training_view_schema_id": training_view_schema_id(),
            "scientific_config": scientific_training_view_config_payload(config),
        },
    )


def training_view_operational_config_id(config: TrainingViewConfig) -> str:
    return make_deterministic_id("viewconfig", training_view_config_to_dict(config))


def training_view_id(dataset_id: str, task: str) -> str:
    return make_deterministic_id(
        "trainingview",
        {
            "training_view_dataset_id": dataset_id,
            "task": task,
            "input_groups": TASK_INPUT_GROUPS[task],
            "target_groups": TASK_TARGET_GROUPS[task],
            "mask_version": TRAINING_VIEW_MASK_VERSION,
        },
    )


def training_view_item_id(
    view_id: str,
    task: str,
    entity_id: str,
    split_group_id: str,
) -> str:
    return make_deterministic_id(
        "viewitem",
        {
            "training_view_id": view_id,
            "task": task,
            "entity_id": entity_id,
            "split_group_id": split_group_id,
            "artifact_version": TRAINING_VIEW_ARTIFACT_VERSION,
        },
    )


def _update_array_hash(hasher: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(canonical_json(list(contiguous.shape)).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(contiguous.tobytes(order="C"))
    hasher.update(b"\0")


def training_view_item_content_hash(item: TrainingViewItem) -> str:
    metadata = {
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
        "metadata": json_copy(item.metadata),
    }
    hasher = hashlib.sha256()
    hasher.update(canonical_json(metadata).encode("utf-8"))
    hasher.update(b"\0")
    for name in sorted(item.arrays):
        _update_array_hash(hasher, name, item.arrays[name])
    return f"sha256:{hasher.hexdigest()}"


__all__ = [
    "scientific_training_view_config_payload",
    "training_view_dataset_id",
    "training_view_id",
    "training_view_item_content_hash",
    "training_view_item_id",
    "training_view_operational_config_id",
    "training_view_schema_id",
]
