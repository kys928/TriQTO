"""Integrity and leakage validation for Phase 12 training views."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import numpy as np

from triqto.graph.utils import normalize_relative_posix_ref
from triqto.storage.training_view_schema import (
    TrainingViewDefinitionRecordV1,
    TrainingViewItemRecordV1,
)

from .config import TrainingViewConfig
from .constants import (
    MANDATORY_ITEM_ARRAY_NAMES,
    SOURCE_DATASET_NAMES,
    SOURCE_USAGE_NAMES,
    SPLIT_ORDER,
    TASK_INPUT_GROUPS,
    TASK_ORDER,
    TASK_TARGET_GROUPS,
)
from .identities import training_view_item_content_hash, training_view_item_id
from .models import TrainingViewItem

_ARRAY_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


def _unicode_vector(value: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind != "U":
        raise TypeError(f"{name} must be a one-dimensional fixed-width Unicode array")
    if not allow_empty and value.size == 0:
        raise ValueError(f"{name} must not be empty")
    if any(not str(item) for item in value.tolist()):
        raise ValueError(f"{name} must contain nonblank strings")
    return value


def _bool_array(value: Any, name: str, ndim: int = 1) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
        raise TypeError(f"{name} must use bool dtype")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return value


def _validate_all_arrays(item: TrainingViewItem) -> None:
    if set(MANDATORY_ITEM_ARRAY_NAMES) - set(item.arrays):
        raise ValueError("Training item is missing mandatory base arrays")
    for name, value in item.arrays.items():
        if not isinstance(name, str) or not _ARRAY_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid training item array name {name!r}")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Training item array {name} must be a NumPy array")
        if value.dtype.kind == "O":
            raise TypeError(f"Training item array {name} must not use object dtype")
        if value.dtype.kind in {"f", "c"}:
            if not np.isfinite(value.real).all():
                raise ValueError(f"Training item array {name} contains non-finite values")
            if value.dtype.kind == "c" and not np.isfinite(value.imag).all():
                raise ValueError(f"Training item array {name} contains non-finite values")


def _validate_source_rows(item: TrainingViewItem, config: TrainingViewConfig) -> None:
    datasets = _unicode_vector(
        item.arrays["source_dataset_names"],
        "source_dataset_names",
        allow_empty=True,
    ).tolist()
    usages = _unicode_vector(
        item.arrays["source_usage_names"],
        "source_usage_names",
        allow_empty=True,
    ).tolist()
    refs = _unicode_vector(
        item.arrays["source_refs"],
        "source_refs",
        allow_empty=True,
    ).tolist()
    if not (len(datasets) == len(usages) == len(refs)):
        raise ValueError("Source dataset/usage/reference arrays must have equal length")
    if len(refs) > config.max_source_refs_per_item:
        raise RuntimeError("Training item source reference count exceeds guardrail")
    rows = list(zip(datasets, usages, refs, strict=True))
    if rows != sorted(set(rows)):
        raise ValueError("Source reference rows must be sorted and unique")
    for dataset, usage, reference in rows:
        if dataset not in SOURCE_DATASET_NAMES:
            raise ValueError(f"Unknown source dataset {dataset!r}")
        if usage not in SOURCE_USAGE_NAMES:
            raise ValueError(f"Unknown source usage {usage!r}")
        normalize_relative_posix_ref(reference, "source reference")
    if item.task in {"born_prediction", "hilbert_to_born"}:
        if any(
            dataset == "phase7" and usage == "input" and "probabilit" in reference
            for dataset, usage, reference in rows
        ):
            raise ValueError("Born target probability artifacts cannot be input references")
    if item.task == "hardware_masked":
        if any(dataset == "phase7" and usage == "input" for dataset, usage, _ in rows):
            raise ValueError("Hardware-masked items must not contain Phase 7 Hilbert inputs")
        if item.metadata.get("phase11_include_hilbert") is True and any(
            dataset == "phase11" for dataset, _, _ in rows
        ):
            raise ValueError(
                "Hardware-masked items cannot use topology built with Hilbert access"
            )


def _validate_action_arrays(item: TrainingViewItem, config: TrainingViewConfig) -> None:
    ids = _unicode_vector(item.arrays.get("action_candidate_ids"), "action_candidate_ids")
    values = [str(value) for value in ids.tolist()]
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("action_candidate_ids must be sorted and unique")
    count = ids.size
    if count > config.max_candidates_per_item:
        raise RuntimeError("Action candidate count exceeds guardrail")
    feature_matrix = item.arrays.get("action_candidate_features")
    if not isinstance(feature_matrix, np.ndarray) or feature_matrix.dtype != np.float64:
        raise TypeError("action_candidate_features must be float64")
    if feature_matrix.ndim != 2 or feature_matrix.shape[0] != count:
        raise ValueError("action_candidate_features row count mismatch")
    for name, dtype in (
        ("action_target_rank", np.int64),
        ("action_target_reward", np.float64),
        ("action_target_selected_mask", np.bool_),
        ("action_target_dominates_baseline_mask", np.bool_),
        ("action_target_primary_metric_nonworsening_mask", np.bool_),
        ("action_privileged_oracle_mask", np.bool_),
    ):
        array = item.arrays.get(name)
        if not isinstance(array, np.ndarray) or array.dtype != dtype:
            raise TypeError(f"{name} has incorrect dtype")
        if array.ndim != 1 or array.size != count:
            raise ValueError(f"{name} candidate length mismatch")
    if int(np.count_nonzero(item.arrays["action_target_selected_mask"])) != 1:
        raise ValueError("Exactly one action candidate must be selected")
    ranks = item.arrays["action_target_rank"]
    if sorted(ranks.tolist()) != list(range(1, count + 1)):
        raise ValueError("Action target ranks must be a complete 1..N permutation")
    edit_ptr = item.arrays.get("action_edit_ptr")
    if not isinstance(edit_ptr, np.ndarray) or edit_ptr.dtype != np.int64:
        raise TypeError("action_edit_ptr must be int64")
    if edit_ptr.shape != (count + 1,) or edit_ptr[0] != 0:
        raise ValueError("action_edit_ptr must have candidate_count+1 entries starting at zero")
    if np.any(np.diff(edit_ptr) < 0):
        raise ValueError("action_edit_ptr must be nondecreasing")
    edit_types = _unicode_vector(
        item.arrays.get("action_edit_types"),
        "action_edit_types",
        allow_empty=True,
    )
    if int(edit_ptr[-1]) != edit_types.size:
        raise ValueError("action_edit_ptr terminal value mismatch")


def validate_training_view_item(
    item: TrainingViewItem,
    config: TrainingViewConfig,
    *,
    require_hash: bool = True,
) -> None:
    if not isinstance(item, TrainingViewItem):
        raise TypeError("item must be TrainingViewItem")
    if item.task not in TASK_ORDER:
        raise ValueError(f"Unknown training-view task {item.task!r}")
    for name in (
        "view_item_id",
        "training_view_id",
        "training_view_dataset_id",
        "split_group_id",
        "entity_id",
    ):
        value = getattr(item, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"TrainingViewItem.{name} must be nonblank")
    if item.split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split {item.split!r}")
    if item.input_groups != TASK_INPUT_GROUPS[item.task]:
        raise ValueError("Training item input group contract mismatch")
    if item.target_groups != TASK_TARGET_GROUPS[item.task]:
        raise ValueError("Training item target group contract mismatch")
    expected_id = training_view_item_id(
        item.training_view_id,
        item.task,
        item.entity_id,
        item.split_group_id,
    )
    if item.view_item_id != expected_id:
        raise ValueError("Training item identity mismatch")
    _validate_all_arrays(item)
    input_names = _unicode_vector(item.arrays["input_group_names"], "input_group_names")
    target_names = _unicode_vector(item.arrays["target_group_names"], "target_group_names")
    if tuple(input_names.tolist()) != item.input_groups:
        raise ValueError("input_group_names array mismatch")
    if tuple(target_names.tolist()) != item.target_groups:
        raise ValueError("target_group_names array mismatch")
    input_mask = _bool_array(
        item.arrays["input_group_available_mask"],
        "input_group_available_mask",
    )
    target_mask = _bool_array(
        item.arrays["target_group_available_mask"],
        "target_group_available_mask",
    )
    if input_mask.size != len(item.input_groups) or target_mask.size != len(item.target_groups):
        raise ValueError("Group availability mask length mismatch")
    for name in (
        "hilbert_available_mask",
        "topology_available_mask",
        "privileged_target_available_mask",
    ):
        if not isinstance(getattr(item, name), bool):
            raise TypeError(f"{name} must be bool")
    if not isinstance(item.metadata, Mapping):
        raise TypeError("TrainingViewItem.metadata must be a mapping")
    if item.metadata.get("topology_loss_weight", 0.0) != 0.0:
        raise ValueError("Phase 12 topology_loss_weight must remain zero")
    _validate_source_rows(item, config)

    if item.task == "diagnosis":
        if "born_input_probabilities" not in item.arrays:
            raise ValueError("Diagnosis item must contain Born input evidence")
        if "diagnosis_distortion_type" not in item.arrays:
            raise ValueError("Diagnosis item must contain distortion targets")
    elif item.task == "action_ranking":
        _validate_action_arrays(item, config)
    elif item.task == "born_prediction":
        if any(name.startswith("born_input_") for name in item.arrays):
            raise ValueError("Born-prediction item must not contain Born inputs")
        if "born_target_probabilities" not in item.arrays:
            raise ValueError("Born-prediction item must contain Born targets")
    elif item.task == "hilbert_to_born":
        if not item.hilbert_available_mask:
            raise ValueError("Hilbert-to-Born items require Hilbert availability")
        if "born_target_probabilities" not in item.arrays:
            raise ValueError("Hilbert-to-Born item must contain Born targets")
    elif item.task == "topology_audit":
        if not item.topology_available_mask:
            raise ValueError("Topology audit item requires topology availability")
        if bool(target_mask[0]):
            raise ValueError("Topology audit remains unsupervised/audit-only in Phase 12")
        if item.metadata.get("topology_loss_weight") != 0.0:
            raise ValueError("Topology audit item must record zero topology loss")
    elif item.task == "joint_multitask":
        mask = _bool_array(item.arrays.get("joint_head_input_mask"), "joint_head_input_mask", 2)
        if mask.shape[0] != 5:
            raise ValueError("joint_head_input_mask must contain five head rows")
        if item.metadata.get("head_specific_mask_enforcement_required") is not True:
            raise ValueError("Joint items must require head-specific mask enforcement")
    elif item.task == "hardware_masked":
        if item.hilbert_available_mask:
            raise ValueError("Hardware-masked item must have Hilbert unavailable")
        if item.metadata.get("hardware_masked_simulation") is not True:
            raise ValueError("Hardware-masked item must identify masked simulation mode")
        if item.metadata.get("hardware_data") is not False:
            raise ValueError("Hardware-masked simulation must not claim hardware data")
        if item.metadata.get("hilbert_values_present") is not False:
            raise ValueError("Hardware-masked item must not contain Hilbert values")

    expected_hash = training_view_item_content_hash(item)
    if require_hash and item.content_hash != expected_hash:
        raise ValueError("Training item content_hash mismatch")
    if not require_hash and item.content_hash not in {"", expected_hash}:
        raise ValueError("Training item content_hash is malformed")


def validate_training_view_dataset_joins(
    definition_records: Sequence[TrainingViewDefinitionRecordV1],
    item_records: Sequence[TrainingViewItemRecordV1],
    *,
    items_by_id: Mapping[str, TrainingViewItem] | None,
    config: TrainingViewConfig,
) -> None:
    definitions: dict[str, TrainingViewDefinitionRecordV1] = {}
    task_to_view: dict[str, str] = {}
    for record in definition_records:
        record.validate()
        if record.training_view_id in definitions:
            raise ValueError(f"Duplicate training_view_id {record.training_view_id}")
        if record.task in task_to_view:
            raise ValueError(f"Duplicate training-view task {record.task}")
        definitions[record.training_view_id] = record
        task_to_view[record.task] = record.training_view_id
    if tuple(task for task in TASK_ORDER if task in task_to_view) != config.tasks:
        raise ValueError("Training-view definition task coverage/order mismatch")

    records: dict[str, TrainingViewItemRecordV1] = {}
    artifact_refs: set[str] = set()
    split_groups: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {view_id: {} for view_id in definitions}
    for record in item_records:
        record.validate()
        if record.view_item_id in records:
            raise ValueError(f"Duplicate view_item_id {record.view_item_id}")
        if record.artifact_ref in artifact_refs:
            raise ValueError(f"Duplicate training item artifact_ref {record.artifact_ref}")
        definition = definitions.get(record.training_view_id)
        if definition is None:
            raise ValueError(
                f"Training item {record.view_item_id} references missing view "
                f"{record.training_view_id}"
            )
        if record.task != definition.task:
            raise ValueError("Training item task does not match view definition")
        previous = split_groups.setdefault(record.split_group_id, record.split)
        if record.split != previous and record.split != "audit_only" and previous != "audit_only":
            raise ValueError(
                f"Split group {record.split_group_id} appears in multiple trainable splits"
            )
        records[record.view_item_id] = record
        artifact_refs.add(record.artifact_ref)
        counts[record.training_view_id][record.split] = (
            counts[record.training_view_id].get(record.split, 0) + 1
        )
    for view_id, definition in definitions.items():
        if definition.item_count != sum(counts[view_id].values()):
            raise ValueError(f"View {view_id} item_count mismatch")
        if dict(definition.split_counts) != dict(sorted(counts[view_id].items())):
            raise ValueError(f"View {view_id} split_counts mismatch")

    if items_by_id is not None:
        if set(items_by_id) != set(records):
            raise ValueError("Training item manifest IDs do not match artifact IDs")
        for item_id, item in items_by_id.items():
            validate_training_view_item(item, config, require_hash=True)
            record = records[item_id]
            expected = {
                "training_view_id": item.training_view_id,
                "training_view_dataset_id": item.training_view_dataset_id,
                "task": item.task,
                "split": item.split,
                "split_group_id": item.split_group_id,
                "entity_id": item.entity_id,
                "input_groups": list(item.input_groups),
                "target_groups": list(item.target_groups),
                "content_hash": item.content_hash,
                "hilbert_available_mask": item.hilbert_available_mask,
                "topology_available_mask": item.topology_available_mask,
                "privileged_target_available_mask": item.privileged_target_available_mask,
            }
            for name, value in expected.items():
                if getattr(record, name) != value:
                    raise ValueError(f"Training item record {item_id} {name} mismatch")


__all__ = [
    "validate_training_view_dataset_joins",
    "validate_training_view_item",
]
