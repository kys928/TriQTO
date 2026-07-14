"""Joint multitask view with explicit per-head input and target masks."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .base_view import make_training_item, sample_scientific_metadata, unicode_array
from .constants import MANDATORY_ITEM_ARRAY_NAMES
from .context import ViewBuildContext
from .models import TrainingViewItem

_BASE_NAMES = set(MANDATORY_ITEM_ARRAY_NAMES)


def _task_arrays(item: TrainingViewItem) -> dict[str, np.ndarray]:
    return {name: value for name, value in item.arrays.items() if name not in _BASE_NAMES}


def _source_rows(item: TrainingViewItem) -> list[tuple[str, str, str]]:
    datasets = item.arrays["source_dataset_names"].tolist()
    usages = item.arrays["source_usage_names"].tolist()
    refs = item.arrays["source_refs"].tolist()
    return [
        (str(dataset), str(usage), str(reference))
        for dataset, usage, reference in zip(datasets, usages, refs, strict=True)
    ]


def _merge_arrays(items: list[TrainingViewItem]) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for item in items:
        for name, value in _task_arrays(item).items():
            if name not in merged:
                merged[name] = value.copy()
            elif not np.array_equal(merged[name], value):
                raise ValueError(
                    f"Joint multitask source items disagree on duplicate array {name}"
                )
    return merged


def build_joint_multitask_items(
    context: ViewBuildContext,
    *,
    diagnosis_items: Mapping[str, TrainingViewItem],
    action_items: Mapping[str, TrainingViewItem],
    born_items: Mapping[str, TrainingViewItem],
    hilbert_items: Mapping[str, TrainingViewItem],
    topology_items: Mapping[str, TrainingViewItem],
) -> list[TrainingViewItem]:
    task = "joint_multitask"
    view_id = context.view_ids[task]
    results: list[TrainingViewItem] = []
    joint_input_groups = (
        "circuit_graph",
        "born",
        "parameter",
        "phasor",
        "action_candidates",
        "hilbert",
        "topology",
        "backend",
    )
    head_names = (
        "diagnosis",
        "action_ranking",
        "born_prediction",
        "hilbert_to_born",
        "topology_audit",
    )
    for sample in sorted(context.sources.phase7.samples, key=lambda value: value.sample_id):
        sample_id = sample.sample_id
        required = [
            diagnosis_items[sample_id],
            action_items[sample_id],
            born_items[sample_id],
        ]
        hilbert = hilbert_items.get(sample_id)
        topology = topology_items.get(sample_id)
        source_items = [*required]
        if hilbert is not None:
            source_items.append(hilbert)
        if topology is not None:
            source_items.append(topology)
        arrays = _merge_arrays(source_items)
        parameter_available = bool(
            arrays.get("graph_parameter_names", np.asarray([], dtype="<U1")).size
        )
        topology_available = topology is not None
        hilbert_available = hilbert is not None
        diagnosis_supervised = bool(arrays["diagnosis_supervision_mask"].reshape(-1)[0])
        action_supervised = bool(arrays["action_supervision_mask"].reshape(-1)[0])
        head_input_mask = np.asarray(
            [
                [True, True, False, False, False, False, False, False],
                [True, False, False, False, True, False, False, False],
                [
                    True,
                    False,
                    parameter_available,
                    parameter_available,
                    False,
                    False,
                    False,
                    False,
                ],
                [False, False, False, False, False, hilbert_available, False, False],
                [False, False, False, False, False, False, topology_available, False],
            ],
            dtype=np.bool_,
        )
        head_target_mask = np.asarray(
            [diagnosis_supervised, action_supervised, True, hilbert_available, False],
            dtype=np.bool_,
        )
        arrays.update(
            {
                "joint_head_names": unicode_array(head_names),
                "joint_head_input_group_names": unicode_array(joint_input_groups),
                "joint_head_input_mask": head_input_mask,
                "joint_head_target_available_mask": head_target_mask,
            }
        )
        source_refs = [row for item in source_items for row in _source_rows(item)]
        privileged = bool(
            np.any(arrays.get("action_privileged_oracle_mask", np.zeros(0, dtype=np.bool_)))
        )
        result = make_training_item(
            dataset_id=context.dataset_id,
            view_id=view_id,
            task=task,
            split=context.sample_splits[sample_id],
            split_group_id=context.sample_split_groups[sample_id],
            entity_id=sample_id,
            input_available=(
                True,
                True,
                parameter_available,
                parameter_available,
                True,
                hilbert_available,
                topology_available,
                False,
            ),
            target_available=(
                diagnosis_supervised,
                action_supervised,
                True,
                hilbert_available,
                False,
            ),
            arrays=arrays,
            source_refs=source_refs,
            hilbert_available=hilbert_available,
            topology_available=topology_available,
            privileged_target_available=privileged,
            metadata={
                **sample_scientific_metadata(context, sample),
                "sample_id": sample_id,
                "head_specific_mask_enforcement_required": True,
                "born_input_is_for_diagnosis_not_born_prediction": True,
                "born_prediction_head_masks_born_input": True,
                "action_head_masks_rollout_target_provenance": True,
                "topology_mode": "audit_and_feature_only",
                "topology_loss_weight": 0.0,
                "topology_supervised_target_present": False,
                "backend_available": False,
                "hardware_data": False,
                "identifiability_status": sample.identifiability_status,
                "identifiability_reason": sample.identifiability_reason,
            },
            max_source_refs=context.config.max_source_refs_per_item,
        )
        results.append(result)
    return results


__all__ = ["build_joint_multitask_items"]
