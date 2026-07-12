"""Topology audit view builder with cross-split cohorts kept audit-only."""
from __future__ import annotations

import numpy as np

from triqto.core.ids import make_deterministic_id

from .base_view import make_training_item
from .context import ViewBuildContext
from .models import TrainingViewItem
from .splits import topology_group_split


def _source_sample_ids(group: object, context: ViewBuildContext) -> list[str]:
    group_kind = getattr(group, "group_kind")
    if group_kind == "action_neighborhood":
        sample_id = getattr(group, "metadata").get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in context.samples_by_id:
            raise ValueError("Action-neighborhood topology group has invalid sample_id")
        return [sample_id]
    sample_ids = [str(value) for value in getattr(group, "point_ids").tolist()]
    for sample_id in sample_ids:
        if sample_id not in context.samples_by_id:
            raise ValueError(
                f"Topology cohort point {sample_id} is not a Phase 7 sample ID"
            )
    return sorted(sample_ids)


def build_topology_audit_items(context: ViewBuildContext) -> list[TrainingViewItem]:
    task = "topology_audit"
    if not context.config.include_topology:
        return []
    view_id = context.view_ids[task]
    items: list[TrainingViewItem] = []
    for group_id, group in sorted(context.sources.topology.groups_by_id.items()):
        record = context.sources.topology.records_by_id[group_id]
        split, source_splits = topology_group_split(
            group,
            sample_splits=context.sample_splits,
            action_to_sample=context.action_to_sample,
            config=context.config,
        )
        sample_ids = _source_sample_ids(group, context)
        source_split_groups = sorted(
            {context.sample_split_groups[sample_id] for sample_id in sample_ids}
        )
        split_group_id = (
            source_split_groups[0]
            if len(source_split_groups) == 1
            else make_deterministic_id(
                "viewsplitgroup",
                {
                    "topology_group_id": group_id,
                    "source_split_group_ids": source_split_groups,
                },
            )
        )
        arrays: dict[str, np.ndarray] = {
            "topology_point_ids": group.point_ids.copy(),
            "topology_manifold_available_mask": group.manifold_available_mask.copy(),
            "topology_feature_names": group.topology_feature_names.copy(),
            "topology_feature_values": group.topology_feature_values.copy(),
            "topology_alignment_feature_names": group.alignment_feature_names.copy(),
            "topology_alignment_feature_values": group.alignment_feature_values.copy(),
        }
        for manifold, summary in sorted(group.persistence.items()):
            arrays[f"topology_{manifold}_feature_names"] = summary.feature_names.copy()
            arrays[f"topology_{manifold}_feature_values"] = summary.feature_values.copy()
        item = make_training_item(
            dataset_id=context.dataset_id,
            view_id=view_id,
            task=task,
            split=split,
            split_group_id=split_group_id,
            entity_id=group_id,
            input_available=(True, False),
            target_available=(False,),
            arrays=arrays,
            source_refs=(("phase11", "audit", record.artifact_ref),),
            hilbert_available=bool(group.manifold_available_mask[1]),
            topology_available=True,
            privileged_target_available=False,
            metadata={
                "topology_group_id": group_id,
                "group_kind": group.group_kind,
                "group_key": group.group_key,
                "source_sample_ids": sample_ids,
                "source_split_group_ids": source_split_groups,
                "source_splits": list(source_splits),
                "cross_split_group": len(source_splits) > 1,
                "cross_split_policy": context.config.topology_cross_split_policy,
                "topology_mode": "audit_and_feature_only",
                "topology_loss_weight": 0.0,
                "supervised_topology_target_present": False,
                "hardware_data": False,
            },
            max_source_refs=context.config.max_source_refs_per_item,
        )
        items.append(item)
    return items


__all__ = ["build_topology_audit_items"]
