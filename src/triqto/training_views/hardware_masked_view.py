"""Hardware-masked simulation view with strict Hilbert and topology leakage removal."""
from __future__ import annotations

import numpy as np

from .base_view import make_training_item, unicode_array
from .constants import MANDATORY_ITEM_ARRAY_NAMES
from .context import ViewBuildContext
from .models import TrainingViewItem

_BASE_NAMES = set(MANDATORY_ITEM_ARRAY_NAMES)


def _source_rows(item: TrainingViewItem) -> list[tuple[str, str, str]]:
    return [
        (str(dataset), str(usage), str(reference))
        for dataset, usage, reference in zip(
            item.arrays["source_dataset_names"].tolist(),
            item.arrays["source_usage_names"].tolist(),
            item.arrays["source_refs"].tolist(),
            strict=True,
        )
    ]


def build_hardware_masked_items(
    context: ViewBuildContext,
    joint_items: dict[str, TrainingViewItem],
) -> list[TrainingViewItem]:
    task = "hardware_masked"
    view_id = context.view_ids[task]
    results: list[TrainingViewItem] = []
    input_groups = (
        "circuit_graph",
        "born",
        "parameter",
        "phasor",
        "action_candidates",
        "hilbert_mask",
        "topology",
        "backend",
    )
    head_names = ("diagnosis", "action_ranking", "born_prediction", "topology_audit")
    for sample_id, joint in sorted(joint_items.items()):
        arrays = {
            name: value.copy()
            for name, value in joint.arrays.items()
            if name not in _BASE_NAMES
            and not name.startswith("joint_head_")
            and not name.startswith("topology_")
        }
        parameter_available = bool(
            arrays.get("graph_parameter_names", np.asarray([], dtype="<U1")).size
        )
        topology_safe = bool(
            joint.topology_available_mask
            and not context.sources.topology.config.include_hilbert
        )
        if topology_safe:
            for name, value in joint.arrays.items():
                if name.startswith("topology_"):
                    arrays[name] = value.copy()
        arrays.update(
            {
                "hardware_mask_value": np.asarray([False], dtype=np.bool_),
                "hardware_head_names": unicode_array(head_names),
                "hardware_head_input_group_names": unicode_array(input_groups),
                "hardware_head_input_mask": np.asarray(
                    [
                        [True, True, False, False, False, True, False, False],
                        [True, False, False, False, True, True, False, False],
                        [
                            True,
                            False,
                            parameter_available,
                            parameter_available,
                            False,
                            True,
                            False,
                            False,
                        ],
                        [False, False, False, False, False, True, topology_safe, False],
                    ],
                    dtype=np.bool_,
                ),
                "hardware_head_target_available_mask": np.asarray(
                    [True, True, True, False],
                    dtype=np.bool_,
                ),
            }
        )
        source_refs: list[tuple[str, str, str]] = []
        for dataset, usage, reference in _source_rows(joint):
            if dataset == "phase7" and usage == "input":
                continue
            if dataset == "phase11" and not topology_safe:
                continue
            source_refs.append((dataset, usage, reference))
        privileged = bool(
            np.any(arrays.get("action_privileged_oracle_mask", np.zeros(0, dtype=np.bool_)))
        )
        result = make_training_item(
            dataset_id=context.dataset_id,
            view_id=view_id,
            task=task,
            split=joint.split,
            split_group_id=joint.split_group_id,
            entity_id=sample_id,
            input_available=(
                True,
                True,
                parameter_available,
                parameter_available,
                True,
                True,
                topology_safe,
                False,
            ),
            target_available=(True, True, True, False),
            arrays=arrays,
            source_refs=source_refs,
            hilbert_available=False,
            topology_available=topology_safe,
            privileged_target_available=privileged,
            metadata={
                "sample_id": sample_id,
                "hardware_masked_simulation": True,
                "hardware_data": False,
                "hilbert_values_present": False,
                "hilbert_input_references_present": False,
                "hilbert_mask_signal_present": True,
                "backend_available": False,
                "topology_included_only_when_phase11_was_hilbert_masked": True,
                "phase11_include_hilbert": context.sources.topology.config.include_hilbert,
                "topology_mode": "audit_and_feature_only",
                "topology_loss_weight": 0.0,
                "not_a_hardware_result": True,
            },
            max_source_refs=context.config.max_source_refs_per_item,
        )
        results.append(result)
    return results


__all__ = ["build_hardware_masked_items"]
