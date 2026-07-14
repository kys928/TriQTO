"""Action-ranking view with safe candidate inputs and isolated rollout targets."""
from __future__ import annotations

import numpy as np

from .base_view import graph_structure_arrays, make_training_item, unicode_array
from .context import ViewBuildContext
from .models import TrainingViewItem


def build_action_ranking_items(context: ViewBuildContext) -> list[TrainingViewItem]:
    task = "action_ranking"
    view_id = context.view_ids[task]
    items: list[TrainingViewItem] = []
    for sample in sorted(context.sources.phase7.samples, key=lambda value: value.sample_id):
        pair_record = context.pair_records_by_sample_id.get(sample.sample_id)
        if pair_record is None:
            raise ValueError(f"Sample {sample.sample_id} has no Phase 8 graph pair")
        distorted_graph = context.sources.graph.graphs_by_id[pair_record.distorted_graph_id]
        graph_record = context.graph_records_by_id[pair_record.distorted_graph_id]
        rollouts = list(context.sources.action.rollouts_by_sample_id[sample.sample_id])
        rollouts.sort(key=lambda value: value.action_id)
        if len(rollouts) > context.config.max_candidates_per_item:
            raise RuntimeError(
                f"Sample {sample.sample_id} has {len(rollouts)} candidates, exceeding "
                f"max_candidates_per_item={context.config.max_candidates_per_item}"
            )
        candidate_ids: list[str] = []
        features: list[list[float]] = []
        ranks: list[int] = []
        rewards: list[float] = []
        selected: list[bool] = []
        dominates: list[bool] = []
        nonworsening: list[bool] = []
        privileged: list[bool] = []
        edit_ptr = [0]
        edit_types: list[str] = []
        edit_magnitudes: list[float] = []
        edit_qubit_ptr = [0]
        edit_qubits: list[int] = []
        source_refs: list[tuple[str, str, str]] = [
            ("phase8", "provenance", graph_record.graph_ref),
            ("phase8", "provenance", pair_record.pair_ref),
        ]
        for rollout in rollouts:
            candidate = context.sources.action.candidates_by_id[rollout.action_id]
            candidate_record = context.candidate_records_by_action_id[rollout.action_id]
            rollout_record = context.rollout_records_by_action_id[rollout.action_id]
            is_privileged = "oracle_inverse" in candidate.generation_sources
            candidate_ids.append(rollout.action_id)
            features.append(
                [
                    float(len(candidate.edits)),
                    float(candidate.risk_score),
                    float(rollout.depth_delta),
                    float(rollout.gate_delta),
                    float(len(candidate.edits) == 0),
                ]
            )
            ranks.append(int(rollout.rank))
            rewards.append(float(rollout.reward))
            selected.append(bool(rollout.selected))
            dominates.append(bool(rollout.dominates_baseline))
            nonworsening.append(bool(rollout.primary_metric_nonworsening))
            privileged.append(is_privileged)
            for edit in candidate.edits:
                edit_types.append(edit.edit_type)
                edit_magnitudes.append(float(edit.magnitude))
                edit_qubits.extend(int(qubit) for qubit in edit.qubits)
                edit_qubit_ptr.append(len(edit_qubits))
            edit_ptr.append(len(edit_types))
            source_refs.extend(
                (
                    ("phase9", "input", candidate_record.circuit_ref),
                    ("phase9", "provenance", candidate_record.action_ref),
                    ("phase9", "target_provenance", rollout_record.rollout_ref),
                )
            )
        if not candidate_ids:
            raise ValueError(f"Sample {sample.sample_id} has no action candidates")
        if sum(selected) != 1:
            raise ValueError(
                f"Sample {sample.sample_id} must have exactly one selected action target"
            )
        arrays = graph_structure_arrays(distorted_graph)
        arrays.update(
            {
                "action_candidate_ids": unicode_array(candidate_ids),
                "action_candidate_feature_names": unicode_array(
                    (
                        "edit_count",
                        "risk_score",
                        "depth_delta",
                        "gate_delta",
                        "is_no_op",
                    )
                ),
                "action_candidate_features": np.asarray(features, dtype=np.float64),
                "action_edit_ptr": np.asarray(edit_ptr, dtype=np.int64),
                "action_edit_types": unicode_array(edit_types),
                "action_edit_magnitudes": np.asarray(
                    edit_magnitudes,
                    dtype=np.float64,
                ),
                "action_edit_qubit_ptr": np.asarray(edit_qubit_ptr, dtype=np.int64),
                "action_edit_qubits": np.asarray(edit_qubits, dtype=np.int64),
                "action_target_rank": np.asarray(ranks, dtype=np.int64),
                "action_target_reward": np.asarray(rewards, dtype=np.float64),
                "action_target_selected_mask": np.asarray(selected, dtype=np.bool_),
                "action_target_dominates_baseline_mask": np.asarray(
                    dominates,
                    dtype=np.bool_,
                ),
                "action_target_primary_metric_nonworsening_mask": np.asarray(
                    nonworsening,
                    dtype=np.bool_,
                ),
                "action_privileged_oracle_mask": np.asarray(
                    privileged,
                    dtype=np.bool_,
                ),
            }
        )
        item = make_training_item(
            dataset_id=context.dataset_id,
            view_id=view_id,
            task=task,
            split=context.sample_splits[sample.sample_id],
            split_group_id=context.sample_split_groups[sample.sample_id],
            entity_id=sample.sample_id,
            input_available=(True, True, False),
            target_available=(True, True, True),
            arrays=arrays,
            source_refs=source_refs,
            hilbert_available=False,
            topology_available=False,
            privileged_target_available=any(privileged),
            metadata={
                "sample_id": sample.sample_id,
                "graph_pair_id": pair_record.graph_pair_id,
                "candidate_order": "sorted_action_id_not_target_rank",
                "candidate_count": len(candidate_ids),
                "metric_context_available": False,
                "clean_target_metrics_are_inputs": False,
                "rollout_artifacts_are_target_provenance_only": True,
                "generation_sources_excluded_from_candidate_inputs": True,
                "privileged_oracle_candidates_retained_with_explicit_mask": True,
                "identifiability_status": sample.metadata.get("identifiability_status"),
                "identifiability_reason": sample.metadata.get("identifiability_reason"),
                "diagnosis_supervision_mask": sample.metadata.get("diagnosis_supervision_mask"),
                "action_supervision_mask": sample.metadata.get("action_supervision_mask"),
                "hardware_data": False,
            },
            max_source_refs=context.config.max_source_refs_per_item,
        )
        items.append(item)
    return items


__all__ = ["build_action_ranking_items"]
