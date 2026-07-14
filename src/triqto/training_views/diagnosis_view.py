"""Distortion-diagnosis view builder with labels isolated from inputs."""
from __future__ import annotations

import numpy as np

from .base_view import (
    backend_arrays_from_metadata,
    born_arrays,
    graph_structure_arrays,
    make_training_item,
    strict_float,
    unicode_array,
)
from .context import ViewBuildContext
from .models import TrainingViewItem


def build_diagnosis_items(context: ViewBuildContext) -> list[TrainingViewItem]:
    task = "diagnosis"
    view_id = context.view_ids[task]
    items: list[TrainingViewItem] = []
    for sample in sorted(context.sources.phase7.samples, key=lambda value: value.sample_id):
        pair_record = context.pair_records_by_sample_id.get(sample.sample_id)
        if pair_record is None:
            raise ValueError(f"Sample {sample.sample_id} has no Phase 8 graph pair")
        pair = context.sources.graph.pairs_by_id[pair_record.graph_pair_id]
        distorted_graph = context.sources.graph.graphs_by_id[pair_record.distorted_graph_id]
        graph_record = context.graph_records_by_id[pair_record.distorted_graph_id]
        distortion = context.distortions_by_id.get(sample.distortion_id)
        if distortion is None:
            raise ValueError(f"Sample {sample.sample_id} has no distortion record")
        arrays = graph_structure_arrays(distorted_graph)
        arrays.update(backend_arrays_from_metadata(sample.metadata))
        arrays.update(
            born_arrays(
                distorted_graph.outcome_bitstrings,
                distorted_graph.exact_probabilities,
                prefix="born_input",
            )
        )
        strength_available = distortion.strength is not None
        strength = (
            strict_float(
                distortion.strength,
                f"Distortion {distortion.distortion_id}.strength",
            )
            if strength_available
            else 0.0
        )
        affected = np.zeros(sample.n_qubits, dtype=np.bool_)
        for qubit in distortion.affected_qubits:
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise TypeError("affected_qubits must contain integers and not bool")
            if qubit < 0 or qubit >= sample.n_qubits:
                raise ValueError(
                    f"Distortion {distortion.distortion_id} has out-of-range qubit {qubit}"
                )
            affected[qubit] = True
        arrays.update(
            {
                "diagnosis_distortion_type": unicode_array(
                    [distortion.distortion_type]
                ),
                "diagnosis_strength": np.asarray([strength], dtype=np.float64),
                "diagnosis_strength_available_mask": np.asarray(
                    [strength_available],
                    dtype=np.bool_,
                ),
                "diagnosis_affected_qubit_mask": affected,
                "diagnosis_born_metric_names": pair.born_metric_names.copy(),
                "diagnosis_born_metric_values": pair.born_metric_values.copy(),
                "diagnosis_born_metric_positive_infinity_mask": (
                    pair.born_metric_positive_infinity_mask.copy()
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
            input_available=(True, True, True),
            target_available=(True, strength_available, True),
            arrays=arrays,
            source_refs=(
                ("phase8", "provenance", graph_record.graph_ref),
                ("phase8", "provenance", pair_record.pair_ref),
            ),
            hilbert_available=False,
            topology_available=False,
            privileged_target_available=True,
            metadata={
                "sample_id": sample.sample_id,
                "graph_pair_id": pair_record.graph_pair_id,
                "distorted_graph_id": pair_record.distorted_graph_id,
                "distortion_id": sample.distortion_id,
                "family": sample.family,
                "n_qubits": sample.n_qubits,
                "backend_available": True,
                "backend_id": sample.metadata.get("backend_id"),
                "backend_assignment_level": sample.metadata.get("backend_assignment_level"),
                "identifiability_status": sample.metadata.get("identifiability_status"),
                "identifiability_reason": sample.metadata.get("identifiability_reason"),
                "diagnosis_supervision_mask": sample.metadata.get("diagnosis_supervision_mask"),
                "action_supervision_mask": sample.metadata.get("action_supervision_mask"),
                "hardware_data": False,
                "input_label_separation": (
                    "distortion type, strength, and affected qubits exist only in "
                    "diagnosis_* target arrays"
                ),
            },
            max_source_refs=context.config.max_source_refs_per_item,
        )
        items.append(item)
    return items


__all__ = ["build_diagnosis_items"]
