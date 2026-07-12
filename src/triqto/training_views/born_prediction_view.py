"""Born-prediction view with target probabilities physically excluded from graph inputs."""
from __future__ import annotations

from .base_view import born_arrays, graph_structure_arrays, make_training_item
from .context import ViewBuildContext
from .models import TrainingViewItem


def build_born_prediction_items(context: ViewBuildContext) -> list[TrainingViewItem]:
    task = "born_prediction"
    view_id = context.view_ids[task]
    items: list[TrainingViewItem] = []
    for sample in sorted(context.sources.phase7.samples, key=lambda value: value.sample_id):
        pair_record = context.pair_records_by_sample_id.get(sample.sample_id)
        if pair_record is None:
            raise ValueError(f"Sample {sample.sample_id} has no Phase 8 graph pair")
        graph = context.sources.graph.graphs_by_id[pair_record.distorted_graph_id]
        graph_record = context.graph_records_by_id[pair_record.distorted_graph_id]
        simulation = context.simulations_by_id.get(sample.distorted_run_id)
        if simulation is None or not simulation.probabilities_ref:
            raise ValueError(
                f"Sample {sample.sample_id} distorted exact run has no probability artifact"
            )
        arrays = graph_structure_arrays(graph)
        arrays.update(
            born_arrays(
                graph.outcome_bitstrings,
                graph.exact_probabilities,
                prefix="born_target",
            )
        )
        parameter_available = bool(graph.parameter_names.size)
        item = make_training_item(
            dataset_id=context.dataset_id,
            view_id=view_id,
            task=task,
            split=context.sample_splits[sample.sample_id],
            split_group_id=context.sample_split_groups[sample.sample_id],
            entity_id=sample.sample_id,
            input_available=(True, parameter_available, parameter_available),
            target_available=(True,),
            arrays=arrays,
            source_refs=(
                ("phase8", "provenance", graph_record.graph_ref),
                ("phase8", "provenance", pair_record.pair_ref),
                ("phase7", "target_provenance", simulation.probabilities_ref),
            ),
            hilbert_available=False,
            topology_available=False,
            privileged_target_available=True,
            metadata={
                "sample_id": sample.sample_id,
                "graph_pair_id": pair_record.graph_pair_id,
                "distorted_graph_id": pair_record.distorted_graph_id,
                "distorted_run_id": sample.distorted_run_id,
                "graph_input_is_materialized_without_born_fields": True,
                "excluded_input_fields": [
                    "outcome_bitstrings",
                    "exact_probabilities",
                    "count_outcome_bitstrings",
                    "supplemental_counts",
                    "born_metric_values",
                ],
                "target_source_is_not_an_input_reference": True,
                "hardware_data": False,
            },
            max_source_refs=context.config.max_source_refs_per_item,
        )
        items.append(item)
    return items


__all__ = ["build_born_prediction_items"]
