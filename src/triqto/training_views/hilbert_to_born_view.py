"""Optional simulation-only Hilbert-to-Born view builder."""
from __future__ import annotations

from .base_view import born_arrays, make_training_item
from .context import ViewBuildContext
from .models import TrainingViewItem


def build_hilbert_to_born_items(context: ViewBuildContext) -> list[TrainingViewItem]:
    task = "hilbert_to_born"
    if not context.config.include_hilbert:
        return []
    view_id = context.view_ids[task]
    items: list[TrainingViewItem] = []
    for sample in sorted(context.sources.phase7.samples, key=lambda value: value.sample_id):
        simulation = context.simulations_by_id.get(sample.distorted_run_id)
        if simulation is None:
            raise ValueError(f"Sample {sample.sample_id} distorted run is missing")
        if not simulation.statevector_ref:
            continue
        if not simulation.probabilities_ref:
            raise ValueError(
                f"Sample {sample.sample_id} Hilbert input has no Born target artifact"
            )
        pair_record = context.pair_records_by_sample_id.get(sample.sample_id)
        if pair_record is None:
            raise ValueError(f"Sample {sample.sample_id} has no Phase 8 graph pair")
        graph = context.sources.graph.graphs_by_id[pair_record.distorted_graph_id]
        arrays = born_arrays(
            graph.outcome_bitstrings,
            graph.exact_probabilities,
            prefix="born_target",
        )
        item = make_training_item(
            dataset_id=context.dataset_id,
            view_id=view_id,
            task=task,
            split=context.sample_splits[sample.sample_id],
            split_group_id=context.sample_split_groups[sample.sample_id],
            entity_id=sample.sample_id,
            input_available=(True,),
            target_available=(True,),
            arrays=arrays,
            source_refs=(
                ("phase7", "input", simulation.statevector_ref),
                ("phase7", "target_provenance", simulation.probabilities_ref),
            ),
            hilbert_available=True,
            topology_available=False,
            privileged_target_available=True,
            metadata={
                "sample_id": sample.sample_id,
                "distorted_run_id": sample.distorted_run_id,
                "simulation_only": True,
                "statevector_materialized_in_view_artifact": False,
                "statevector_loaded_by_future_dataloader_from_input_ref": True,
                "hardware_compatible": False,
                "hardware_data": False,
            },
            max_source_refs=context.config.max_source_refs_per_item,
        )
        items.append(item)
    if not items and not context.config.allow_empty_hilbert_view:
        raise ValueError(
            "No Phase 7 distorted statevector artifacts are available for hilbert_to_born"
        )
    return items


__all__ = ["build_hilbert_to_born_items"]
