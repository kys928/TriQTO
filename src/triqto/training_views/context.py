"""Indexed source context shared by Phase 12 task builders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TrainingViewConfig
from .models import TrainingViewSources
from .splits import build_sample_split_maps


@dataclass(slots=True)
class ViewBuildContext:
    sources: TrainingViewSources
    config: TrainingViewConfig
    dataset_id: str
    view_ids: dict[str, str]
    sample_splits: dict[str, str]
    sample_split_groups: dict[str, str]
    samples_by_id: dict[str, Any]
    distortions_by_id: dict[str, Any]
    simulations_by_id: dict[str, Any]
    metrics_by_id: dict[str, Any]
    graph_records_by_id: dict[str, Any]
    pair_records_by_sample_id: dict[str, Any]
    candidate_records_by_action_id: dict[str, Any]
    rollout_records_by_action_id: dict[str, Any]
    action_to_sample: dict[str, str]
    topology_by_sample_id: dict[str, Any]
    topology_record_by_sample_id: dict[str, Any]


def _unique_index(values: list[Any], field: str, kind: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        key = getattr(value, field)
        if not isinstance(key, str) or not key:
            raise ValueError(f"{kind} {field} must be nonblank")
        if key in result:
            raise ValueError(f"Duplicate {kind} {field} {key}")
        result[key] = value
    return result


def build_view_context(
    sources: TrainingViewSources,
    config: TrainingViewConfig,
    dataset_id: str,
    view_ids: dict[str, str],
) -> ViewBuildContext:
    sample_splits, sample_groups = build_sample_split_maps(sources.phase7, config)
    samples = _unique_index(sources.phase7.samples, "sample_id", "sample")
    distortions = _unique_index(
        sources.phase7.distortions,
        "distortion_id",
        "distortion",
    )
    simulations = _unique_index(
        sources.phase7.simulations,
        "run_id",
        "simulation",
    )
    metrics = _unique_index(sources.phase7.metrics, "metric_id", "metric")
    graph_records = _unique_index(
        sources.graph.graph_records,
        "graph_id",
        "graph record",
    )
    candidate_records = _unique_index(
        sources.action.candidate_records,
        "action_id",
        "candidate record",
    )
    rollout_records_by_action: dict[str, Any] = {}
    action_to_sample: dict[str, str] = {}
    for record in sources.action.rollout_records:
        if record.action_id in rollout_records_by_action:
            raise ValueError(f"Duplicate rollout record action_id {record.action_id}")
        rollout_records_by_action[record.action_id] = record
        action_to_sample[record.action_id] = record.sample_id

    topology_by_sample: dict[str, Any] = {}
    topology_record_by_sample: dict[str, Any] = {}
    for group_id, group in sources.topology.groups_by_id.items():
        if group.group_kind != "action_neighborhood":
            continue
        sample_id = group.metadata.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in samples:
            raise ValueError(
                f"Action-neighborhood topology group {group_id} has invalid sample_id"
            )
        if sample_id in topology_by_sample:
            raise ValueError(
                f"Multiple action-neighborhood topology groups for sample {sample_id}"
            )
        topology_by_sample[sample_id] = group
        topology_record_by_sample[sample_id] = sources.topology.records_by_id[group_id]

    return ViewBuildContext(
        sources=sources,
        config=config,
        dataset_id=dataset_id,
        view_ids=dict(view_ids),
        sample_splits=sample_splits,
        sample_split_groups=sample_groups,
        samples_by_id=samples,
        distortions_by_id=distortions,
        simulations_by_id=simulations,
        metrics_by_id=metrics,
        graph_records_by_id=graph_records,
        pair_records_by_sample_id=dict(sources.graph.pair_records_by_sample_id),
        candidate_records_by_action_id=candidate_records,
        rollout_records_by_action_id=rollout_records_by_action,
        action_to_sample=action_to_sample,
        topology_by_sample_id=topology_by_sample,
        topology_record_by_sample_id=topology_record_by_sample,
    )


__all__ = ["ViewBuildContext", "build_view_context"]
