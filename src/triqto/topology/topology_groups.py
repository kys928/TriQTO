"""Deterministic point-cloud grouping for the Phase 11 topology audit."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import TopologyAuditConfig


@dataclass(frozen=True, slots=True)
class TopologyGroupSpec:
    """One deterministic group definition before point-cloud materialization."""

    group_kind: str
    group_key: str
    point_ids: tuple[str, ...]
    metadata: dict[str, Any]


def _distortion_index(sources: Any) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for record in sources.phase7.distortions:
        if record.distortion_id in indexed:
            raise ValueError(f"Duplicate distortion ID {record.distortion_id}")
        indexed[record.distortion_id] = record
    return indexed


def _append_if_eligible(
    output: list[TopologyGroupSpec],
    *,
    group_kind: str,
    group_key: str,
    point_ids: list[str],
    metadata: dict[str, Any],
    config: TopologyAuditConfig,
) -> None:
    ordered = tuple(sorted(point_ids))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"Topology group {group_key} contains duplicate point IDs")
    if len(ordered) < config.min_points:
        return
    if len(ordered) > config.max_points_per_group:
        raise RuntimeError(
            f"Topology group {group_key!r} has {len(ordered)} points, exceeding "
            f"max_points_per_group={config.max_points_per_group}"
        )
    output.append(
        TopologyGroupSpec(
            group_kind=group_kind,
            group_key=group_key,
            point_ids=ordered,
            metadata=dict(metadata),
        )
    )


def _action_ids_for_sample(action_source: Any, sample_id: str) -> list[str]:
    lazy_method = getattr(action_source, "action_ids_for_sample", None)
    if callable(lazy_method):
        return list(lazy_method(sample_id))
    rollouts = action_source.rollouts_by_sample_id.get(sample_id)
    if rollouts is None:
        raise ValueError(f"Phase 7 sample {sample_id} has no Phase 9 rollout neighborhood")
    return [rollout.action_id for rollout in rollouts]


def build_topology_group_specs(
    sources: Any,
    config: TopologyAuditConfig,
) -> tuple[list[TopologyGroupSpec], dict[str, int]]:
    """Build action neighborhoods and deterministic Phase 7 cohort groups."""
    if not isinstance(config, TopologyAuditConfig):
        raise TypeError("config must be TopologyAuditConfig")
    specs: list[TopologyGroupSpec] = []
    skipped_counts: dict[str, int] = defaultdict(int)

    if "action_neighborhood" in config.group_kinds:
        total_samples = len(sources.phase7.samples)
        for sample_index, sample in enumerate(
            sorted(sources.phase7.samples, key=lambda item: item.sample_id),
            start=1,
        ):
            point_ids = _action_ids_for_sample(sources.action, sample.sample_id)
            if sample_index == 1 or sample_index % 250 == 0 or sample_index == total_samples:
                print(
                    "[Phase 11][group-plan] "
                    f"action neighborhoods indexed {sample_index:,}/{total_samples:,} | "
                    f"latest_points={len(point_ids):,}",
                    flush=True,
                )
            before = len(specs)
            _append_if_eligible(
                specs,
                group_kind="action_neighborhood",
                group_key=f"sample={sample.sample_id}",
                point_ids=point_ids,
                metadata={
                    "sample_id": sample.sample_id,
                    "family": sample.family,
                    "n_qubits": sample.n_qubits,
                    "distortion_id": sample.distortion_id,
                    "point_semantics": "phase9_candidate_action_rollouts",
                    "action_source_mode": (
                        "lazy_per_sample" if getattr(sources.action, "is_lazy", False)
                        else "fully_materialized"
                    ),
                },
                config=config,
            )
            if len(specs) == before:
                skipped_counts["action_neighborhood_below_min_points"] += 1

    if "family_qubit_cohort" in config.group_kinds:
        grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
        for sample in sources.phase7.samples:
            grouped[(sample.family, sample.n_qubits)].append(sample)
        for (family, n_qubits), samples in sorted(grouped.items()):
            before = len(specs)
            _append_if_eligible(
                specs,
                group_kind="family_qubit_cohort",
                group_key=f"family={family}|n_qubits={n_qubits}",
                point_ids=[sample.sample_id for sample in samples],
                metadata={
                    "family": family,
                    "n_qubits": n_qubits,
                    "point_semantics": "phase7_distorted_samples",
                    "distortion_mixed": True,
                },
                config=config,
            )
            if len(specs) == before:
                skipped_counts["family_qubit_cohort_below_min_points"] += 1

    if "family_qubit_distortion_cohort" in config.group_kinds:
        distortions = _distortion_index(sources)
        grouped_distortion: dict[tuple[str, int, str], list[Any]] = defaultdict(list)
        for sample in sources.phase7.samples:
            distortion = distortions.get(sample.distortion_id)
            if distortion is None:
                raise ValueError(
                    f"Sample {sample.sample_id} references missing distortion "
                    f"{sample.distortion_id}"
                )
            grouped_distortion[
                (sample.family, sample.n_qubits, distortion.distortion_type)
            ].append(sample)
        for (family, n_qubits, distortion_type), samples in sorted(
            grouped_distortion.items()
        ):
            before = len(specs)
            _append_if_eligible(
                specs,
                group_kind="family_qubit_distortion_cohort",
                group_key=(
                    f"family={family}|n_qubits={n_qubits}|"
                    f"distortion_type={distortion_type}"
                ),
                point_ids=[sample.sample_id for sample in samples],
                metadata={
                    "family": family,
                    "n_qubits": n_qubits,
                    "distortion_type": distortion_type,
                    "point_semantics": "phase7_distorted_samples",
                    "distortion_mixed": False,
                },
                config=config,
            )
            if len(specs) == before:
                skipped_counts[
                    "family_qubit_distortion_cohort_below_min_points"
                ] += 1

    specs.sort(key=lambda item: (item.group_kind, item.group_key, item.point_ids))
    seen_keys: set[tuple[str, str]] = set()
    for spec in specs:
        key = (spec.group_kind, spec.group_key)
        if key in seen_keys:
            raise ValueError(f"Duplicate topology group definition {key}")
        seen_keys.add(key)
    if len(specs) > config.max_groups:
        raise RuntimeError(
            f"Topology group count {len(specs)} exceeds max_groups={config.max_groups}"
        )
    return specs, dict(sorted(skipped_counts.items()))


__all__ = ["TopologyGroupSpec", "build_topology_group_specs"]
