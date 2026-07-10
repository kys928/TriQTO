"""Dataset-level orchestration for the Phase 11 persistent-homology audit."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from triqto.storage.topology_schema import TopologyGroupRecordV1

from .alignment import build_alignment_features
from .config import TopologyAuditConfig
from .constants import MANIFOLD_ORDER
from .distances import compute_manifold_distance_matrices
from .features import build_persistence_summary
from .identities import (
    topology_audit_id,
    topology_group_content_hash,
    topology_group_id,
    topology_operational_config_id,
    topology_schema_id,
)
from .models import TopologyAuditResult, TopologyGroupResult
from .persistent_homology import (
    compute_persistence_diagrams,
    make_filtration_grid,
)
from .point_clouds import build_point_cloud_group
from .source import load_topology_sources, verify_topology_source_snapshots
from .topology_groups import build_topology_group_specs
from .validators import (
    validate_topology_dataset_joins,
    validate_topology_group_result,
)


def _unicode_array(values: list[str]) -> np.ndarray:
    width = max([1, *[len(value) for value in values]])
    return np.asarray(values, dtype=f"<U{width}")


def _combined_topology_features(
    persistence: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    for manifold in MANIFOLD_ORDER:
        summary = persistence.get(manifold)
        if summary is None:
            continue
        for name, value in zip(
            summary.feature_names.tolist(),
            summary.feature_values.tolist(),
            strict=True,
        ):
            names.append(f"{manifold}_{name}")
            values.append(float(value))
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Combined topology features contain non-finite values")
    return _unicode_array(names), array


def build_topology_audit_result(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    action_source_root: str | Path,
    config: TopologyAuditConfig | None = None,
) -> TopologyAuditResult:
    """Build deterministic H0/H1 and optional H2 audits over aligned point clouds."""
    topology_config = config or TopologyAuditConfig()
    if not isinstance(topology_config, TopologyAuditConfig):
        raise TypeError("config must be TopologyAuditConfig or None")
    sources = load_topology_sources(
        phase7_source_root,
        graph_source_root,
        action_source_root,
    )
    phase7 = sources.phase7
    graph = sources.graph
    action = sources.action
    audit_id = topology_audit_id(
        phase7.source_scientific_generation_id,
        graph.completion_marker["graph_conversion_id"],
        action.completion_marker["action_engine_id"],
        topology_config,
    )
    specs, skipped_groups = build_topology_group_specs(sources, topology_config)
    if not specs:
        raise ValueError(
            "No topology groups satisfy min_points; lower min_points or generate a "
            "larger candidate/sample universe"
        )

    filtration_grid = make_filtration_grid(topology_config)
    groups: list[TopologyGroupResult] = []
    for spec in specs:
        point_cloud = build_point_cloud_group(spec, sources, topology_config)
        matrices, distance_metadata = compute_manifold_distance_matrices(
            parameter_coordinates=point_cloud.parameter_coordinates,
            parameter_coordinate_mask=point_cloud.parameter_coordinate_mask,
            born_coordinates=point_cloud.born_coordinates,
            statevectors=point_cloud.statevectors,
            config=topology_config,
        )
        hilbert_available = point_cloud.statevectors is not None
        available_manifolds = ["parameter"]
        if hilbert_available:
            available_manifolds.append("hilbert")
        available_manifolds.append("born")
        persistence: dict[str, Any] = {}
        ph_metadata: dict[str, Any] = {}
        for manifold in available_manifolds:
            diagrams, engine_metadata = compute_persistence_diagrams(
                matrices[manifold],
                topology_config,
            )
            summary = build_persistence_summary(
                manifold=manifold,
                diagrams=diagrams,
                filtration_grid=filtration_grid,
                point_count=int(point_cloud.point_ids.size),
                config=topology_config,
                metadata={
                    "distance_scale": distance_metadata["normalization_scales"][
                        manifold
                    ],
                    "distance_normalized": topology_config.normalize_distance_matrices,
                    "engine": engine_metadata,
                },
            )
            persistence[manifold] = summary
            ph_metadata[manifold] = engine_metadata

        topology_feature_names, topology_feature_values = (
            _combined_topology_features(persistence)
        )
        alignment_feature_names, alignment_feature_values, alignment_metadata = (
            build_alignment_features(persistence, topology_config)
        )
        point_ids_tuple = tuple(str(value) for value in point_cloud.point_ids.tolist())
        group_id = topology_group_id(
            audit_id,
            spec.group_kind,
            spec.group_key,
            point_ids_tuple,
        )
        manifold_mask = np.asarray(
            [True, hilbert_available, True],
            dtype=np.bool_,
        )
        group = TopologyGroupResult(
            topology_group_id=group_id,
            topology_audit_id=audit_id,
            group_kind=spec.group_kind,
            group_key=spec.group_key,
            point_ids=point_cloud.point_ids.copy(),
            parameter_coordinate_names=point_cloud.parameter_coordinate_names.copy(),
            parameter_coordinates=point_cloud.parameter_coordinates.copy(),
            parameter_coordinate_mask=point_cloud.parameter_coordinate_mask.copy(),
            born_outcome_bitstrings=point_cloud.born_outcome_bitstrings.copy(),
            born_coordinates=point_cloud.born_coordinates.copy(),
            parameter_distance_matrix=matrices["parameter"].copy(),
            hilbert_distance_matrix=matrices["hilbert"].copy(),
            born_distance_matrix=matrices["born"].copy(),
            filtration_grid=filtration_grid.copy(),
            manifold_available_mask=manifold_mask,
            persistence=persistence,
            topology_feature_names=topology_feature_names,
            topology_feature_values=topology_feature_values,
            alignment_feature_names=alignment_feature_names,
            alignment_feature_values=alignment_feature_values,
            metadata={
                **point_cloud.metadata,
                "distance_metadata": distance_metadata,
                "persistent_homology_metadata": ph_metadata,
                "alignment_metadata": alignment_metadata,
                "available_manifolds": available_manifolds,
                "hilbert_available": hilbert_available,
                "latent_available": False,
                "density_matrix_available": False,
                "topology_loss_weight": 0.0,
                "topology_mode": "audit_and_feature_only",
                "raw_statevectors_persisted": False,
                "topology_predictions_present": False,
                "model_present": False,
            },
        )
        group.content_hash = topology_group_content_hash(group)
        validate_topology_group_result(group, topology_config, require_hash=True)
        groups.append(group)

    groups.sort(key=lambda item: item.topology_group_id)
    records: list[TopologyGroupRecordV1] = []
    for group in groups:
        manifolds = [
            name
            for index, name in enumerate(MANIFOLD_ORDER)
            if bool(group.manifold_available_mask[index])
        ]
        record = TopologyGroupRecordV1(
            topology_group_id=group.topology_group_id,
            topology_audit_id=group.topology_audit_id,
            group_kind=group.group_kind,
            group_key=group.group_key,
            point_count=int(group.point_ids.size),
            homology_dimensions=list(topology_config.homology_dimensions),
            manifolds=manifolds,
            artifact_ref=f"artifacts/groups/{group.topology_group_id}.npz",
            content_hash=group.content_hash,
            hilbert_available=bool(group.manifold_available_mask[1]),
            latent_available=False,
            topology_feature_dim=int(group.topology_feature_values.size),
            alignment_feature_dim=int(group.alignment_feature_values.size),
            metadata={
                "phase": 11,
                "topology_mode": "audit_and_feature_only",
                "topology_loss_weight": 0.0,
                "raw_statevectors_persisted": False,
            },
        )
        record.validate()
        records.append(record)
    validate_topology_dataset_joins(
        records,
        groups_by_id={group.topology_group_id: group for group in groups},
        config=topology_config,
    )
    verify_topology_source_snapshots(sources)

    kind_counts = Counter(group.group_kind for group in groups)
    manifold_counts = Counter()
    groups_with_h1 = 0
    alignment_scores: list[float] = []
    unique_points: set[str] = set()
    for group in groups:
        unique_points.update(str(value) for value in group.point_ids.tolist())
        for index, manifold in enumerate(MANIFOLD_ORDER):
            if bool(group.manifold_available_mask[index]):
                manifold_counts[manifold] += 1
        if any(
            group.persistence[manifold].diagrams[1].shape[0] > 0
            for manifold in group.persistence
        ):
            groups_with_h1 += 1
        feature_index = {
            name: index
            for index, name in enumerate(group.alignment_feature_names.tolist())
        }
        preservation_index = feature_index.get("topology_preservation_score")
        if preservation_index is not None:
            alignment_scores.append(
                float(group.alignment_feature_values[preservation_index])
            )

    summary = {
        "source_scientific_generation_id": phase7.source_scientific_generation_id,
        "graph_conversion_id": graph.completion_marker["graph_conversion_id"],
        "action_engine_id": action.completion_marker["action_engine_id"],
        "topology_audit_id": audit_id,
        "operational_config_id": topology_operational_config_id(topology_config),
        "topology_schema_id": topology_schema_id(),
        "group_count": len(groups),
        "group_kind_counts": dict(sorted(kind_counts.items())),
        "skipped_group_counts": skipped_groups,
        "total_group_point_count": sum(int(group.point_ids.size) for group in groups),
        "unique_point_id_count": len(unique_points),
        "manifold_group_counts": dict(sorted(manifold_counts.items())),
        "hilbert_group_count": manifold_counts.get("hilbert", 0),
        "groups_with_nonempty_h1_diagram": groups_with_h1,
        "mean_topology_preservation_score": (
            fmean(alignment_scores) if alignment_scores else 0.0
        ),
        "homology_dimensions": list(topology_config.homology_dimensions),
        "h2_active": 2 in topology_config.homology_dimensions,
        "topology_loss_weight": 0.0,
        "topology_mode": "audit_and_feature_only",
        "latent_topology_available": False,
        "density_matrix_topology_available": False,
        "raw_statevectors_persisted": False,
        "source_immutability_verified": True,
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": action.snapshot.aggregate_sha256,
        "learned_model_present": False,
        "topology_utility_claimed": False,
        "quantum_advantage_claimed": False,
    }
    return TopologyAuditResult(
        phase7_source_root=phase7.source_root,
        graph_source_root=graph.root,
        action_source_root=action.root,
        config=topology_config,
        source_scientific_generation_id=phase7.source_scientific_generation_id,
        graph_conversion_id=graph.completion_marker["graph_conversion_id"],
        action_engine_id=action.completion_marker["action_engine_id"],
        topology_audit_id=audit_id,
        operational_config_id=topology_operational_config_id(topology_config),
        topology_schema_id=topology_schema_id(),
        groups=groups,
        group_records=records,
        phase7_snapshot=phase7.source_snapshot,
        graph_snapshot=graph.snapshot,
        action_snapshot=action.snapshot,
        summary=summary,
    )


__all__ = ["build_topology_audit_result"]
