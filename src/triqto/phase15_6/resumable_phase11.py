"""Restartable Phase 11 construction with one validated checkpoint per topology group."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import numpy as np

from triqto.graph.utils import strict_json_load, write_strict_json
from triqto.storage.topology_schema import TopologyGroupRecordV1
from triqto.topology.alignment import build_alignment_features
from triqto.topology.config import TopologyAuditConfig
from triqto.topology.constants import MANIFOLD_ORDER
from triqto.topology.distances import compute_manifold_distance_matrices
from triqto.topology.features import build_persistence_summary
from triqto.topology.identities import (
    topology_audit_id,
    topology_group_content_hash,
    topology_group_id,
    topology_operational_config_id,
    topology_schema_id,
)
from triqto.topology.models import TopologyAuditResult, TopologyGroupResult
from triqto.topology.persistent_homology import compute_persistence_diagrams, make_filtration_grid
from triqto.topology.pipeline import _combined_topology_features
from triqto.topology.point_clouds import build_point_cloud_group
from triqto.topology.source import load_topology_sources, verify_topology_source_snapshots
from triqto.topology.topology_groups import build_topology_group_specs
from triqto.topology.validators import validate_topology_dataset_joins, validate_topology_group_result
from triqto.topology.artifacts import load_topology_group_artifact, save_topology_group_artifact

ProgressCallback = Callable[[dict[str, Any]], None]


def _record_for_group(group: TopologyGroupResult, config: TopologyAuditConfig) -> TopologyGroupRecordV1:
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
        homology_dimensions=list(config.homology_dimensions),
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
    return record


def _build_one_group(spec: Any, sources: Any, config: TopologyAuditConfig, audit_id: str, filtration_grid: np.ndarray) -> TopologyGroupResult:
    point_cloud = build_point_cloud_group(spec, sources, config)
    matrices, distance_metadata = compute_manifold_distance_matrices(
        parameter_coordinates=point_cloud.parameter_coordinates,
        parameter_coordinate_mask=point_cloud.parameter_coordinate_mask,
        born_coordinates=point_cloud.born_coordinates,
        statevectors=point_cloud.statevectors,
        config=config,
    )
    hilbert_available = point_cloud.statevectors is not None
    available_manifolds = ["parameter"]
    if hilbert_available:
        available_manifolds.append("hilbert")
    available_manifolds.append("born")
    persistence: dict[str, Any] = {}
    ph_metadata: dict[str, Any] = {}
    for manifold in available_manifolds:
        diagrams, engine_metadata = compute_persistence_diagrams(matrices[manifold], config)
        summary = build_persistence_summary(
            manifold=manifold,
            diagrams=diagrams,
            filtration_grid=filtration_grid,
            point_count=int(point_cloud.point_ids.size),
            config=config,
            metadata={
                "distance_scale": distance_metadata["normalization_scales"][manifold],
                "distance_normalized": config.normalize_distance_matrices,
                "engine": engine_metadata,
            },
        )
        persistence[manifold] = summary
        ph_metadata[manifold] = engine_metadata
    topology_feature_names, topology_feature_values = _combined_topology_features(persistence)
    alignment_feature_names, alignment_feature_values, alignment_metadata = build_alignment_features(persistence, config)
    point_ids_tuple = tuple(str(value) for value in point_cloud.point_ids.tolist())
    group_id = topology_group_id(audit_id, spec.group_kind, spec.group_key, point_ids_tuple)
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
        manifold_available_mask=np.asarray([True, hilbert_available, True], dtype=np.bool_),
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
    validate_topology_group_result(group, config, require_hash=True)
    return group


def build_topology_audit_result_resumable(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    action_source_root: str | Path,
    checkpoint_root: str | Path,
    config: TopologyAuditConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> TopologyAuditResult:
    """Build Phase 11 and reuse every fully validated completed group checkpoint."""
    sources = load_topology_sources(phase7_source_root, graph_source_root, action_source_root)
    audit_id = topology_audit_id(
        sources.phase7.source_scientific_generation_id,
        sources.graph.completion_marker["graph_conversion_id"],
        sources.action.completion_marker["action_engine_id"],
        config,
    )
    specs, skipped_groups = build_topology_group_specs(sources, config)
    if not specs:
        raise ValueError("No topology groups satisfy min_points")
    root = Path(checkpoint_root)
    groups_root = root / "groups"
    markers_root = root / "markers"
    groups_root.mkdir(parents=True, exist_ok=True)
    markers_root.mkdir(parents=True, exist_ok=True)
    filtration_grid = make_filtration_grid(config)
    groups: list[TopologyGroupResult] = []
    resumed = 0
    for index, spec in enumerate(specs, start=1):
        expected_id = topology_group_id(
            audit_id,
            spec.group_kind,
            spec.group_key,
            tuple(sorted(spec.point_ids)),
        )
        artifact = groups_root / f"{expected_id}.npz"
        marker = markers_root / f"{expected_id}.json"
        group: TopologyGroupResult
        if artifact.is_file() and marker.is_file():
            payload = strict_json_load(marker)
            if not isinstance(payload, dict) or payload.get("complete") is not True:
                raise ValueError(f"Invalid Phase 11 checkpoint marker {marker}")
            if payload.get("topology_audit_id") != audit_id or payload.get("topology_group_id") != expected_id:
                raise ValueError(f"Stale Phase 11 checkpoint {marker}")
            group = load_topology_group_artifact(artifact, config, payload.get("content_hash"))
            resumed += 1
        else:
            artifact.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            group = _build_one_group(spec, sources, config, audit_id, filtration_grid)
            save_topology_group_artifact(group, config, artifact)
            write_strict_json(
                marker,
                {
                    "complete": True,
                    "topology_audit_id": audit_id,
                    "topology_group_id": group.topology_group_id,
                    "content_hash": group.content_hash,
                    "group_kind": group.group_kind,
                    "group_key": group.group_key,
                    "point_count": int(group.point_ids.size),
                },
            )
        groups.append(group)
        if progress_callback is not None:
            progress_callback({
                "completed_groups": index,
                "total_groups": len(specs),
                "resumed_groups": resumed,
                "group_kind": spec.group_kind,
                "group_key": spec.group_key,
            })
    groups.sort(key=lambda item: item.topology_group_id)
    records = [_record_for_group(group, config) for group in groups]
    validate_topology_dataset_joins(records, groups_by_id={g.topology_group_id: g for g in groups}, config=config)
    verify_topology_source_snapshots(sources)
    kind_counts = Counter(group.group_kind for group in groups)
    manifold_counts: Counter[str] = Counter()
    groups_with_h1 = 0
    alignment_scores: list[float] = []
    unique_points: set[str] = set()
    for group in groups:
        unique_points.update(str(value) for value in group.point_ids.tolist())
        for idx, manifold in enumerate(MANIFOLD_ORDER):
            if bool(group.manifold_available_mask[idx]):
                manifold_counts[manifold] += 1
        if any(group.persistence[m].diagrams[1].shape[0] > 0 for m in group.persistence):
            groups_with_h1 += 1
        feature_index = {name: idx for idx, name in enumerate(group.alignment_feature_names.tolist())}
        preservation = feature_index.get("topology_preservation_score")
        if preservation is not None:
            alignment_scores.append(float(group.alignment_feature_values[preservation]))
    summary = {
        "source_scientific_generation_id": sources.phase7.source_scientific_generation_id,
        "graph_conversion_id": sources.graph.completion_marker["graph_conversion_id"],
        "action_engine_id": sources.action.completion_marker["action_engine_id"],
        "topology_audit_id": audit_id,
        "operational_config_id": topology_operational_config_id(config),
        "topology_schema_id": topology_schema_id(),
        "group_count": len(groups),
        "group_kind_counts": dict(sorted(kind_counts.items())),
        "skipped_group_counts": skipped_groups,
        "total_group_point_count": sum(int(g.point_ids.size) for g in groups),
        "unique_point_id_count": len(unique_points),
        "manifold_group_counts": dict(sorted(manifold_counts.items())),
        "hilbert_group_count": manifold_counts.get("hilbert", 0),
        "groups_with_nonempty_h1_diagram": groups_with_h1,
        "mean_topology_preservation_score": fmean(alignment_scores) if alignment_scores else 0.0,
        "homology_dimensions": list(config.homology_dimensions),
        "h2_active": 2 in config.homology_dimensions,
        "topology_loss_weight": 0.0,
        "topology_mode": "audit_and_feature_only",
        "latent_topology_available": False,
        "density_matrix_topology_available": False,
        "raw_statevectors_persisted": False,
        "source_immutability_verified": True,
        "phase7_snapshot_hash": sources.phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": sources.graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": sources.action.snapshot.aggregate_sha256,
        "learned_model_present": False,
        "topology_utility_claimed": False,
        "quantum_advantage_claimed": False,
        "checkpoint_resume": {"enabled": True, "resumed_group_count": resumed},
    }
    return TopologyAuditResult(
        phase7_source_root=sources.phase7.source_root,
        graph_source_root=sources.graph.root,
        action_source_root=sources.action.root,
        config=config,
        source_scientific_generation_id=sources.phase7.source_scientific_generation_id,
        graph_conversion_id=sources.graph.completion_marker["graph_conversion_id"],
        action_engine_id=sources.action.completion_marker["action_engine_id"],
        topology_audit_id=audit_id,
        operational_config_id=topology_operational_config_id(config),
        topology_schema_id=topology_schema_id(),
        groups=groups,
        group_records=records,
        phase7_snapshot=sources.phase7.source_snapshot,
        graph_snapshot=sources.graph.snapshot,
        action_snapshot=sources.action.snapshot,
        summary=summary,
    )


__all__ = ["build_topology_audit_result_resumable"]
