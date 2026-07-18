"""Stage-checkpointed Phase 11 topology construction.

Each topology group is a deterministic work unit. Point-cloud construction, distance
matrices, each manifold's persistent homology, and the final validated group artifact
are committed independently. A restart resumes at the first missing validated stage.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import time
from typing import Any, Callable, TypeVar

import numpy as np

from triqto.storage.topology_schema import TopologyGroupRecordV1
from triqto.topology.alignment import build_alignment_features
from triqto.topology.artifacts import (
    load_topology_group_artifact,
    save_topology_group_artifact,
)
from triqto.topology.config import TopologyAuditConfig
from triqto.topology.constants import MANIFOLD_ORDER
from triqto.topology.distances import (
    compute_manifold_distance_matrices,
    validate_distance_matrix,
)
from triqto.topology.features import build_persistence_summary
from triqto.topology.identities import (
    topology_audit_id,
    topology_group_content_hash,
    topology_group_id,
    topology_operational_config_id,
    topology_schema_id,
)
from triqto.topology.models import (
    PersistenceSummary,
    TopologyAuditResult,
    TopologyGroupResult,
    TopologyPointCloudGroup,
)
from triqto.topology.persistent_homology import (
    compute_persistence_diagrams,
    make_filtration_grid,
)
from triqto.topology.pipeline import _combined_topology_features
from triqto.topology.point_clouds import build_point_cloud_group
from triqto.topology.source import (
    load_topology_sources,
    verify_topology_source_snapshots,
)
from triqto.topology.topology_groups import build_topology_group_specs
from triqto.topology.validators import (
    validate_topology_dataset_joins,
    validate_topology_group_result,
)

from .resumable import (
    atomic_write_npz,
    clear_checkpoint_failure,
    commit_checkpoint_artifact,
    decode_json_bytes,
    json_bytes,
    load_checkpoint_artifact,
    prepare_checkpoint_root,
    record_checkpoint_failure,
)

ProgressCallback = Callable[[dict[str, Any]], None]
T = TypeVar("T")
_PHASE = "phase11"
_POINT_METADATA = "__point_metadata_json__"
_DISTANCE_METADATA = "__distance_metadata_json__"
_PERSISTENCE_METADATA = "__persistence_metadata_json__"


def _record_for_group(
    group: TopologyGroupResult,
    config: TopologyAuditConfig,
) -> TopologyGroupRecordV1:
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
            "ephemeral_statevectors_checkpointed": bool(
                group.manifold_available_mask[1]
            ),
        },
    )
    record.validate()
    return record


def _point_cloud_arrays(point_cloud: TopologyPointCloudGroup) -> dict[str, np.ndarray]:
    statevectors = point_cloud.statevectors
    return {
        "point_ids": point_cloud.point_ids,
        "parameter_coordinate_names": point_cloud.parameter_coordinate_names,
        "parameter_coordinates": point_cloud.parameter_coordinates,
        "parameter_coordinate_mask": point_cloud.parameter_coordinate_mask,
        "born_outcome_bitstrings": point_cloud.born_outcome_bitstrings,
        "born_coordinates": point_cloud.born_coordinates,
        "statevectors": (
            statevectors
            if statevectors is not None
            else np.zeros((0, 0), dtype=np.complex128)
        ),
        "statevectors_available": np.asarray(
            [statevectors is not None], dtype=np.bool_
        ),
        _POINT_METADATA: json_bytes(
            {
                "group_kind": point_cloud.group_kind,
                "group_key": point_cloud.group_key,
                "metadata": point_cloud.metadata,
            }
        ),
    }


def _load_point_cloud(path: Path) -> TopologyPointCloudGroup:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "point_ids",
            "parameter_coordinate_names",
            "parameter_coordinates",
            "parameter_coordinate_mask",
            "born_outcome_bitstrings",
            "born_coordinates",
            "statevectors",
            "statevectors_available",
            _POINT_METADATA,
        }
        if set(payload.files) != required:
            raise ValueError("Phase 11 point-cloud checkpoint array-name mismatch")
        metadata = decode_json_bytes(payload[_POINT_METADATA], _POINT_METADATA)
        available = payload["statevectors_available"]
        if available.dtype != np.bool_ or available.shape != (1,):
            raise TypeError("statevectors_available must be shape-(1,) bool")
        statevectors = payload["statevectors"].copy() if bool(available[0]) else None
        point_cloud = TopologyPointCloudGroup(
            group_kind=metadata["group_kind"],
            group_key=metadata["group_key"],
            point_ids=payload["point_ids"].copy(),
            parameter_coordinate_names=payload[
                "parameter_coordinate_names"
            ].copy(),
            parameter_coordinates=payload["parameter_coordinates"].copy(),
            parameter_coordinate_mask=payload["parameter_coordinate_mask"].copy(),
            born_outcome_bitstrings=payload["born_outcome_bitstrings"].copy(),
            born_coordinates=payload["born_coordinates"].copy(),
            statevectors=statevectors,
            metadata=dict(metadata["metadata"]),
        )
    count = int(point_cloud.point_ids.size)
    if count < 1:
        raise ValueError("Point-cloud checkpoint must contain at least one point")
    if point_cloud.parameter_coordinates.shape[0] != count:
        raise ValueError("Point-cloud parameter row count mismatch")
    if point_cloud.born_coordinates.shape[0] != count:
        raise ValueError("Point-cloud Born row count mismatch")
    if point_cloud.statevectors is not None:
        if point_cloud.statevectors.dtype != np.complex128:
            raise TypeError("Checkpointed statevectors must be complex128")
        if point_cloud.statevectors.shape[0] != count:
            raise ValueError("Point-cloud statevector row count mismatch")
    return point_cloud


def _distance_arrays(
    matrices: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {
        "parameter": matrices["parameter"],
        "hilbert": matrices["hilbert"],
        "born": matrices["born"],
        _DISTANCE_METADATA: json_bytes(metadata),
    }


def _load_distances(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "parameter",
            "hilbert",
            "born",
            _DISTANCE_METADATA,
        }:
            raise ValueError("Phase 11 distance checkpoint array-name mismatch")
        matrices = {
            name: payload[name].copy() for name in ("parameter", "hilbert", "born")
        }
        metadata = decode_json_bytes(
            payload[_DISTANCE_METADATA], _DISTANCE_METADATA
        )
    for name, matrix in matrices.items():
        validate_distance_matrix(matrix, f"checkpoint_{name}_distance_matrix")
    if len({matrix.shape for matrix in matrices.values()}) != 1:
        raise ValueError("Checkpointed distance matrices must share one shape")
    return matrices, metadata


def _persistence_arrays(summary: PersistenceSummary) -> dict[str, np.ndarray]:
    arrays = {
        "feature_names": summary.feature_names,
        "feature_values": summary.feature_values,
        _PERSISTENCE_METADATA: json_bytes(
            {"manifold": summary.manifold, "metadata": summary.metadata}
        ),
    }
    for dimension, diagram in summary.diagrams.items():
        arrays[f"h{dimension}_diagram"] = diagram
    for dimension, curve in summary.betti_curves.items():
        arrays[f"h{dimension}_betti_curve"] = curve
    return arrays


def _load_persistence(
    path: Path,
    manifold: str,
    config: TopologyAuditConfig,
) -> PersistenceSummary:
    with np.load(path, allow_pickle=False) as payload:
        expected = {"feature_names", "feature_values", _PERSISTENCE_METADATA}
        expected.update(
            f"h{dimension}_diagram" for dimension in config.homology_dimensions
        )
        expected.update(
            f"h{dimension}_betti_curve" for dimension in config.homology_dimensions
        )
        if set(payload.files) != expected:
            raise ValueError(
                f"Phase 11 {manifold} persistence checkpoint array-name mismatch"
            )
        metadata = decode_json_bytes(
            payload[_PERSISTENCE_METADATA], _PERSISTENCE_METADATA
        )
        if metadata.get("manifold") != manifold:
            raise ValueError("Persistence checkpoint manifold mismatch")
        summary = PersistenceSummary(
            manifold=manifold,
            diagrams={
                dimension: payload[f"h{dimension}_diagram"].copy()
                for dimension in config.homology_dimensions
            },
            betti_curves={
                dimension: payload[f"h{dimension}_betti_curve"].copy()
                for dimension in config.homology_dimensions
            },
            feature_names=payload["feature_names"].copy(),
            feature_values=payload["feature_values"].copy(),
            metadata=dict(metadata["metadata"]),
        )
    for dimension, diagram in summary.diagrams.items():
        if diagram.dtype != np.float64 or diagram.ndim != 2 or diagram.shape[1] != 2:
            raise TypeError(f"Checkpointed H{dimension} diagram must be Nx2 float64")
    return summary


def _paths(root: Path, unit_id: str, stage: str) -> tuple[Path, Path]:
    unit = root / "units" / unit_id
    return unit / f"{stage}.npz", unit / f"{stage}.json"


def _emit(
    callback: ProgressCallback | None,
    *,
    index: int,
    total: int,
    spec: Any,
    stage: str,
    status: str,
    resumed_stages: int,
    elapsed_seconds: float | None = None,
    workers: int = 1,
) -> None:
    if callback is None:
        return
    payload: dict[str, Any] = {
        "event": "work_unit",
        "completed_groups": index - (0 if status == "group_complete" else 1),
        "current_group_index": index,
        "total_groups": total,
        "group_kind": spec.group_kind,
        "group_key": spec.group_key,
        "point_count": len(spec.point_ids),
        "stage": stage,
        "stage_status": status,
        "resumed_stages": resumed_stages,
        "workers": workers,
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    callback(payload)


def _checkpointed_stage(
    *,
    root: Path,
    unit_id: str,
    stage: str,
    identity: dict[str, Any],
    resume_mode: str,
    loader: Callable[[Path], T],
    writer: Callable[[Path, T], None],
    compute: Callable[[], T],
    progress_callback: ProgressCallback | None,
    index: int,
    total: int,
    spec: Any,
    resumed_stages: list[int],
    workers: int,
) -> T:
    artifact, marker = _paths(root, unit_id, stage)
    loaded = load_checkpoint_artifact(
        root=root,
        phase=_PHASE,
        unit_id=unit_id,
        stage=stage,
        artifact_path=artifact,
        marker_path=marker,
        identity=identity,
        resume_mode=resume_mode,
        loader=lambda path, _payload: loader(path),
    )
    if loaded is not None:
        resumed_stages[0] += 1
        _emit(
            progress_callback,
            index=index,
            total=total,
            spec=spec,
            stage=stage,
            status="resumed",
            resumed_stages=resumed_stages[0],
            workers=workers,
        )
        return loaded
    started = time.monotonic()
    _emit(
        progress_callback,
        index=index,
        total=total,
        spec=spec,
        stage=stage,
        status="started",
        resumed_stages=resumed_stages[0],
        workers=workers,
    )
    value = compute()
    committed = commit_checkpoint_artifact(
        phase=_PHASE,
        unit_id=unit_id,
        stage=stage,
        artifact_path=artifact,
        marker_path=marker,
        identity=identity,
        writer=lambda path: writer(path, value),
        validator=loader,
        marker_metadata={
            "group_kind": spec.group_kind,
            "group_key": spec.group_key,
            "point_count": len(spec.point_ids),
        },
    )
    _emit(
        progress_callback,
        index=index,
        total=total,
        spec=spec,
        stage=stage,
        status="completed",
        resumed_stages=resumed_stages[0],
        elapsed_seconds=time.monotonic() - started,
        workers=workers,
    )
    return committed


def _process_group(
    *,
    spec: Any,
    index: int,
    total: int,
    sources: Any,
    config: TopologyAuditConfig,
    audit_id: str,
    root: Path,
    resume_mode: str,
    progress_callback: ProgressCallback | None,
    workers: int,
) -> str:
    expected_id = topology_group_id(
        audit_id,
        spec.group_kind,
        spec.group_key,
        tuple(sorted(spec.point_ids)),
    )
    identity = {
        "checkpoint_schema": "triqto.phase15_6.phase11_group.v2",
        "topology_audit_id": audit_id,
        "topology_group_id": expected_id,
        "group_kind": spec.group_kind,
        "group_key": spec.group_key,
        "point_ids": list(sorted(spec.point_ids)),
        "operational_config_id": topology_operational_config_id(config),
        "phase7_snapshot_hash": sources.phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": sources.graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": sources.action.snapshot.aggregate_sha256,
    }
    resumed_stages = [0]
    current_stage = "point_cloud"
    try:
        point_cloud = _checkpointed_stage(
            root=root,
            unit_id=expected_id,
            stage=current_stage,
            identity=identity,
            resume_mode=resume_mode,
            loader=_load_point_cloud,
            writer=lambda path, value: atomic_write_npz(
                path, _point_cloud_arrays(value)
            ),
            compute=lambda: build_point_cloud_group(spec, sources, config),
            progress_callback=progress_callback,
            index=index,
            total=total,
            spec=spec,
            resumed_stages=resumed_stages,
            workers=workers,
        )
        current_stage = "distance_matrices"
        matrices, distance_metadata = _checkpointed_stage(
            root=root,
            unit_id=expected_id,
            stage=current_stage,
            identity=identity,
            resume_mode=resume_mode,
            loader=_load_distances,
            writer=lambda path, value: atomic_write_npz(
                path, _distance_arrays(value[0], value[1])
            ),
            compute=lambda: compute_manifold_distance_matrices(
                parameter_coordinates=point_cloud.parameter_coordinates,
                parameter_coordinate_mask=point_cloud.parameter_coordinate_mask,
                born_coordinates=point_cloud.born_coordinates,
                statevectors=point_cloud.statevectors,
                config=config,
            ),
            progress_callback=progress_callback,
            index=index,
            total=total,
            spec=spec,
            resumed_stages=resumed_stages,
            workers=workers,
        )
        hilbert_available = point_cloud.statevectors is not None
        available_manifolds = ["parameter"]
        if hilbert_available:
            available_manifolds.append("hilbert")
        available_manifolds.append("born")
        filtration_grid = make_filtration_grid(config)
        persistence: dict[str, PersistenceSummary] = {}
        for manifold in available_manifolds:
            current_stage = f"persistence_{manifold}"

            def compute_summary(active: str = manifold) -> PersistenceSummary:
                diagrams, engine_metadata = compute_persistence_diagrams(
                    matrices[active], config
                )
                return build_persistence_summary(
                    manifold=active,
                    diagrams=diagrams,
                    filtration_grid=filtration_grid,
                    point_count=int(point_cloud.point_ids.size),
                    config=config,
                    metadata={
                        "distance_scale": distance_metadata[
                            "normalization_scales"
                        ][active],
                        "distance_normalized": config.normalize_distance_matrices,
                        "engine": engine_metadata,
                    },
                )

            persistence[manifold] = _checkpointed_stage(
                root=root,
                unit_id=expected_id,
                stage=current_stage,
                identity=identity,
                resume_mode=resume_mode,
                loader=lambda path, active=manifold: _load_persistence(
                    path, active, config
                ),
                writer=lambda path, value: atomic_write_npz(
                    path, _persistence_arrays(value)
                ),
                compute=compute_summary,
                progress_callback=progress_callback,
                index=index,
                total=total,
                spec=spec,
                resumed_stages=resumed_stages,
                workers=workers,
            )
        current_stage = "final_group"

        def build_final_group() -> TopologyGroupResult:
            topology_feature_names, topology_feature_values = (
                _combined_topology_features(persistence)
            )
            alignment_feature_names, alignment_feature_values, alignment_metadata = (
                build_alignment_features(persistence, config)
            )
            ph_metadata = {
                manifold: summary.metadata["engine"]
                for manifold, summary in persistence.items()
            }
            group = TopologyGroupResult(
                topology_group_id=expected_id,
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
                manifold_available_mask=np.asarray(
                    [True, hilbert_available, True], dtype=np.bool_
                ),
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
                    "ephemeral_statevectors_checkpointed": hilbert_available,
                    "topology_predictions_present": False,
                    "model_present": False,
                },
            )
            group.content_hash = topology_group_content_hash(group)
            validate_topology_group_result(group, config, require_hash=True)
            return group

        _checkpointed_stage(
            root=root,
            unit_id=expected_id,
            stage=current_stage,
            identity=identity,
            resume_mode=resume_mode,
            loader=lambda path: load_topology_group_artifact(path, config),
            writer=lambda path, value: save_topology_group_artifact(
                value, config, path
            ),
            compute=build_final_group,
            progress_callback=progress_callback,
            index=index,
            total=total,
            spec=spec,
            resumed_stages=resumed_stages,
            workers=workers,
        )
        clear_checkpoint_failure(root, expected_id)
        _emit(
            progress_callback,
            index=index,
            total=total,
            spec=spec,
            stage=current_stage,
            status="group_complete",
            resumed_stages=resumed_stages[0],
            workers=workers,
        )
        return expected_id
    except Exception as exc:
        record_checkpoint_failure(
            root=root,
            phase=_PHASE,
            unit_id=expected_id,
            stage=current_stage,
            error=exc,
            context={
                "group_kind": spec.group_kind,
                "group_key": spec.group_key,
                "point_count": len(spec.point_ids),
                "topology_audit_id": audit_id,
            },
        )
        raise


def _drain(futures: list[Future[str]], completed: list[str]) -> None:
    for future in futures:
        completed.append(future.result())
    futures.clear()


def _estimated_checkpoint_bytes(specs: list[Any], include_hilbert: bool) -> int:
    manifold_count = 3 if include_hilbert else 2
    total = 0
    for spec in specs:
        points = len(spec.point_ids)
        total += points * points * 8 * manifold_count
        total += points * points * 8 * 3
    return total


def build_topology_audit_result_resumable(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    action_source_root: str | Path,
    checkpoint_root: str | Path,
    config: TopologyAuditConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    resume_mode: str = "strict",
    workers: int = 1,
    exclusive_point_threshold: int = 512,
) -> TopologyAuditResult:
    """Build Phase 11 with stage-level checkpoint resume and bounded parallel groups."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Phase 11 workers must be a positive integer")
    if (
        isinstance(exclusive_point_threshold, bool)
        or not isinstance(exclusive_point_threshold, int)
        or exclusive_point_threshold < 1
    ):
        raise ValueError("exclusive_point_threshold must be a positive integer")
    sources = load_topology_sources(
        phase7_source_root, graph_source_root, action_source_root
    )
    audit_id = topology_audit_id(
        sources.phase7.source_scientific_generation_id,
        sources.graph.completion_marker["graph_conversion_id"],
        sources.action.completion_marker["action_engine_id"],
        config,
    )
    specs, skipped_groups = build_topology_group_specs(sources, config)
    if not specs:
        raise ValueError("No topology groups satisfy min_points")
    root = prepare_checkpoint_root(checkpoint_root, resume_mode)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "plan",
                "total_groups": len(specs),
                "workers": workers,
                "exclusive_point_threshold": exclusive_point_threshold,
                "estimated_checkpoint_bytes": _estimated_checkpoint_bytes(
                    specs, config.include_hilbert
                ),
            }
        )
    completed_ids: list[str] = []
    if workers == 1:
        for index, spec in enumerate(specs, start=1):
            completed_ids.append(
                _process_group(
                    spec=spec,
                    index=index,
                    total=len(specs),
                    sources=sources,
                    config=config,
                    audit_id=audit_id,
                    root=root,
                    resume_mode=resume_mode,
                    progress_callback=progress_callback,
                    workers=workers,
                )
            )
    else:
        futures: list[Future[str]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, spec in enumerate(specs, start=1):
                if len(spec.point_ids) >= exclusive_point_threshold:
                    _drain(futures, completed_ids)
                    completed_ids.append(
                        _process_group(
                            spec=spec,
                            index=index,
                            total=len(specs),
                            sources=sources,
                            config=config,
                            audit_id=audit_id,
                            root=root,
                            resume_mode=resume_mode,
                            progress_callback=progress_callback,
                            workers=workers,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            _process_group,
                            spec=spec,
                            index=index,
                            total=len(specs),
                            sources=sources,
                            config=config,
                            audit_id=audit_id,
                            root=root,
                            resume_mode=resume_mode,
                            progress_callback=progress_callback,
                            workers=workers,
                        )
                    )
            _drain(futures, completed_ids)
    groups = [
        load_topology_group_artifact(
            _paths(root, group_id, "final_group")[0], config
        )
        for group_id in sorted(completed_ids)
    ]
    records = [_record_for_group(group, config) for group in groups]
    validate_topology_dataset_joins(
        records,
        groups_by_id={group.topology_group_id: group for group in groups},
        config=config,
    )
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
        if any(
            group.persistence[manifold].diagrams[1].shape[0] > 0
            for manifold in group.persistence
        ):
            groups_with_h1 += 1
        feature_index = {
            name: idx
            for idx, name in enumerate(group.alignment_feature_names.tolist())
        }
        preservation = feature_index.get("topology_preservation_score")
        if preservation is not None:
            alignment_scores.append(
                float(group.alignment_feature_values[preservation])
            )
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
        "total_group_point_count": sum(int(group.point_ids.size) for group in groups),
        "unique_point_id_count": len(unique_points),
        "manifold_group_counts": dict(sorted(manifold_counts.items())),
        "hilbert_group_count": manifold_counts.get("hilbert", 0),
        "groups_with_nonempty_h1_diagram": groups_with_h1,
        "mean_topology_preservation_score": (
            fmean(alignment_scores) if alignment_scores else 0.0
        ),
        "homology_dimensions": list(config.homology_dimensions),
        "h2_active": 2 in config.homology_dimensions,
        "topology_loss_weight": 0.0,
        "topology_mode": "audit_and_feature_only",
        "latent_topology_available": False,
        "density_matrix_topology_available": False,
        "raw_statevectors_persisted": False,
        "ephemeral_statevectors_checkpointed": config.include_hilbert,
        "source_immutability_verified": True,
        "phase7_snapshot_hash": sources.phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": sources.graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": sources.action.snapshot.aggregate_sha256,
        "learned_model_present": False,
        "topology_utility_claimed": False,
        "quantum_advantage_claimed": False,
        "checkpoint_resume": {
            "enabled": True,
            "granularity": "point_cloud_distance_per_manifold_persistence_final_group",
            "validated_final_group_count": len(groups),
            "resume_mode": resume_mode,
            "workers": workers,
            "exclusive_point_threshold": exclusive_point_threshold,
        },
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
