"""Strict NPZ artifacts and immutable publication for Phase 11 topology audits."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np

from triqto.graph import snapshot_managed_files
from triqto.graph.utils import (
    json_copy,
    resolve_safe_file,
    strict_json_load,
    strict_json_loads,
    write_strict_json,
)
from triqto.storage.manifest import ManifestReader, ManifestWriter
from triqto.storage.topology_schema import TopologyGroupRecordV1

from .config import (
    TopologyAuditConfig,
    load_topology_config,
    save_topology_config,
)
from .constants import (
    BASE_ARRAY_NAMES,
    MANIFOLD_ORDER,
    TOPOLOGY_ARTIFACT_SCHEMA_VERSION,
    TOPOLOGY_METADATA_ARRAY_NAME,
)
from .identities import topology_group_content_hash
from .models import (
    PersistenceSummary,
    TopologyAuditResult,
    TopologyGroupResult,
    TopologyWriteResult,
)
from .validators import (
    validate_topology_dataset_joins,
    validate_topology_group_result,
)


def _json_bytes(payload: Mapping[str, Any]) -> np.ndarray:
    text = json.dumps(
        json_copy(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def _decode_json_bytes(array: np.ndarray, name: str) -> dict[str, Any]:
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional uint8 array")
    try:
        text = array.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    payload = strict_json_loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must decode to a JSON object")
    return dict(payload)


def _artifact_metadata(group: TopologyGroupResult) -> dict[str, Any]:
    persistence_metadata = {
        manifold: summary.metadata
        for manifold, summary in group.persistence.items()
    }
    available = [
        name
        for index, name in enumerate(MANIFOLD_ORDER)
        if bool(group.manifold_available_mask[index])
    ]
    dimensions = sorted(
        next(iter(group.persistence.values())).diagrams
        if group.persistence
        else []
    )
    return {
        "artifact_schema_version": TOPOLOGY_ARTIFACT_SCHEMA_VERSION,
        "topology_group_id": group.topology_group_id,
        "topology_audit_id": group.topology_audit_id,
        "group_kind": group.group_kind,
        "group_key": group.group_key,
        "available_manifolds": available,
        "homology_dimensions": dimensions,
        "persistence_metadata": persistence_metadata,
        "metadata": group.metadata,
        "content_hash": topology_group_content_hash(group),
    }


def _artifact_arrays(group: TopologyGroupResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "point_ids": group.point_ids,
        "parameter_coordinate_names": group.parameter_coordinate_names,
        "parameter_coordinates": group.parameter_coordinates,
        "parameter_coordinate_mask": group.parameter_coordinate_mask,
        "parameter_distance_matrix": group.parameter_distance_matrix,
        "hilbert_distance_matrix": group.hilbert_distance_matrix,
        "born_outcome_bitstrings": group.born_outcome_bitstrings,
        "born_coordinates": group.born_coordinates,
        "born_distance_matrix": group.born_distance_matrix,
        "filtration_grid": group.filtration_grid,
        "manifold_available_mask": group.manifold_available_mask,
        "topology_feature_names": group.topology_feature_names,
        "topology_feature_values": group.topology_feature_values,
        "alignment_feature_names": group.alignment_feature_names,
        "alignment_feature_values": group.alignment_feature_values,
    }
    for manifold, summary in group.persistence.items():
        arrays[f"{manifold}_feature_names"] = summary.feature_names
        arrays[f"{manifold}_feature_values"] = summary.feature_values
        for dimension, diagram in summary.diagrams.items():
            arrays[f"{manifold}_h{dimension}_diagram"] = diagram
        for dimension, curve in summary.betti_curves.items():
            arrays[f"{manifold}_h{dimension}_betti_curve"] = curve
    return arrays


def save_topology_group_artifact(
    group: TopologyGroupResult,
    config: TopologyAuditConfig,
    path: str | Path,
) -> Path:
    validate_topology_group_result(group, config, require_hash=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = _artifact_arrays(group)
    np.savez_compressed(
        target,
        **arrays,
        **{TOPOLOGY_METADATA_ARRAY_NAME: _json_bytes(_artifact_metadata(group))},
    )
    return target


def load_topology_group_artifact(
    path: str | Path,
    config: TopologyAuditConfig,
    expected_content_hash: str | None = None,
) -> TopologyGroupResult:
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        if TOPOLOGY_METADATA_ARRAY_NAME not in payload.files:
            raise ValueError("Topology artifact metadata array is missing")
        metadata = _decode_json_bytes(
            payload[TOPOLOGY_METADATA_ARRAY_NAME],
            TOPOLOGY_METADATA_ARRAY_NAME,
        )
        metadata_keys = {
            "artifact_schema_version",
            "topology_group_id",
            "topology_audit_id",
            "group_kind",
            "group_key",
            "available_manifolds",
            "homology_dimensions",
            "persistence_metadata",
            "metadata",
            "content_hash",
        }
        if set(metadata) != metadata_keys:
            raise ValueError("Topology artifact metadata-key mismatch")
        if metadata["artifact_schema_version"] != TOPOLOGY_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported topology artifact schema version")
        available = metadata["available_manifolds"]
        dimensions = metadata["homology_dimensions"]
        if not isinstance(available, list) or any(
            name not in MANIFOLD_ORDER for name in available
        ):
            raise ValueError("Topology artifact available_manifolds is invalid")
        if available != [name for name in MANIFOLD_ORDER if name in available]:
            raise ValueError("Topology artifact manifolds must follow fixed order")
        if not isinstance(dimensions, list) or dimensions != list(
            config.homology_dimensions
        ):
            raise ValueError("Topology artifact homology dimensions mismatch")
        expected_names = set(BASE_ARRAY_NAMES) | {TOPOLOGY_METADATA_ARRAY_NAME}
        for manifold in available:
            expected_names.update(
                {
                    f"{manifold}_feature_names",
                    f"{manifold}_feature_values",
                }
            )
            for dimension in dimensions:
                expected_names.add(f"{manifold}_h{dimension}_diagram")
                expected_names.add(f"{manifold}_h{dimension}_betti_curve")
        actual_names = set(payload.files)
        if actual_names != expected_names:
            raise ValueError(
                "Topology artifact array-name mismatch; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        arrays = {name: payload[name].copy() for name in expected_names if name != TOPOLOGY_METADATA_ARRAY_NAME}

    persistence_metadata = metadata["persistence_metadata"]
    if not isinstance(persistence_metadata, Mapping):
        raise TypeError("Topology artifact persistence_metadata must be a mapping")
    persistence: dict[str, PersistenceSummary] = {}
    for manifold in available:
        manifold_metadata = persistence_metadata.get(manifold)
        if not isinstance(manifold_metadata, Mapping):
            raise TypeError(
                f"Topology artifact persistence metadata for {manifold} must be a mapping"
            )
        persistence[manifold] = PersistenceSummary(
            manifold=manifold,
            diagrams={
                dimension: arrays[f"{manifold}_h{dimension}_diagram"]
                for dimension in dimensions
            },
            betti_curves={
                dimension: arrays[f"{manifold}_h{dimension}_betti_curve"]
                for dimension in dimensions
            },
            feature_names=arrays[f"{manifold}_feature_names"],
            feature_values=arrays[f"{manifold}_feature_values"],
            metadata=dict(manifold_metadata),
        )
    group_metadata = metadata["metadata"]
    if not isinstance(group_metadata, Mapping):
        raise TypeError("Topology artifact metadata.metadata must be a mapping")
    group = TopologyGroupResult(
        topology_group_id=metadata["topology_group_id"],
        topology_audit_id=metadata["topology_audit_id"],
        group_kind=metadata["group_kind"],
        group_key=metadata["group_key"],
        point_ids=arrays["point_ids"],
        parameter_coordinate_names=arrays["parameter_coordinate_names"],
        parameter_coordinates=arrays["parameter_coordinates"],
        parameter_coordinate_mask=arrays["parameter_coordinate_mask"],
        born_outcome_bitstrings=arrays["born_outcome_bitstrings"],
        born_coordinates=arrays["born_coordinates"],
        parameter_distance_matrix=arrays["parameter_distance_matrix"],
        hilbert_distance_matrix=arrays["hilbert_distance_matrix"],
        born_distance_matrix=arrays["born_distance_matrix"],
        filtration_grid=arrays["filtration_grid"],
        manifold_available_mask=arrays["manifold_available_mask"],
        persistence=persistence,
        topology_feature_names=arrays["topology_feature_names"],
        topology_feature_values=arrays["topology_feature_values"],
        alignment_feature_names=arrays["alignment_feature_names"],
        alignment_feature_values=arrays["alignment_feature_values"],
        metadata=dict(group_metadata),
        content_hash=metadata["content_hash"],
    )
    validate_topology_group_result(group, config, require_hash=True)
    if expected_content_hash is not None and group.content_hash != expected_content_hash:
        raise ValueError("Topology group content_hash does not match manifest")
    return group


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _verify_result_sources(result: TopologyAuditResult) -> None:
    checks = (
        ("Phase 7", result.phase7_source_root, result.phase7_snapshot),
        ("Phase 8", result.graph_source_root, result.graph_snapshot),
        ("Phase 9", result.action_source_root, result.action_snapshot),
    )
    for name, root, expected in checks:
        actual = snapshot_managed_files(
            root,
            tuple(entry.reference for entry in expected.entries),
        )
        if actual != expected:
            raise RuntimeError(f"{name} managed source files changed during Phase 11")


def write_topology_dataset(
    result: TopologyAuditResult,
    output_root: str | Path,
) -> TopologyWriteResult:
    """Publish a fully validated Phase 11 topology audit into a fresh root."""
    if not isinstance(result, TopologyAuditResult):
        raise TypeError("result must be TopologyAuditResult")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Topology output root already exists: {output}")
    resolved_output = output.resolve()
    for source_name, source_root in (
        ("Phase 7", result.phase7_source_root),
        ("Phase 8", result.graph_source_root),
        ("Phase 9", result.action_source_root),
    ):
        resolved_source = Path(source_root).resolve()
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError(
                f"Topology output root must not be inside the {source_name} source root"
            )
    _verify_result_sources(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected existing staging directory: {staging}")

    try:
        (staging / "manifests").mkdir(parents=True)
        (staging / "artifacts" / "groups").mkdir(parents=True)
        managed: list[str] = []
        save_topology_config(result.config, staging / "topology_config.json")
        managed.append("topology_config.json")
        write_strict_json(staging / "topology_summary.json", result.summary)
        managed.append("topology_summary.json")
        for group in result.groups:
            reference = f"artifacts/groups/{group.topology_group_id}.npz"
            save_topology_group_artifact(group, result.config, staging / reference)
            managed.append(reference)
        writer = ManifestWriter(staging / "manifests")
        writer.write_records(
            "topology_group_manifest",
            result.group_records,
            overwrite=False,
        )
        managed.append("manifests/topology_group_manifest.parquet")

        persisted_config = load_topology_config(staging / "topology_config.json")
        if persisted_config != result.config:
            raise ValueError("Persisted topology config mismatch")
        reader = ManifestReader(staging / "manifests")
        records = reader.read_typed_records(
            "topology_group_manifest",
            TopologyGroupRecordV1,
        )
        loaded: dict[str, TopologyGroupResult] = {}
        for record in records:
            record.validate()
            group = load_topology_group_artifact(
                resolve_safe_file(
                    staging,
                    record.artifact_ref,
                    f"TopologyGroupRecordV1 {record.topology_group_id}.artifact_ref",
                ),
                persisted_config,
                record.content_hash,
            )
            if group.topology_group_id in loaded:
                raise ValueError(
                    f"Duplicate persisted topology group {group.topology_group_id}"
                )
            loaded[group.topology_group_id] = group
        validate_topology_dataset_joins(
            records,
            groups_by_id=loaded,
            config=persisted_config,
        )

        if len(set(managed)) != len(managed):
            raise ValueError("Managed Phase 11 file inventory contains duplicates")
        expected_before_marker = set(managed)
        actual_before_marker = _relative_file_set(staging)
        if actual_before_marker != expected_before_marker:
            raise ValueError(
                "Staging topology dataset inventory mismatch; "
                f"missing={sorted(expected_before_marker - actual_before_marker)}, "
                f"unexpected={sorted(actual_before_marker - expected_before_marker)}"
            )
        managed_files = tuple(sorted([*managed, "topology_complete.json"]))
        completion = {
            "complete": True,
            "source_scientific_generation_id": result.source_scientific_generation_id,
            "graph_conversion_id": result.graph_conversion_id,
            "action_engine_id": result.action_engine_id,
            "topology_audit_id": result.topology_audit_id,
            "operational_config_id": result.operational_config_id,
            "topology_schema_id": result.topology_schema_id,
            "group_count": len(result.groups),
            "point_count": result.summary["total_group_point_count"],
            "phase7_snapshot_hash": result.phase7_snapshot.aggregate_sha256,
            "graph_snapshot_hash": result.graph_snapshot.aggregate_sha256,
            "action_snapshot_hash": result.action_snapshot.aggregate_sha256,
            "topology_loss_weight": 0.0,
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "topology_complete.json", completion)
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed topology file inventory does not match staging")
        if strict_json_load(staging / "topology_complete.json") != completion:
            raise ValueError("topology_complete.json content mismatch")
        _verify_result_sources(result)
        if output.exists():
            raise FileExistsError(
                f"Topology output root appeared during publication: {output}"
            )
        os.replace(staging, output)
        manifest_paths = (
            output / "manifests" / "topology_group_manifest.parquet",
        )
        artifact_paths = tuple(
            sorted(
                [
                    output / reference
                    for reference in managed_files
                    if reference.startswith("artifacts/")
                ],
                key=lambda path: path.as_posix(),
            )
        )
        written_paths = tuple(
            sorted(
                [output / reference for reference in managed_files],
                key=lambda path: path.as_posix(),
            )
        )
        return TopologyWriteResult(
            output_root=output,
            topology_complete_path=output / "topology_complete.json",
            manifest_paths=manifest_paths,
            artifact_paths=artifact_paths,
            written_paths=written_paths,
            managed_files=managed_files,
            group_count=len(result.groups),
            point_count=result.summary["total_group_point_count"],
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "load_topology_group_artifact",
    "save_topology_group_artifact",
    "write_topology_dataset",
]
