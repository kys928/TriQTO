"""Integrity validation for Phase 11 topology groups and manifests."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from triqto.storage.topology_schema import TopologyGroupRecordV1

from .config import TopologyAuditConfig
from .constants import GROUP_KINDS, MANIFOLD_ORDER
from .distances import validate_distance_matrix
from .identities import topology_group_content_hash, topology_group_id
from .models import PersistenceSummary, TopologyGroupResult
from .persistent_homology import validate_persistence_diagram


def _unicode_vector(value: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind != "U":
        raise TypeError(f"{name} must be a one-dimensional fixed-width Unicode array")
    if not allow_empty and value.size == 0:
        raise ValueError(f"{name} must not be empty")
    strings = [str(item) for item in value.tolist()]
    if any(not item for item in strings):
        raise ValueError(f"{name} must contain nonblank strings")
    return value


def _float64_array(value: Any, name: str, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        raise TypeError(f"{name} must be float64")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def validate_persistence_summary(
    summary: PersistenceSummary,
    filtration_grid: np.ndarray,
    config: TopologyAuditConfig,
) -> None:
    if not isinstance(summary, PersistenceSummary):
        raise TypeError("persistence values must be PersistenceSummary")
    if summary.manifold not in MANIFOLD_ORDER:
        raise ValueError(f"Unknown persistence manifold {summary.manifold!r}")
    if set(summary.diagrams) != set(config.homology_dimensions):
        raise ValueError("Persistence diagram dimensions do not match config")
    if set(summary.betti_curves) != set(config.homology_dimensions):
        raise ValueError("Betti curve dimensions do not match config")
    for dimension in config.homology_dimensions:
        validate_persistence_diagram(
            summary.diagrams[dimension],
            dimension,
            f"{summary.manifold} H{dimension} diagram",
        )
        curve = _float64_array(
            summary.betti_curves[dimension],
            f"{summary.manifold} H{dimension} Betti curve",
            1,
        )
        if curve.shape != filtration_grid.shape:
            raise ValueError("Betti curve shape must match filtration grid")
        if np.any(curve < 0.0) or not np.allclose(curve, np.rint(curve), atol=0.0):
            raise ValueError("Betti curves must contain nonnegative integer-valued counts")
    names = _unicode_vector(summary.feature_names, "persistence feature names")
    values = _float64_array(summary.feature_values, "persistence feature values", 1)
    if names.size != values.size:
        raise ValueError("Persistence feature names and values must have equal length")
    if len(set(names.tolist())) != names.size:
        raise ValueError("Persistence feature names must be unique")
    if not isinstance(summary.metadata, Mapping):
        raise TypeError("Persistence summary metadata must be a mapping")


def validate_topology_group_result(
    group: TopologyGroupResult,
    config: TopologyAuditConfig,
    *,
    require_hash: bool = True,
) -> None:
    if not isinstance(group, TopologyGroupResult):
        raise TypeError("group must be TopologyGroupResult")
    if not isinstance(config, TopologyAuditConfig):
        raise TypeError("config must be TopologyAuditConfig")
    for name in (
        "topology_group_id",
        "topology_audit_id",
        "group_kind",
        "group_key",
    ):
        value = getattr(group, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"TopologyGroupResult.{name} must be nonblank")
    if group.group_kind not in GROUP_KINDS:
        raise ValueError(f"Unknown topology group kind {group.group_kind!r}")
    point_ids = _unicode_vector(group.point_ids, "point_ids")
    points = int(point_ids.size)
    if points < config.min_points:
        raise ValueError("Topology group point count is below min_points")
    point_strings = point_ids.tolist()
    if point_strings != sorted(point_strings) or len(set(point_strings)) != points:
        raise ValueError("point_ids must be sorted and unique")
    expected_id = topology_group_id(
        group.topology_audit_id,
        group.group_kind,
        group.group_key,
        tuple(point_strings),
    )
    if group.topology_group_id != expected_id:
        raise ValueError("TopologyGroupResult identity mismatch")

    parameter_names = _unicode_vector(
        group.parameter_coordinate_names,
        "parameter_coordinate_names",
        allow_empty=True,
    )
    if parameter_names.tolist() != sorted(parameter_names.tolist()):
        raise ValueError("parameter_coordinate_names must be sorted")
    if len(set(parameter_names.tolist())) != parameter_names.size:
        raise ValueError("parameter_coordinate_names must be unique")
    coordinates = _float64_array(
        group.parameter_coordinates,
        "parameter_coordinates",
        2,
    )
    if coordinates.shape != (points, parameter_names.size):
        raise ValueError("parameter coordinate shape mismatch")
    if not isinstance(group.parameter_coordinate_mask, np.ndarray):
        raise TypeError("parameter_coordinate_mask must be a NumPy array")
    if group.parameter_coordinate_mask.dtype != np.bool_:
        raise TypeError("parameter_coordinate_mask must use bool dtype")
    if group.parameter_coordinate_mask.shape != coordinates.shape:
        raise ValueError("parameter_coordinate_mask shape mismatch")

    outcomes = _unicode_vector(group.born_outcome_bitstrings, "born outcomes")
    outcome_strings = outcomes.tolist()
    if outcome_strings != sorted(outcome_strings) or len(set(outcome_strings)) != outcomes.size:
        raise ValueError("Born outcomes must be sorted and unique")
    born = _float64_array(group.born_coordinates, "born_coordinates", 2)
    if born.shape != (points, outcomes.size):
        raise ValueError("Born coordinate shape mismatch")
    if np.any(born < 0.0):
        raise ValueError("Born coordinates must be nonnegative")
    if not np.allclose(np.sum(born, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Born coordinate rows must sum to one")

    for name, matrix in (
        ("parameter_distance_matrix", group.parameter_distance_matrix),
        ("hilbert_distance_matrix", group.hilbert_distance_matrix),
        ("born_distance_matrix", group.born_distance_matrix),
    ):
        validate_distance_matrix(matrix, name)
        if matrix.shape != (points, points):
            raise ValueError(f"{name} point-count shape mismatch")
    grid = _float64_array(group.filtration_grid, "filtration_grid", 1)
    if grid.size != config.betti_grid_size:
        raise ValueError("filtration_grid size mismatch")
    if not np.allclose(
        grid,
        np.linspace(0.0, config.max_filtration, config.betti_grid_size),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("filtration_grid content mismatch")
    if not isinstance(group.manifold_available_mask, np.ndarray):
        raise TypeError("manifold_available_mask must be a NumPy array")
    if group.manifold_available_mask.dtype != np.bool_:
        raise TypeError("manifold_available_mask must use bool dtype")
    if group.manifold_available_mask.shape != (len(MANIFOLD_ORDER),):
        raise ValueError("manifold_available_mask shape mismatch")
    if not bool(group.manifold_available_mask[0]) or not bool(
        group.manifold_available_mask[2]
    ):
        raise ValueError("Parameter and Born topology must be available")
    hilbert_available = bool(group.manifold_available_mask[1])
    expected_manifolds = {"parameter", "born"}
    if hilbert_available:
        expected_manifolds.add("hilbert")
    elif not np.allclose(group.hilbert_distance_matrix, 0.0, atol=0.0):
        raise ValueError("Unavailable Hilbert distance matrix must use zero placeholders")
    if set(group.persistence) != expected_manifolds:
        raise ValueError("Persistence manifolds do not match availability mask")
    for summary in group.persistence.values():
        validate_persistence_summary(summary, grid, config)

    topology_names = _unicode_vector(
        group.topology_feature_names,
        "topology_feature_names",
    )
    topology_values = _float64_array(
        group.topology_feature_values,
        "topology_feature_values",
        1,
    )
    if topology_names.size != topology_values.size:
        raise ValueError("Topology feature names and values length mismatch")
    if len(set(topology_names.tolist())) != topology_names.size:
        raise ValueError("Topology feature names must be unique")
    alignment_names = _unicode_vector(
        group.alignment_feature_names,
        "alignment_feature_names",
    )
    alignment_values = _float64_array(
        group.alignment_feature_values,
        "alignment_feature_values",
        1,
    )
    if alignment_names.size != alignment_values.size:
        raise ValueError("Alignment feature names and values length mismatch")
    if len(set(alignment_names.tolist())) != alignment_names.size:
        raise ValueError("Alignment feature names must be unique")
    if not isinstance(group.metadata, Mapping):
        raise TypeError("TopologyGroupResult.metadata must be a mapping")
    if group.metadata.get("topology_loss_weight") != 0.0:
        raise ValueError("Phase 11 topology_loss_weight must remain exactly zero")
    if group.metadata.get("topology_mode") != "audit_and_feature_only":
        raise ValueError("Phase 11 topology_mode must remain audit_and_feature_only")
    if group.metadata.get("latent_available") is not False:
        raise ValueError("Phase 11 must not fabricate latent topology")
    if group.metadata.get("raw_statevectors_persisted") is not False:
        raise ValueError("Phase 11 topology artifacts must not persist raw statevectors")

    expected_hash = topology_group_content_hash(group)
    if require_hash and group.content_hash != expected_hash:
        raise ValueError("TopologyGroupResult content_hash mismatch")
    if not require_hash and group.content_hash not in {"", expected_hash}:
        raise ValueError("TopologyGroupResult content_hash is malformed")


def validate_topology_dataset_joins(
    records: Sequence[TopologyGroupRecordV1],
    *,
    groups_by_id: Mapping[str, TopologyGroupResult] | None = None,
    config: TopologyAuditConfig,
) -> None:
    record_index: dict[str, TopologyGroupRecordV1] = {}
    artifact_refs: set[str] = set()
    kind_key_pairs: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, TopologyGroupRecordV1):
            raise TypeError("records must contain TopologyGroupRecordV1 values")
        record.validate()
        if record.topology_group_id in record_index:
            raise ValueError(
                f"Duplicate topology_group_id {record.topology_group_id}"
            )
        pair = (record.group_kind, record.group_key)
        if pair in kind_key_pairs:
            raise ValueError(f"Duplicate topology group kind/key {pair}")
        if record.artifact_ref in artifact_refs:
            raise ValueError(f"Duplicate topology artifact reference {record.artifact_ref}")
        record_index[record.topology_group_id] = record
        kind_key_pairs.add(pair)
        artifact_refs.add(record.artifact_ref)

    if groups_by_id is not None:
        if set(groups_by_id) != set(record_index):
            raise ValueError("Topology manifest IDs do not match loaded artifact IDs")
        for group_id, group in groups_by_id.items():
            record = record_index[group_id]
            validate_topology_group_result(group, config, require_hash=True)
            expected_manifolds = [
                name
                for index, name in enumerate(MANIFOLD_ORDER)
                if bool(group.manifold_available_mask[index])
            ]
            expected = {
                "topology_audit_id": group.topology_audit_id,
                "group_kind": group.group_kind,
                "group_key": group.group_key,
                "point_count": int(group.point_ids.size),
                "homology_dimensions": list(config.homology_dimensions),
                "manifolds": expected_manifolds,
                "content_hash": group.content_hash,
                "hilbert_available": bool(group.manifold_available_mask[1]),
                "latent_available": False,
                "topology_feature_dim": int(group.topology_feature_values.size),
                "alignment_feature_dim": int(group.alignment_feature_values.size),
            }
            for name, value in expected.items():
                if getattr(record, name) != value:
                    raise ValueError(
                        f"TopologyGroupRecordV1 {group_id} {name} mismatch"
                    )


__all__ = [
    "validate_persistence_summary",
    "validate_topology_dataset_joins",
    "validate_topology_group_result",
]
