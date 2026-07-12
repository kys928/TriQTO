"""Deterministic identities and logical content hashes for Phase 11."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.graph.utils import json_copy

from .config import TopologyAuditConfig, topology_config_to_dict
from .constants import (
    MANIFOLD_ORDER,
    TOPOLOGY_ALIGNMENT_VERSION,
    TOPOLOGY_ARTIFACT_SCHEMA_VERSION,
    TOPOLOGY_DISTANCE_VERSION,
    TOPOLOGY_FEATURE_VERSION,
    TOPOLOGY_GROUPING_VERSION,
    TOPOLOGY_GROUP_MANIFEST_VERSION,
    TOPOLOGY_PH_VERSION,
    TOPOLOGY_SCHEMA_VERSION,
)
from .models import TopologyGroupResult


def topology_schema_id() -> str:
    return make_deterministic_id(
        "topologyschema",
        {
            "schema_version": TOPOLOGY_SCHEMA_VERSION,
            "artifact_schema_version": TOPOLOGY_ARTIFACT_SCHEMA_VERSION,
            "group_manifest_version": TOPOLOGY_GROUP_MANIFEST_VERSION,
            "grouping_version": TOPOLOGY_GROUPING_VERSION,
            "distance_version": TOPOLOGY_DISTANCE_VERSION,
            "persistent_homology_version": TOPOLOGY_PH_VERSION,
            "feature_version": TOPOLOGY_FEATURE_VERSION,
            "alignment_version": TOPOLOGY_ALIGNMENT_VERSION,
            "manifold_order": MANIFOLD_ORDER,
            "latent_manifold_available": False,
            "density_matrix_manifold_available": False,
        },
    )


def scientific_topology_config_payload(config: TopologyAuditConfig) -> dict[str, Any]:
    if not isinstance(config, TopologyAuditConfig):
        raise TypeError("config must be TopologyAuditConfig")
    return {
        "schema_version": config.schema_version,
        "group_kinds": list(config.group_kinds),
        "min_points": config.min_points,
        "homology_dimensions": list(config.homology_dimensions),
        "include_hilbert": config.include_hilbert,
        "born_distance": config.born_distance,
        "normalize_distance_matrices": config.normalize_distance_matrices,
        "raw_parameter_weight": config.raw_parameter_weight,
        "born_pullback_weight": config.born_pullback_weight,
        "hilbert_pullback_weight": config.hilbert_pullback_weight,
        "betti_grid_size": config.betti_grid_size,
        "top_k_lifetimes": config.top_k_lifetimes,
        "max_filtration": config.max_filtration,
    }


def topology_audit_id(
    source_scientific_generation_id: str,
    graph_conversion_id: str,
    action_engine_id: str,
    config: TopologyAuditConfig,
) -> str:
    return make_deterministic_id(
        "topologyaudit",
        {
            "source_scientific_generation_id": source_scientific_generation_id,
            "graph_conversion_id": graph_conversion_id,
            "action_engine_id": action_engine_id,
            "topology_schema_id": topology_schema_id(),
            "scientific_config": scientific_topology_config_payload(config),
        },
    )


def topology_operational_config_id(config: TopologyAuditConfig) -> str:
    return make_deterministic_id("topologyconfig", topology_config_to_dict(config))


def topology_group_id(
    audit_id: str,
    group_kind: str,
    group_key: str,
    point_ids: tuple[str, ...],
) -> str:
    return make_deterministic_id(
        "topologygroup",
        {
            "topology_audit_id": audit_id,
            "group_kind": group_kind,
            "group_key": group_key,
            "point_ids": list(point_ids),
            "grouping_version": TOPOLOGY_GROUPING_VERSION,
        },
    )


def _update_array_hash(hasher: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(canonical_json(list(contiguous.shape)).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(contiguous.tobytes(order="C"))
    hasher.update(b"\0")


def topology_group_content_hash(group: TopologyGroupResult) -> str:
    arrays: dict[str, np.ndarray] = {
        "point_ids": group.point_ids,
        "parameter_coordinate_names": group.parameter_coordinate_names,
        "parameter_coordinates": group.parameter_coordinates,
        "parameter_coordinate_mask": group.parameter_coordinate_mask,
        "born_outcome_bitstrings": group.born_outcome_bitstrings,
        "born_coordinates": group.born_coordinates,
        "parameter_distance_matrix": group.parameter_distance_matrix,
        "hilbert_distance_matrix": group.hilbert_distance_matrix,
        "born_distance_matrix": group.born_distance_matrix,
        "filtration_grid": group.filtration_grid,
        "manifold_available_mask": group.manifold_available_mask,
        "topology_feature_names": group.topology_feature_names,
        "topology_feature_values": group.topology_feature_values,
        "alignment_feature_names": group.alignment_feature_names,
        "alignment_feature_values": group.alignment_feature_values,
    }
    for manifold in MANIFOLD_ORDER:
        summary = group.persistence.get(manifold)
        if summary is None:
            continue
        arrays[f"{manifold}_feature_names"] = summary.feature_names
        arrays[f"{manifold}_feature_values"] = summary.feature_values
        for dimension, diagram in summary.diagrams.items():
            arrays[f"{manifold}_h{dimension}_diagram"] = diagram
        for dimension, curve in summary.betti_curves.items():
            arrays[f"{manifold}_h{dimension}_betti_curve"] = curve
    metadata = {
        "artifact_schema_version": TOPOLOGY_ARTIFACT_SCHEMA_VERSION,
        "topology_group_id": group.topology_group_id,
        "topology_audit_id": group.topology_audit_id,
        "group_kind": group.group_kind,
        "group_key": group.group_key,
        "metadata": json_copy(group.metadata),
    }
    hasher = hashlib.sha256()
    hasher.update(canonical_json(metadata).encode("utf-8"))
    hasher.update(b"\0")
    for name in sorted(arrays):
        _update_array_hash(hasher, name, arrays[name])
    return f"sha256:{hasher.hexdigest()}"


__all__ = [
    "scientific_topology_config_payload",
    "topology_audit_id",
    "topology_group_content_hash",
    "topology_group_id",
    "topology_operational_config_id",
    "topology_schema_id",
]
