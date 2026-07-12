"""Versioned constants for the Phase 11 persistent-homology audit."""
from __future__ import annotations

TOPOLOGY_SCHEMA_VERSION = "triqto.topology.phase11.v1"
TOPOLOGY_ARTIFACT_SCHEMA_VERSION = "triqto.topology_group.npz.v1"
TOPOLOGY_GROUP_MANIFEST_VERSION = "triqto.topology_group_manifest.v1"
TOPOLOGY_DISTANCE_VERSION = "triqto.topology_distances.v1"
TOPOLOGY_FEATURE_VERSION = "triqto.topology_features.v1"
TOPOLOGY_ALIGNMENT_VERSION = "triqto.topology_alignment.v1"
TOPOLOGY_GROUPING_VERSION = "triqto.topology_groups.v1"
TOPOLOGY_PH_VERSION = "triqto.vietoris_rips.ripser.v1"

MANIFOLD_ORDER = ("parameter", "hilbert", "born")
GROUP_KINDS = (
    "action_neighborhood",
    "family_qubit_cohort",
    "family_qubit_distortion_cohort",
)
BORN_DISTANCE_NAMES = (
    "hellinger",
    "jensen_shannon",
    "fisher_rao",
)

DEFAULT_HOMOLOGY_DIMENSIONS = (0, 1)
DEFAULT_TOP_K_LIFETIMES = 8
DEFAULT_BETTI_GRID_SIZE = 32
DISTANCE_ATOL = 1e-12
PROBABILITY_ATOL = 1e-12

TOPOLOGY_METADATA_ARRAY_NAME = "topology_metadata_json_utf8"
BASE_ARRAY_NAMES = (
    "point_ids",
    "parameter_coordinate_names",
    "parameter_coordinates",
    "parameter_coordinate_mask",
    "parameter_distance_matrix",
    "hilbert_distance_matrix",
    "born_outcome_bitstrings",
    "born_coordinates",
    "born_distance_matrix",
    "filtration_grid",
    "manifold_available_mask",
    "topology_feature_names",
    "topology_feature_values",
    "alignment_feature_names",
    "alignment_feature_values",
)

__all__ = [
    "BASE_ARRAY_NAMES",
    "BORN_DISTANCE_NAMES",
    "DEFAULT_BETTI_GRID_SIZE",
    "DEFAULT_HOMOLOGY_DIMENSIONS",
    "DEFAULT_TOP_K_LIFETIMES",
    "DISTANCE_ATOL",
    "GROUP_KINDS",
    "MANIFOLD_ORDER",
    "PROBABILITY_ATOL",
    "TOPOLOGY_ALIGNMENT_VERSION",
    "TOPOLOGY_ARTIFACT_SCHEMA_VERSION",
    "TOPOLOGY_DISTANCE_VERSION",
    "TOPOLOGY_FEATURE_VERSION",
    "TOPOLOGY_GROUPING_VERSION",
    "TOPOLOGY_GROUP_MANIFEST_VERSION",
    "TOPOLOGY_METADATA_ARRAY_NAME",
    "TOPOLOGY_PH_VERSION",
    "TOPOLOGY_SCHEMA_VERSION",
]
