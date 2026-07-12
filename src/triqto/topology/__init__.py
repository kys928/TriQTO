"""Public Phase 11 persistent-homology and topology-audit APIs."""
from __future__ import annotations

from .alignment import (
    bottleneck_distance,
    build_alignment_features,
    finite_diagram,
    wasserstein_distance_1,
)
from .artifacts import (
    load_topology_group_artifact,
    save_topology_group_artifact,
    write_topology_dataset,
)
from .config import (
    TopologyAuditConfig,
    load_topology_config,
    save_topology_config,
    topology_config_from_dict,
    topology_config_to_dict,
)
from .constants import GROUP_KINDS, MANIFOLD_ORDER
from .distances import (
    born_distance_matrix,
    circular_parameter_distance_matrix,
    compute_manifold_distance_matrices,
    fubini_study_distance_matrix,
    induced_parameter_distance_matrix,
    normalize_distance_matrix,
    validate_distance_matrix,
)
from .features import (
    build_persistence_summary,
    diagram_statistics,
    finite_lifetimes,
    persistence_entropy,
)
from .identities import (
    scientific_topology_config_payload,
    topology_audit_id,
    topology_group_content_hash,
    topology_group_id,
    topology_operational_config_id,
    topology_schema_id,
)
from .models import (
    PersistenceSummary,
    TopologyAuditResult,
    TopologyCache,
    TopologyGroupResult,
    TopologyPointCloudGroup,
    TopologyWriteResult,
)
from .persistent_homology import (
    betti_curve,
    compute_persistence_diagrams,
    make_filtration_grid,
    validate_persistence_diagram,
)
from .pipeline import build_topology_audit_result
from .point_clouds import build_point_cloud_group
from .source import load_topology_sources, verify_topology_source_snapshots
from .topology_cache import make_topology_cache
from .topology_groups import TopologyGroupSpec, build_topology_group_specs
from .validators import (
    validate_persistence_summary,
    validate_topology_dataset_joins,
    validate_topology_group_result,
)

__all__ = [
    "GROUP_KINDS",
    "MANIFOLD_ORDER",
    "PersistenceSummary",
    "TopologyAuditConfig",
    "TopologyAuditResult",
    "TopologyCache",
    "TopologyGroupResult",
    "TopologyGroupSpec",
    "TopologyPointCloudGroup",
    "TopologyWriteResult",
    "betti_curve",
    "born_distance_matrix",
    "bottleneck_distance",
    "build_alignment_features",
    "build_persistence_summary",
    "build_point_cloud_group",
    "build_topology_audit_result",
    "build_topology_group_specs",
    "circular_parameter_distance_matrix",
    "compute_manifold_distance_matrices",
    "compute_persistence_diagrams",
    "diagram_statistics",
    "finite_diagram",
    "finite_lifetimes",
    "fubini_study_distance_matrix",
    "induced_parameter_distance_matrix",
    "load_topology_config",
    "load_topology_group_artifact",
    "load_topology_sources",
    "make_filtration_grid",
    "make_topology_cache",
    "normalize_distance_matrix",
    "persistence_entropy",
    "save_topology_config",
    "save_topology_group_artifact",
    "scientific_topology_config_payload",
    "topology_audit_id",
    "topology_config_from_dict",
    "topology_config_to_dict",
    "topology_group_content_hash",
    "topology_group_id",
    "topology_operational_config_id",
    "topology_schema_id",
    "validate_distance_matrix",
    "validate_persistence_diagram",
    "validate_persistence_summary",
    "validate_topology_dataset_joins",
    "validate_topology_group_result",
    "verify_topology_source_snapshots",
    "wasserstein_distance_1",
    "write_topology_dataset",
]
