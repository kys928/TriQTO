"""In-memory records for the Phase 11 persistent-homology audit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triqto.graph import SourceFileSnapshot

from .config import TopologyAuditConfig


@dataclass(slots=True)
class TopologyPointCloudGroup:
    """Aligned parameter/Hilbert/Born points with shared deterministic point IDs."""

    group_kind: str
    group_key: str
    point_ids: np.ndarray
    parameter_coordinate_names: np.ndarray
    parameter_coordinates: np.ndarray
    parameter_coordinate_mask: np.ndarray
    born_outcome_bitstrings: np.ndarray
    born_coordinates: np.ndarray
    statevectors: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PersistenceSummary:
    """Persistence diagrams, curves, and fixed-order features for one manifold."""

    manifold: str
    diagrams: dict[int, np.ndarray]
    betti_curves: dict[int, np.ndarray]
    feature_names: np.ndarray
    feature_values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TopologyGroupResult:
    """Complete topology audit result for one aligned point-cloud group."""

    topology_group_id: str
    topology_audit_id: str
    group_kind: str
    group_key: str
    point_ids: np.ndarray
    parameter_coordinate_names: np.ndarray
    parameter_coordinates: np.ndarray
    parameter_coordinate_mask: np.ndarray
    born_outcome_bitstrings: np.ndarray
    born_coordinates: np.ndarray
    parameter_distance_matrix: np.ndarray
    hilbert_distance_matrix: np.ndarray
    born_distance_matrix: np.ndarray
    filtration_grid: np.ndarray
    manifold_available_mask: np.ndarray
    persistence: dict[str, PersistenceSummary]
    topology_feature_names: np.ndarray
    topology_feature_values: np.ndarray
    alignment_feature_names: np.ndarray
    alignment_feature_values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(slots=True)
class TopologyAuditResult:
    """Complete in-memory Phase 11 topology audit dataset."""

    phase7_source_root: Path
    graph_source_root: Path
    action_source_root: Path
    config: TopologyAuditConfig
    source_scientific_generation_id: str
    graph_conversion_id: str
    action_engine_id: str
    topology_audit_id: str
    operational_config_id: str
    topology_schema_id: str
    groups: list[TopologyGroupResult]
    group_records: list[Any]
    phase7_snapshot: SourceFileSnapshot
    graph_snapshot: SourceFileSnapshot
    action_snapshot: SourceFileSnapshot
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TopologyWriteResult:
    """Committed paths returned by immutable Phase 11 publication."""

    output_root: Path
    topology_complete_path: Path
    manifest_paths: tuple[Path, ...]
    artifact_paths: tuple[Path, ...]
    written_paths: tuple[Path, ...]
    managed_files: tuple[str, ...]
    group_count: int
    point_count: int


class TopologyCache:
    """Small explicit in-memory cache with defensive copies and no global state."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._items)

    def contains(self, key: str) -> bool:
        return key in self._items

    def get(self, key: str) -> Any | None:
        value = self._items.get(key)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def put(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("TopologyCache key must be nonblank text")
        self._items[key] = value.copy() if isinstance(value, np.ndarray) else value

    def clear(self) -> None:
        self._items.clear()


__all__ = [
    "PersistenceSummary",
    "TopologyAuditResult",
    "TopologyCache",
    "TopologyGroupResult",
    "TopologyPointCloudGroup",
    "TopologyWriteResult",
]
