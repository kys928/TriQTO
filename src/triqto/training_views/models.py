"""In-memory records for deterministic Phase 12 task-specific views."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triqto.graph import SourceFileSnapshot

from .config import TrainingViewConfig


@dataclass(slots=True)
class CompletedTopologyDataset:
    """Validated read-only Phase 11 topology dataset consumed by Phase 12."""

    root: Path
    config: Any
    completion_marker: dict[str, Any]
    summary: dict[str, Any]
    records: list[Any]
    groups_by_id: dict[str, Any]
    records_by_id: dict[str, Any]
    managed_files: tuple[str, ...]
    snapshot: SourceFileSnapshot


@dataclass(slots=True)
class TrainingViewSources:
    """Cross-validated Phase 7/8/9/11 source chain."""

    phase7: Any
    graph: Any
    action: Any
    topology: CompletedTopologyDataset


@dataclass(slots=True)
class TrainingViewItem:
    """One materialized task-specific index/target artifact."""

    view_item_id: str
    training_view_id: str
    training_view_dataset_id: str
    task: str
    split: str
    split_group_id: str
    entity_id: str
    input_groups: tuple[str, ...]
    target_groups: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    hilbert_available_mask: bool
    topology_available_mask: bool
    privileged_target_available_mask: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(slots=True)
class TrainingViewDefinition:
    """Dataset-level definition for one Phase 12 task view."""

    training_view_id: str
    training_view_dataset_id: str
    task: str
    input_groups: tuple[str, ...]
    target_groups: tuple[str, ...]
    mask_policy: str
    split_policy: str
    item_count: int
    split_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingViewBuildResult:
    """Complete in-memory Phase 12 result before immutable publication."""

    phase7_source_root: Path
    graph_source_root: Path
    action_source_root: Path
    topology_source_root: Path
    config: TrainingViewConfig
    source_scientific_generation_id: str
    graph_conversion_id: str
    action_engine_id: str
    topology_audit_id: str
    training_view_dataset_id: str
    operational_config_id: str
    training_view_schema_id: str
    items: list[TrainingViewItem]
    definitions: list[TrainingViewDefinition]
    item_records: list[Any]
    definition_records: list[Any]
    phase7_snapshot: SourceFileSnapshot
    graph_snapshot: SourceFileSnapshot
    action_snapshot: SourceFileSnapshot
    topology_snapshot: SourceFileSnapshot
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingViewWriteResult:
    """Committed paths returned by immutable Phase 12 publication."""

    output_root: Path
    training_view_complete_path: Path
    manifest_paths: tuple[Path, ...]
    artifact_paths: tuple[Path, ...]
    written_paths: tuple[Path, ...]
    managed_files: tuple[str, ...]
    view_count: int
    item_count: int


__all__ = [
    "CompletedTopologyDataset",
    "TrainingViewBuildResult",
    "TrainingViewDefinition",
    "TrainingViewItem",
    "TrainingViewSources",
    "TrainingViewWriteResult",
]
