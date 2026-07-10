"""In-memory records for deterministic Phase 9 action generation and rollout."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from triqto.graph import CompletedPhase7Dataset, SourceFileSnapshot

from .config import ActionEngineConfig


@dataclass(frozen=True, slots=True)
class ActionEdit:
    """One bounded circuit edit applied before final measurements."""

    edit_type: str
    qubits: tuple[int, ...]
    magnitude: float


@dataclass(slots=True)
class ActionCandidate:
    """One deterministic candidate action for a distorted circuit."""

    action_id: str
    sample_id: str
    graph_pair_id: str
    source_circuit_id: str
    source_run_id: str
    distortion_id: str
    edits: tuple[ActionEdit, ...]
    generation_sources: tuple[str, ...]
    risk_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(slots=True)
class AppliedAction:
    """A validated action and the candidate circuit it produced."""

    action_id: str
    candidate_circuit_id: str
    circuit: QuantumCircuit
    source_depth: int
    candidate_depth: int
    source_gate_count: int
    candidate_gate_count: int
    decomposition_metadata: dict[str, Any] = field(default_factory=dict)
    circuit_hash: str = ""


@dataclass(slots=True)
class ActionRollout:
    """Ideal-simulator validation result for one candidate action."""

    rollout_id: str
    action_id: str
    sample_id: str
    graph_pair_id: str
    candidate_circuit_id: str
    clean_target_run_id: str
    scientific_config_id: str
    rank: int
    reward: float
    risk_score: float
    metric_names: np.ndarray
    baseline_metric_values: np.ndarray
    candidate_metric_values: np.ndarray
    improvement_values: np.ndarray
    outcome_bitstrings: np.ndarray
    exact_probabilities: np.ndarray
    dominates_baseline: bool
    primary_metric_nonworsening: bool
    selected: bool
    candidate_circuit: QuantumCircuit
    depth_delta: int
    gate_delta: int
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(slots=True)
class CompletedGraphDataset:
    """Validated read-only view of a completed Phase 8 graph dataset."""

    root: Path
    config: Any
    completion_marker: dict[str, Any]
    summary: dict[str, Any]
    graph_records: list[Any]
    pair_records: list[Any]
    graphs_by_id: dict[str, Any]
    pairs_by_id: dict[str, Any]
    pair_records_by_sample_id: dict[str, Any]
    managed_files: tuple[str, ...]
    snapshot: SourceFileSnapshot


@dataclass(slots=True)
class ActionEngineSources:
    """Cross-validated Phase 7 and Phase 8 source datasets."""

    phase7: CompletedPhase7Dataset
    graph: CompletedGraphDataset


@dataclass(slots=True)
class ActionEngineResult:
    """Complete in-memory result of Phase 9 candidate generation and validation."""

    phase7_source_root: Path
    graph_source_root: Path
    config: ActionEngineConfig
    source_scientific_generation_id: str
    graph_conversion_id: str
    action_engine_id: str
    operational_config_id: str
    action_schema_id: str
    candidates: list[ActionCandidate]
    rollouts: list[ActionRollout]
    candidate_records: list[Any]
    rollout_records: list[Any]
    phase7_snapshot: SourceFileSnapshot
    graph_snapshot: SourceFileSnapshot
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ActionWriteResult:
    """Committed paths returned by immutable Phase 9 publication."""

    output_root: Path
    action_complete_path: Path
    manifest_paths: tuple[Path, ...]
    artifact_paths: tuple[Path, ...]
    written_paths: tuple[Path, ...]
    managed_files: tuple[str, ...]
    candidate_count: int
    rollout_count: int


__all__ = [
    "ActionCandidate",
    "ActionEdit",
    "ActionEngineResult",
    "ActionEngineSources",
    "ActionRollout",
    "ActionWriteResult",
    "AppliedAction",
    "CompletedGraphDataset",
]
