"""In-memory records for deterministic Phase 10 baseline comparisons."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triqto.graph import SourceFileSnapshot

from .config import BaselineSuiteConfig


@dataclass(frozen=True, slots=True)
class OptimizerAxis:
    """One deterministic append-rotation coordinate in the optimizer parameterization."""

    kind: str
    qubits: tuple[int, ...]


@dataclass(slots=True)
class BaselineResult:
    """One baseline outcome for one Phase 7 sample."""

    baseline_result_id: str
    baseline_suite_id: str
    sample_id: str
    graph_pair_id: str
    baseline_name: str
    source_circuit_id: str
    clean_target_run_id: str
    selected_action_id: str | None
    metric_names: np.ndarray
    baseline_metric_values: np.ndarray
    result_metric_values: np.ndarray
    improvement_values: np.ndarray
    outcome_bitstrings: np.ndarray
    exact_probabilities: np.ndarray
    parameter_vector: np.ndarray
    objective_before: float
    objective_after: float
    objective_improvement: float
    success: bool
    evaluations: int
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(slots=True)
class CompletedActionDataset:
    """Validated read-only view of one completed Phase 9 action dataset."""

    root: Path
    config: Any
    completion_marker: dict[str, Any]
    summary: dict[str, Any]
    candidate_records: list[Any]
    rollout_records: list[Any]
    candidates_by_id: dict[str, Any]
    circuits_by_id: dict[str, Any]
    rollouts_by_id: dict[str, Any]
    rollouts_by_sample_id: dict[str, tuple[Any, ...]]
    managed_files: tuple[str, ...]
    snapshot: SourceFileSnapshot


@dataclass(slots=True)
class BaselineSources:
    """Cross-validated immutable Phase 7, Phase 8, and Phase 9 sources."""

    phase7: Any
    graph: Any
    action: CompletedActionDataset


@dataclass(slots=True)
class BaselineSuiteResult:
    """Complete in-memory Phase 10 baseline result universe."""

    phase7_source_root: Path
    graph_source_root: Path
    action_source_root: Path
    config: BaselineSuiteConfig
    source_scientific_generation_id: str
    graph_conversion_id: str
    action_engine_id: str
    baseline_suite_id: str
    operational_config_id: str
    baseline_schema_id: str
    results: list[BaselineResult]
    result_records: list[Any]
    phase7_snapshot: SourceFileSnapshot
    graph_snapshot: SourceFileSnapshot
    action_snapshot: SourceFileSnapshot
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BaselineWriteResult:
    """Committed Phase 10 output paths."""

    output_root: Path
    baseline_complete_path: Path
    manifest_paths: tuple[Path, ...]
    artifact_paths: tuple[Path, ...]
    written_paths: tuple[Path, ...]
    managed_files: tuple[str, ...]
    result_count: int
    sample_count: int


@dataclass(slots=True)
class EvaluationSnapshot:
    """One exact Born objective evaluation for an optimizer or transpiler baseline."""

    vector: np.ndarray
    metric_values: np.ndarray
    objective: float
    outcome_bitstrings: np.ndarray
    exact_probabilities: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BaselineResult",
    "BaselineSources",
    "BaselineSuiteResult",
    "BaselineWriteResult",
    "CompletedActionDataset",
    "EvaluationSnapshot",
    "OptimizerAxis",
]
