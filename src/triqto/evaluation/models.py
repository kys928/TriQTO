"""Typed in-memory records for deterministic Phase 15 evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triqto.model import TriQTOModelConfig
from triqto.training import TrainingConfig, TrainingDataSpec

from .config import EvaluationConfig


@dataclass(slots=True)
class CompletedTrainingRun:
    root: Path
    completion_marker: dict[str, Any]
    summary: dict[str, Any]
    training_config: TrainingConfig
    model_config: TriQTOModelConfig
    data_spec: TrainingDataSpec
    checkpoint_record: Any
    checkpoint_path: Path
    checkpoint_metadata: dict[str, Any]
    managed_files: tuple[str, ...]
    snapshot_hash: str

    @property
    def training_run_id(self) -> str:
        value = self.completion_marker.get("training_run_id")
        if not isinstance(value, str) or not value:
            raise ValueError("Completed Phase 14 run has no training_run_id")
        return value


@dataclass(slots=True)
class CompletedBaselineDataset:
    root: Path
    completion_marker: dict[str, Any]
    config: Any
    records: list[Any]
    results_by_sample_and_name: dict[tuple[str, str], Any]
    managed_files: tuple[str, ...]
    snapshot_hash: str


@dataclass(slots=True)
class EvaluationItemResult:
    evaluation_item_id: str
    evaluation_run_id: str
    view_item_id: str
    entity_id: str
    task: str
    split: str
    ablation: str
    family: str | None
    n_qubits: int
    distortion_id: str | None
    metrics: dict[str, float]
    calibration: dict[str, float]
    predicted_action_id: str | None = None
    target_action_id: str | None = None
    target_action_rank: int | None = None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    artifact_ref: str = ""


@dataclass(slots=True)
class EvaluationAggregate:
    evaluation_aggregate_id: str
    evaluation_run_id: str
    task: str
    ablation: str
    group_dimension: str
    group_value: str
    item_count: int
    metrics: dict[str, float]
    calibration: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaselineComparison:
    evaluation_baseline_id: str
    evaluation_run_id: str
    sample_id: str
    baseline_name: str
    learned_action_id: str
    baseline_action_id: str | None
    objective_before: float
    learned_objective_after: float
    baseline_objective_after: float
    learned_minus_baseline: float
    learned_success: bool
    baseline_success: bool
    baseline_privileged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationRunResult:
    training_view_root: Path
    training_run_root: Path
    phase7_root: Path | None
    graph_root: Path | None
    action_root: Path | None
    baseline_root: Path | None
    config: EvaluationConfig
    evaluation_schema_id: str
    evaluation_recipe_id: str
    operational_config_id: str
    evaluation_run_id: str
    training_view_dataset_id: str
    training_run_id: str
    checkpoint_id: str
    item_results: list[EvaluationItemResult]
    aggregates: list[EvaluationAggregate]
    baseline_comparisons: list[BaselineComparison]
    summary: dict[str, Any]


__all__ = [
    "BaselineComparison",
    "CompletedBaselineDataset",
    "CompletedTrainingRun",
    "EvaluationAggregate",
    "EvaluationItemResult",
    "EvaluationRunResult",
]
