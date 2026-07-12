"""Typed in-memory records for Phase 14 data, epochs, and training runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from triqto.core.ids import canonical_json
from triqto.model import TriQTOBatch

from .config import TrainingConfig


@dataclass(frozen=True, slots=True)
class ManagedFileEntry:
    reference: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ManagedFileSnapshot:
    entries: tuple[ManagedFileEntry, ...]
    aggregate_sha256: str


@dataclass(slots=True)
class CompletedTrainingViewDataset:
    root: Path
    config: Any
    completion_marker: dict[str, Any]
    summary: dict[str, Any]
    definition_records: list[Any]
    item_records: list[Any]
    records_by_id: dict[str, Any]
    records_by_task_split: dict[tuple[str, str], tuple[Any, ...]]
    graph_anchor_record_by_entity_id: dict[str, Any]
    managed_files: tuple[str, ...]
    snapshot: ManagedFileSnapshot

    @property
    def training_view_dataset_id(self) -> str:
        value = self.completion_marker.get("training_view_dataset_id")
        if not isinstance(value, str) or not value:
            raise ValueError("Completed Phase 12 source has no training_view_dataset_id")
        return value


@dataclass(frozen=True, slots=True)
class TrainingDataSpec:
    """Training-only feature vocabulary and normalization derived from train split."""

    training_view_dataset_id: str
    distortion_labels: tuple[str, ...]
    distortion_mapping: tuple[tuple[str, str], ...]
    action_edit_types: tuple[str, ...]
    action_edit_mapping: tuple[tuple[str, str], ...]
    action_feature_names: tuple[str, ...]
    action_feature_mean: tuple[float, ...]
    action_feature_std: tuple[float, ...]
    topology_feature_names: tuple[str, ...]
    topology_feature_mean: tuple[float, ...]
    topology_feature_std: tuple[float, ...]
    topology_input_dim: int
    normalize_action_features: bool
    normalize_topology_features: bool
    adapter_version: str

    def validate(self) -> None:
        if not self.training_view_dataset_id:
            raise ValueError("training_view_dataset_id must be nonblank")
        for name, values in (
            ("distortion_labels", self.distortion_labels),
            ("action_edit_types", self.action_edit_types),
            ("action_feature_names", self.action_feature_names),
            ("topology_feature_names", self.topology_feature_names),
        ):
            if len(set(values)) != len(values) or any(not value for value in values):
                raise ValueError(f"{name} must contain unique nonblank strings")
        if len(self.action_feature_mean) != len(self.action_feature_names):
            raise ValueError("action_feature_mean width mismatch")
        if len(self.action_feature_std) != len(self.action_feature_names):
            raise ValueError("action_feature_std width mismatch")
        if any(value <= 0 for value in self.action_feature_std):
            raise ValueError("action_feature_std must be strictly positive")
        if len(self.topology_feature_mean) != len(self.topology_feature_names):
            raise ValueError("topology_feature_mean width mismatch")
        if len(self.topology_feature_std) != len(self.topology_feature_names):
            raise ValueError("topology_feature_std width mismatch")
        if any(value <= 0 for value in self.topology_feature_std):
            raise ValueError("topology_feature_std must be strictly positive")
        if self.topology_input_dim <= 0:
            raise ValueError("topology_input_dim must be positive")
        if len(self.topology_feature_names) > self.topology_input_dim:
            raise ValueError("topology feature vocabulary exceeds model input dimension")
        if not isinstance(self.normalize_action_features, bool):
            raise TypeError("normalize_action_features must be bool")
        if not isinstance(self.normalize_topology_features, bool):
            raise TypeError("normalize_topology_features must be bool")
        if not self.adapter_version:
            raise ValueError("adapter_version must be nonblank")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "training_view_dataset_id": self.training_view_dataset_id,
            "distortion_labels": list(self.distortion_labels),
            "distortion_mapping": [list(value) for value in self.distortion_mapping],
            "action_edit_types": list(self.action_edit_types),
            "action_edit_mapping": [list(value) for value in self.action_edit_mapping],
            "action_feature_names": list(self.action_feature_names),
            "action_feature_mean": list(self.action_feature_mean),
            "action_feature_std": list(self.action_feature_std),
            "topology_feature_names": list(self.topology_feature_names),
            "topology_feature_mean": list(self.topology_feature_mean),
            "topology_feature_std": list(self.topology_feature_std),
            "topology_input_dim": self.topology_input_dim,
            "normalize_action_features": self.normalize_action_features,
            "normalize_topology_features": self.normalize_topology_features,
            "adapter_version": self.adapter_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingDataSpec":
        if not isinstance(payload, dict):
            raise TypeError("TrainingDataSpec payload must be a dictionary")
        expected = {
            "training_view_dataset_id",
            "distortion_labels",
            "distortion_mapping",
            "action_edit_types",
            "action_edit_mapping",
            "action_feature_names",
            "action_feature_mean",
            "action_feature_std",
            "topology_feature_names",
            "topology_feature_mean",
            "topology_feature_std",
            "topology_input_dim",
            "normalize_action_features",
            "normalize_topology_features",
            "adapter_version",
        }
        if set(payload) != expected:
            raise ValueError(
                "TrainingDataSpec key mismatch; "
                f"missing={sorted(expected - set(payload))}, "
                f"unexpected={sorted(set(payload) - expected)}"
            )
        result = cls(
            training_view_dataset_id=payload["training_view_dataset_id"],
            distortion_labels=tuple(payload["distortion_labels"]),
            distortion_mapping=tuple(tuple(value) for value in payload["distortion_mapping"]),
            action_edit_types=tuple(payload["action_edit_types"]),
            action_edit_mapping=tuple(tuple(value) for value in payload["action_edit_mapping"]),
            action_feature_names=tuple(payload["action_feature_names"]),
            action_feature_mean=tuple(float(value) for value in payload["action_feature_mean"]),
            action_feature_std=tuple(float(value) for value in payload["action_feature_std"]),
            topology_feature_names=tuple(payload["topology_feature_names"]),
            topology_feature_mean=tuple(float(value) for value in payload["topology_feature_mean"]),
            topology_feature_std=tuple(float(value) for value in payload["topology_feature_std"]),
            topology_input_dim=int(payload["topology_input_dim"]),
            normalize_action_features=payload["normalize_action_features"],
            normalize_topology_features=payload["normalize_topology_features"],
            adapter_version=payload["adapter_version"],
        )
        result.validate()
        return result

    @property
    def content_hash(self) -> str:
        import hashlib

        digest = hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


@dataclass(slots=True)
class DiagnosisTargets:
    class_index: Tensor
    class_mask: Tensor
    strength: Tensor
    strength_mask: Tensor
    affected_qubit: Tensor
    affected_qubit_mask: Tensor


@dataclass(slots=True)
class ActionTargets:
    rank: Tensor
    reward: Tensor
    selected_mask: Tensor
    candidate_target_mask: Tensor
    privileged_oracle_mask: Tensor
    candidate_batch: Tensor


@dataclass(slots=True)
class BornTargets:
    probabilities: Tensor
    outcome_batch: Tensor
    row_mask: Tensor


@dataclass(slots=True)
class GeometryTargets:
    target_distance: Tensor
    pair_mask: Tensor


@dataclass(slots=True)
class TrainingTargets:
    diagnosis: DiagnosisTargets
    action: ActionTargets
    born_prediction: BornTargets
    hilbert_to_born: BornTargets
    geometry: GeometryTargets


@dataclass(slots=True)
class TrainingExample:
    view_item_id: str
    entity_id: str
    task: str
    split: str
    split_group_id: str
    model_batch: TriQTOBatch
    targets: TrainingTargets
    n_qubits: int
    born_distribution: tuple[tuple[str, float], ...]
    hilbert_state: Tensor | None
    privileged_target_available: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SupervisedBatch:
    item_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    tasks: tuple[str, ...]
    splits: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    model_batch: TriQTOBatch
    auxiliary_hilbert_to_born_batch: TriQTOBatch | None
    targets: TrainingTargets
    graph_task_names: tuple[str, ...]
    privileged_item_mask: Tensor

    @property
    def graph_count(self) -> int:
        return self.model_batch.graph.graph_count


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    stage_index: int
    stage_name: str
    active_tasks: tuple[str, ...]
    global_step: int
    train_item_count: int
    validation_item_count: int
    train_batch_count: int
    validation_batch_count: int
    learning_rate: float
    gradient_norm: float
    train_total_loss: float
    validation_total_loss: float
    train_losses: dict[str, float]
    validation_losses: dict[str, float]
    mask_utilization: dict[str, float]
    privileged_candidate_fraction: float
    topology_loss_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    checkpoint_id: str
    kind: str
    epoch_completed: int
    global_step: int
    artifact_ref: str
    content_hash: str
    model_state_signature: str
    validation_loss: float
    optimizer_state_present: bool
    scheduler_state_present: bool
    rng_state_present: bool


@dataclass(slots=True)
class TrainingRunResult:
    training_view_root: Path
    phase7_root: Path | None
    config: TrainingConfig
    data_spec: TrainingDataSpec
    training_schema_id: str
    training_recipe_id: str
    operational_config_id: str
    training_run_id: str
    model_architecture_id: str
    model_config_id: str
    training_view_dataset_id: str
    source_snapshot: ManagedFileSnapshot
    phase7_snapshot: ManagedFileSnapshot | None
    epoch_metrics: list[EpochMetrics]
    checkpoints: list[CheckpointSummary]
    best_epoch: int
    best_validation_loss: float
    final_epoch: int
    global_step: int
    stopped_early: bool
    resumed_from_checkpoint_id: str | None
    summary: dict[str, Any]


__all__ = [
    "ActionTargets",
    "BornTargets",
    "CheckpointSummary",
    "CompletedTrainingViewDataset",
    "DiagnosisTargets",
    "EpochMetrics",
    "GeometryTargets",
    "ManagedFileEntry",
    "ManagedFileSnapshot",
    "SupervisedBatch",
    "TrainingDataSpec",
    "TrainingExample",
    "TrainingRunResult",
    "TrainingTargets",
]
