"""Typed model-ready examples and vectorized multi-task batches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from triqto.model import TriQTOBatch

from .types import ModelReadyActionTargets, ModelReadyExample


@dataclass(slots=True)
class ModelReadyDiagnosisTargets:
    class_index: Tensor
    class_mask: Tensor
    strength: Tensor
    strength_mask: Tensor
    affected_qubit: Tensor
    affected_qubit_mask: Tensor


@dataclass(slots=True)
class ModelReadyBornTargets:
    probabilities: Tensor
    outcome_batch: Tensor
    row_mask: Tensor


@dataclass(slots=True)
class ModelReadyGeometryTargets:
    target_distance: Tensor
    pair_mask: Tensor


@dataclass(slots=True)
class ModelReadyMultitaskExample:
    base: ModelReadyExample
    diagnosis_targets: ModelReadyDiagnosisTargets
    born_targets: ModelReadyBornTargets
    n_qubits: int
    born_distribution: tuple[tuple[str, float], ...]
    metadata: dict[str, Any]

    @property
    def view_item_id(self) -> str:
        return self.base.view_item_id

    @property
    def entity_id(self) -> str:
        return self.base.entity_id

    @property
    def task(self) -> str:
        return self.base.task

    @property
    def split(self) -> str:
        return self.base.split

    @property
    def split_group_id(self) -> str:
        return self.base.split_group_id

    @property
    def model_batch(self) -> TriQTOBatch:
        return self.base.model_batch

    @property
    def action_targets(self) -> ModelReadyActionTargets:
        return self.base.action_targets


@dataclass(slots=True)
class ModelReadyMultitaskTargets:
    diagnosis: ModelReadyDiagnosisTargets
    action: ModelReadyActionTargets
    born_prediction: ModelReadyBornTargets
    geometry: ModelReadyGeometryTargets


@dataclass(slots=True)
class ModelReadySupervisedBatch:
    item_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    tasks: tuple[str, ...]
    splits: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    model_batch: TriQTOBatch
    targets: ModelReadyMultitaskTargets

    @property
    def graph_count(self) -> int:
        return self.model_batch.graph.graph_count


__all__ = [
    "ModelReadyBornTargets",
    "ModelReadyDiagnosisTargets",
    "ModelReadyGeometryTargets",
    "ModelReadyMultitaskExample",
    "ModelReadyMultitaskTargets",
    "ModelReadySupervisedBatch",
]
