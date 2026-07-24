"""Typed records and fixed dimensions for model-ready training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from triqto.model import TriQTOBatch

MODEL_READY_SOURCE_SCHEMA = "triqto.phase12.model_preprocessing.v1"
TOPOLOGY_ATTACHMENT_SCHEMA = "triqto.phase11_phase12.topology_attachment.v1"
MODEL_READY_ADAPTER_VERSION = "triqto.training.model_ready_adapter.v1"
CANONICAL_TOPOLOGY_FEATURE_DIM = 110
CANONICAL_ALIGNMENT_FEATURE_DIM = 11
CANONICAL_TOPOLOGY_INPUT_DIM = 121
PARAMETER_TOPOLOGY_ABLATION_DIM = 55
BORN_TOPOLOGY_ABLATION_DIM = 55


@dataclass(frozen=True, slots=True)
class ModelReadyDataset:
    root: Path
    completion_marker: dict[str, Any]
    input_contract: dict[str, Any]
    class_weights: dict[str, float]
    records: tuple[dict[str, Any], ...]
    records_by_task_split: dict[tuple[str, str], tuple[dict[str, Any], ...]]
    training_view_dataset_id: str
    manifest_sha256: str


@dataclass(slots=True)
class ModelReadyArtifact:
    record: dict[str, Any]
    inputs: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    metadata: dict[str, np.ndarray]


@dataclass(slots=True)
class ModelReadyActionTargets:
    should_act: Tensor
    should_act_weight: Tensor
    should_act_mask: Tensor
    ranking_loss_mask: Tensor
    candidate_reward: Tensor
    candidate_rank: Tensor
    candidate_selected_mask: Tensor
    candidate_listwise_distribution: Tensor
    candidate_target_mask: Tensor
    candidate_batch: Tensor


@dataclass(slots=True)
class ModelReadyExample:
    view_item_id: str
    entity_id: str
    task: str
    split: str
    split_group_id: str
    model_batch: TriQTOBatch
    action_targets: ModelReadyActionTargets
    targets: dict[str, np.ndarray]
    topology_ablation_inputs: dict[str, np.ndarray]
