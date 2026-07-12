"""Typed Phase 14 epoch and checkpoint manifest records."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar

from .schema import JsonMap, ManifestRecordMixin

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")
    return value


def _safe_ref(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed == PurePosixPath(".") or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise ValueError(f"{name} is unsafe")
    if parsed.as_posix() != value:
        raise ValueError(f"{name} must be normalized")
    return value


def _numeric_map(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key, item in value.items():
        _text(key, f"{name} key")
        _finite(item, f"{name}[{key}]")


@dataclass(slots=True)
class TrainingEpochRecordV1(ManifestRecordMixin):
    required_fields: ClassVar[tuple[str, ...]] = (
        "training_run_id",
        "training_recipe_id",
        "training_view_dataset_id",
        "model_architecture_id",
        "stage_name",
    )

    training_run_id: str
    training_recipe_id: str
    training_view_dataset_id: str
    model_architecture_id: str
    epoch: int
    stage_index: int
    stage_name: str
    active_tasks: list[str]
    global_step: int
    train_item_count: int
    validation_item_count: int
    train_batch_count: int
    validation_batch_count: int
    learning_rate: float
    gradient_norm: float
    train_total_loss: float
    validation_total_loss: float
    train_losses: JsonMap = field(default_factory=dict)
    validation_losses: JsonMap = field(default_factory=dict)
    mask_utilization: JsonMap = field(default_factory=dict)
    privileged_candidate_fraction: float = 0.0
    topology_loss_weight: float = 0.0

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _text(getattr(self, name), name)
        for name in (
            "epoch", "stage_index", "global_step", "train_item_count",
            "validation_item_count", "train_batch_count", "validation_batch_count",
        ):
            _integer(getattr(self, name), name)
        if not isinstance(self.active_tasks, list) or not self.active_tasks:
            raise ValueError("active_tasks must be a nonempty list")
        if len(set(self.active_tasks)) != len(self.active_tasks):
            raise ValueError("active_tasks must be unique")
        for value in self.active_tasks:
            _text(value, "active_tasks entry")
        for name in (
            "learning_rate", "gradient_norm", "train_total_loss",
            "validation_total_loss", "privileged_candidate_fraction",
            "topology_loss_weight",
        ):
            _finite(getattr(self, name), name)
        if self.topology_loss_weight != 0.0:
            raise ValueError("Phase 14 topology_loss_weight must remain zero")
        if not 0.0 <= self.privileged_candidate_fraction <= 1.0:
            raise ValueError("privileged_candidate_fraction must be in [0,1]")
        _numeric_map(self.train_losses, "train_losses")
        _numeric_map(self.validation_losses, "validation_losses")
        _numeric_map(self.mask_utilization, "mask_utilization")


@dataclass(slots=True)
class TrainingCheckpointRecordV1(ManifestRecordMixin):
    required_fields: ClassVar[tuple[str, ...]] = (
        "checkpoint_id",
        "training_run_id",
        "kind",
        "artifact_ref",
        "content_hash",
        "model_state_signature",
    )

    checkpoint_id: str
    training_run_id: str
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
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _text(getattr(self, name), name)
        if self.kind not in {"epoch", "best", "final"}:
            raise ValueError("checkpoint kind must be epoch, best, or final")
        _integer(self.epoch_completed, "epoch_completed")
        _integer(self.global_step, "global_step")
        _safe_ref(self.artifact_ref, "artifact_ref")
        _hash(self.content_hash, "content_hash")
        _hash(self.model_state_signature, "model_state_signature")
        _finite(self.validation_loss, "validation_loss")
        for name in (
            "optimizer_state_present", "scheduler_state_present", "rng_state_present"
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not all((self.optimizer_state_present, self.scheduler_state_present, self.rng_state_present)):
            raise ValueError("Phase 14 checkpoints must contain optimizer, scheduler, and RNG state")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


__all__ = ["TrainingCheckpointRecordV1", "TrainingEpochRecordV1"]
