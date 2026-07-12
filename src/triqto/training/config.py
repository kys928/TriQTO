"""Strict configuration for the Phase 14 TriQTO training engine."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    DEVICE_NAMES,
    DTYPE_NAMES,
    OPTIMIZER_NAMES,
    SCHEDULER_NAMES,
    TOPOLOGY_LOSS_WEIGHT,
    TRAINABLE_TASKS,
    TRAINING_SCHEMA_VERSION,
)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(
    value: Any,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < minimum or (maximum is not None and numeric > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be in {bound}")
    return numeric


def _choice(value: Any, name: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    normalized = value.strip()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}")
    return normalized


def _tasks(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of task names")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}[{index}] must be nonblank text")
        normalized = item.strip()
        if normalized not in TRAINABLE_TASKS:
            raise ValueError(
                f"{name}[{index}]={normalized!r} is not trainable in Phase 14; "
                f"allowed={list(TRAINABLE_TASKS)}"
            )
        result.append(normalized)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    expected = tuple(task for task in TRAINABLE_TASKS if task in result)
    if tuple(result) != expected:
        raise ValueError(
            f"{name} must follow fixed order {list(TRAINABLE_TASKS)}"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9

    def __post_init__(self) -> None:
        name = _choice(self.name, "optimizer.name", OPTIMIZER_NAMES)
        lr = _float(self.learning_rate, "optimizer.learning_rate", minimum=1e-12)
        wd = _float(self.weight_decay, "optimizer.weight_decay")
        beta1 = _float(self.beta1, "optimizer.beta1", maximum=0.999999)
        beta2 = _float(self.beta2, "optimizer.beta2", maximum=0.999999)
        if beta1 >= beta2:
            raise ValueError("optimizer.beta1 must be smaller than beta2")
        epsilon = _float(self.epsilon, "optimizer.epsilon", minimum=1e-16)
        momentum = _float(self.momentum, "optimizer.momentum", maximum=0.999999)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "learning_rate", lr)
        object.__setattr__(self, "weight_decay", wd)
        object.__setattr__(self, "beta1", beta1)
        object.__setattr__(self, "beta2", beta2)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "momentum", momentum)


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: str = "warmup_cosine"
    warmup_steps: int = 0
    minimum_learning_rate_ratio: float = 0.1

    def __post_init__(self) -> None:
        name = _choice(self.name, "scheduler.name", SCHEDULER_NAMES)
        warmup = _int(self.warmup_steps, "scheduler.warmup_steps")
        minimum = _float(
            self.minimum_learning_rate_ratio,
            "scheduler.minimum_learning_rate_ratio",
            maximum=1.0,
        )
        if name == "constant" and warmup != 0:
            raise ValueError("constant scheduler requires warmup_steps=0")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "warmup_steps", warmup)
        object.__setattr__(self, "minimum_learning_rate_ratio", minimum)


@dataclass(frozen=True, slots=True)
class LossConfig:
    diagnosis_type_weight: float = 1.0
    diagnosis_strength_weight: float = 0.5
    diagnosis_affected_qubit_weight: float = 0.5
    action_selection_weight: float = 1.0
    action_rank_distribution_weight: float = 0.5
    action_reward_weight: float = 0.25
    born_kl_weight: float = 1.0
    born_hellinger_weight: float = 1.0
    hilbert_to_born_weight: float = 1.0
    geometry_weight: float = 0.1
    uncertainty_weighting: bool = True
    privileged_oracle_loss_weight: float = 1.0
    topology_weight: float = TOPOLOGY_LOSS_WEIGHT

    def __post_init__(self) -> None:
        for name in (
            "diagnosis_type_weight",
            "diagnosis_strength_weight",
            "diagnosis_affected_qubit_weight",
            "action_selection_weight",
            "action_rank_distribution_weight",
            "action_reward_weight",
            "born_kl_weight",
            "born_hellinger_weight",
            "hilbert_to_born_weight",
            "geometry_weight",
            "privileged_oracle_loss_weight",
            "topology_weight",
        ):
            object.__setattr__(self, name, _float(getattr(self, name), f"loss.{name}"))
        object.__setattr__(
            self,
            "uncertainty_weighting",
            _bool(self.uncertainty_weighting, "loss.uncertainty_weighting"),
        )
        if self.topology_weight != 0.0:
            raise ValueError("Phase 14 topology loss weight must remain exactly 0.0")


@dataclass(frozen=True, slots=True)
class CurriculumStageConfig:
    name: str
    epochs: int
    tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("curriculum stage name must be nonblank text")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "epochs", _int(self.epochs, f"stage[{self.name}].epochs", minimum=1))
        object.__setattr__(self, "tasks", _tasks(self.tasks, f"stage[{self.name}].tasks"))


_DEFAULT_STAGES = (
    CurriculumStageConfig(
        name="single_task_foundation",
        epochs=1,
        tasks=("diagnosis", "action_ranking", "born_prediction", "hilbert_to_born"),
    ),
    CurriculumStageConfig(
        name="joint_multitask",
        epochs=1,
        tasks=("joint_multitask",),
    ),
    CurriculumStageConfig(
        name="hardware_masked_simulation",
        epochs=1,
        tasks=("hardware_masked",),
    ),
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Scientific recipe plus explicit execution and fail-only guardrails."""

    schema_version: str = TRAINING_SCHEMA_VERSION
    run_name: str = "triqto_phase14"
    seed: int = 2026
    stages: tuple[CurriculumStageConfig, ...] = _DEFAULT_STAGES
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    deterministic_algorithms: bool = True
    device: str = "cpu"
    dtype: str = "float32"
    num_workers: int = 0
    checkpoint_every_epochs: int = 1
    keep_best_checkpoint: bool = True
    early_stopping_patience: int = 0
    normalize_action_features: bool = True
    normalize_topology_features: bool = True
    max_items: int = 1_000_000
    max_nodes_per_batch: int = 65_536
    max_edges_per_batch: int = 262_144
    max_gates_per_batch: int = 262_144
    max_candidates_per_batch: int = 65_536
    max_outcomes_per_batch: int = 262_144
    max_hilbert_amplitudes_per_batch: int = 262_144
    topology_loss_weight: float = TOPOLOGY_LOSS_WEIGHT

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or self.schema_version.strip() != TRAINING_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {TRAINING_SCHEMA_VERSION!r}")
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("run_name must be nonblank text")
        seed = _int(self.seed, "seed")
        stages_raw = self.stages
        if isinstance(stages_raw, (str, bytes)) or not isinstance(stages_raw, Sequence):
            raise TypeError("stages must be a sequence")
        stages: list[CurriculumStageConfig] = []
        for index, stage in enumerate(stages_raw):
            if isinstance(stage, Mapping):
                stage = curriculum_stage_from_dict(stage)
            if not isinstance(stage, CurriculumStageConfig):
                raise TypeError(f"stages[{index}] must be CurriculumStageConfig or mapping")
            stages.append(stage)
        if not stages:
            raise ValueError("stages must not be empty")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("curriculum stage names must be unique")
        batch_size = _int(self.batch_size, "batch_size", minimum=1)
        accumulation = _int(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
            minimum=1,
        )
        max_grad = _float(self.max_gradient_norm, "max_gradient_norm", minimum=1e-12)
        optimizer = self.optimizer
        if isinstance(optimizer, Mapping):
            optimizer = optimizer_config_from_dict(optimizer)
        if not isinstance(optimizer, OptimizerConfig):
            raise TypeError("optimizer must be OptimizerConfig or mapping")
        scheduler = self.scheduler
        if isinstance(scheduler, Mapping):
            scheduler = scheduler_config_from_dict(scheduler)
        if not isinstance(scheduler, SchedulerConfig):
            raise TypeError("scheduler must be SchedulerConfig or mapping")
        loss = self.loss
        if isinstance(loss, Mapping):
            loss = loss_config_from_dict(loss)
        if not isinstance(loss, LossConfig):
            raise TypeError("loss must be LossConfig or mapping")
        deterministic = _bool(self.deterministic_algorithms, "deterministic_algorithms")
        device = _choice(self.device, "device", DEVICE_NAMES)
        dtype = _choice(self.dtype, "dtype", DTYPE_NAMES)
        workers = _int(self.num_workers, "num_workers")
        if workers != 0:
            raise ValueError("Phase 14 v1 requires num_workers=0 for exact ordering")
        checkpoint_every = _int(
            self.checkpoint_every_epochs,
            "checkpoint_every_epochs",
            minimum=1,
        )
        keep_best = _bool(self.keep_best_checkpoint, "keep_best_checkpoint")
        patience = _int(self.early_stopping_patience, "early_stopping_patience")
        normalize_action = _bool(self.normalize_action_features, "normalize_action_features")
        normalize_topology = _bool(self.normalize_topology_features, "normalize_topology_features")
        for name in (
            "max_items",
            "max_nodes_per_batch",
            "max_edges_per_batch",
            "max_gates_per_batch",
            "max_candidates_per_batch",
            "max_outcomes_per_batch",
            "max_hilbert_amplitudes_per_batch",
        ):
            object.__setattr__(self, name, _int(getattr(self, name), name, minimum=1))
        topology = _float(self.topology_loss_weight, "topology_loss_weight")
        if topology != 0.0 or loss.topology_weight != 0.0:
            raise ValueError("Phase 14 topology loss weight must remain exactly 0.0")
        object.__setattr__(self, "schema_version", TRAINING_SCHEMA_VERSION)
        object.__setattr__(self, "run_name", self.run_name.strip())
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "stages", tuple(stages))
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "gradient_accumulation_steps", accumulation)
        object.__setattr__(self, "max_gradient_norm", max_grad)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "scheduler", scheduler)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "deterministic_algorithms", deterministic)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "num_workers", workers)
        object.__setattr__(self, "checkpoint_every_epochs", checkpoint_every)
        object.__setattr__(self, "keep_best_checkpoint", keep_best)
        object.__setattr__(self, "early_stopping_patience", patience)
        object.__setattr__(self, "normalize_action_features", normalize_action)
        object.__setattr__(self, "normalize_topology_features", normalize_topology)
        object.__setattr__(self, "topology_loss_weight", 0.0)
        json.dumps(training_config_to_dict(self), sort_keys=True, allow_nan=False)

    @property
    def total_epochs(self) -> int:
        return sum(stage.epochs for stage in self.stages)

    @property
    def configured_tasks(self) -> tuple[str, ...]:
        present = {task for stage in self.stages for task in stage.tasks}
        return tuple(task for task in TRAINABLE_TASKS if task in present)


def optimizer_config_from_dict(payload: Mapping[str, Any]) -> OptimizerConfig:
    return _strict_dataclass(OptimizerConfig, payload, "optimizer config")


def scheduler_config_from_dict(payload: Mapping[str, Any]) -> SchedulerConfig:
    return _strict_dataclass(SchedulerConfig, payload, "scheduler config")


def loss_config_from_dict(payload: Mapping[str, Any]) -> LossConfig:
    return _strict_dataclass(LossConfig, payload, "loss config")


def curriculum_stage_from_dict(payload: Mapping[str, Any]) -> CurriculumStageConfig:
    return _strict_dataclass(CurriculumStageConfig, payload, "curriculum stage")


def _strict_dataclass(cls: type[Any], payload: Mapping[str, Any], name: str) -> Any:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} payload must be a mapping")
    allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown {name} fields: {sorted(extra)}")
    return cls(**dict(payload))


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    payload = asdict(config)
    payload["stages"] = [asdict(stage) for stage in config.stages]
    for stage in payload["stages"]:
        stage["tasks"] = list(stage["tasks"])
    return payload


def training_config_from_dict(payload: Mapping[str, Any]) -> TrainingConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("training config payload must be a mapping")
    allowed = set(TrainingConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown training config fields: {sorted(extra)}")
    values = dict(payload)
    if "stages" in values:
        raw = values["stages"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError("stages must be a sequence")
        values["stages"] = tuple(
            curriculum_stage_from_dict(stage) if isinstance(stage, Mapping) else stage
            for stage in raw
        )
    if "optimizer" in values and isinstance(values["optimizer"], Mapping):
        values["optimizer"] = optimizer_config_from_dict(values["optimizer"])
    if "scheduler" in values and isinstance(values["scheduler"], Mapping):
        values["scheduler"] = scheduler_config_from_dict(values["scheduler"])
    if "loss" in values and isinstance(values["loss"], Mapping):
        values["loss"] = loss_config_from_dict(values["loss"])
    return TrainingConfig(**values)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite training config constant: {value}")


def load_training_config(path: str | Path) -> TrainingConfig:
    target = Path(path)
    text = target.read_text()
    payload = (
        yaml.safe_load(text)
        if target.suffix.lower() in {".yaml", ".yml"}
        else json.loads(text, parse_constant=_reject_constant)
    )
    if not isinstance(payload, Mapping):
        raise TypeError("training config document must contain a mapping")
    return training_config_from_dict(payload)


def save_training_config(config: TrainingConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(training_config_to_dict(config), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    return target


__all__ = [
    "CurriculumStageConfig",
    "LossConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "curriculum_stage_from_dict",
    "load_training_config",
    "loss_config_from_dict",
    "optimizer_config_from_dict",
    "save_training_config",
    "scheduler_config_from_dict",
    "training_config_from_dict",
    "training_config_to_dict",
]
