"""Strict configuration for deterministic Phase 12 task-specific views."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .constants import TASK_ORDER, TRAINING_VIEW_SCHEMA_VERSION


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _require_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return numeric


def _require_tasks(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("tasks must be a sequence of strings")
    tasks: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"tasks[{index}] must be nonblank text")
        tasks.append(item.strip())
    if not tasks:
        raise ValueError("tasks must not be empty")
    if len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be unique")
    unknown = set(tasks) - set(TASK_ORDER)
    if unknown:
        raise ValueError(f"Unknown Phase 12 tasks: {sorted(unknown)}")
    expected = tuple(name for name in TASK_ORDER if name in tasks)
    if tuple(tasks) != expected:
        raise ValueError(f"tasks must follow the fixed Phase 12 order: {list(TASK_ORDER)}")
    return tuple(tasks)


@dataclass(frozen=True, slots=True)
class TrainingViewConfig:
    """Scientific split/mask policy plus fail-only operational ceilings."""

    schema_version: str = TRAINING_VIEW_SCHEMA_VERSION
    tasks: tuple[str, ...] = TASK_ORDER
    split_seed: int = 2026
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_grouping: str = "clean_circuit_id"
    topology_cross_split_policy: str = "audit_only"
    include_hilbert: bool = True
    include_topology: bool = True
    allow_empty_hilbert_view: bool = True
    topology_loss_weight: float = 0.0
    max_items: int = 1000000
    max_candidates_per_item: int = 4096
    max_source_refs_per_item: int = 16384

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be nonblank text")
        schema = self.schema_version.strip()
        if schema != TRAINING_VIEW_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported training-view schema {schema!r}; expected "
                f"{TRAINING_VIEW_SCHEMA_VERSION!r}"
            )
        tasks = _require_tasks(self.tasks)
        seed = _require_int(self.split_seed, "split_seed")
        train = _require_fraction(self.train_fraction, "train_fraction")
        validation = _require_fraction(
            self.validation_fraction,
            "validation_fraction",
        )
        test = _require_fraction(self.test_fraction, "test_fraction")
        if not math.isclose(train + validation + test, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("train/validation/test fractions must sum to exactly one")
        if not isinstance(self.split_grouping, str):
            raise TypeError("split_grouping must be text")
        split_grouping = self.split_grouping.strip()
        if split_grouping != "clean_circuit_id":
            raise ValueError(
                "Phase 12 split_grouping must remain clean_circuit_id to prevent "
                "related distortion/action leakage"
            )
        if not isinstance(self.topology_cross_split_policy, str):
            raise TypeError("topology_cross_split_policy must be text")
        topology_policy = self.topology_cross_split_policy.strip()
        if topology_policy != "audit_only":
            raise ValueError("Phase 12 topology_cross_split_policy must remain audit_only")
        include_hilbert = _require_bool(self.include_hilbert, "include_hilbert")
        include_topology = _require_bool(self.include_topology, "include_topology")
        allow_empty = _require_bool(
            self.allow_empty_hilbert_view,
            "allow_empty_hilbert_view",
        )
        loss_weight = _require_fraction(
            self.topology_loss_weight,
            "topology_loss_weight",
        )
        if loss_weight != 0.0:
            raise ValueError("Phase 12 topology_loss_weight must remain exactly 0.0")
        max_items = _require_int(self.max_items, "max_items", positive=True)
        max_candidates = _require_int(
            self.max_candidates_per_item,
            "max_candidates_per_item",
            positive=True,
        )
        max_refs = _require_int(
            self.max_source_refs_per_item,
            "max_source_refs_per_item",
            positive=True,
        )
        if "topology_audit" in tasks and not include_topology:
            raise ValueError("topology_audit task requires include_topology=true")
        if "hilbert_to_born" in tasks and not include_hilbert and not allow_empty:
            raise ValueError(
                "hilbert_to_born with include_hilbert=false requires "
                "allow_empty_hilbert_view=true"
            )

        object.__setattr__(self, "schema_version", schema)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "split_seed", seed)
        object.__setattr__(self, "train_fraction", train)
        object.__setattr__(self, "validation_fraction", validation)
        object.__setattr__(self, "test_fraction", test)
        object.__setattr__(self, "split_grouping", split_grouping)
        object.__setattr__(self, "topology_cross_split_policy", topology_policy)
        object.__setattr__(self, "include_hilbert", include_hilbert)
        object.__setattr__(self, "include_topology", include_topology)
        object.__setattr__(self, "allow_empty_hilbert_view", allow_empty)
        object.__setattr__(self, "topology_loss_weight", loss_weight)
        object.__setattr__(self, "max_items", max_items)
        object.__setattr__(self, "max_candidates_per_item", max_candidates)
        object.__setattr__(self, "max_source_refs_per_item", max_refs)
        json.dumps(training_view_config_to_dict(self), sort_keys=True, allow_nan=False)


def training_view_config_to_dict(config: TrainingViewConfig) -> dict[str, Any]:
    if not isinstance(config, TrainingViewConfig):
        raise TypeError("config must be TrainingViewConfig")
    payload = asdict(config)
    payload["tasks"] = list(config.tasks)
    return payload


def training_view_config_from_dict(payload: Mapping[str, Any]) -> TrainingViewConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("training-view config payload must be a mapping")
    allowed = set(TrainingViewConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown training-view config fields: {sorted(extra)}")
    return TrainingViewConfig(**dict(payload))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite training-view config constant: {value}")


def load_training_view_config(path: str | Path) -> TrainingViewConfig:
    target = Path(path)
    text = target.read_text()
    if target.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text, parse_constant=_reject_json_constant)
    if not isinstance(payload, Mapping):
        raise TypeError("training-view config document must contain a mapping")
    return training_view_config_from_dict(payload)


def save_training_view_config(config: TrainingViewConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            training_view_config_to_dict(config),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return target


__all__ = [
    "TrainingViewConfig",
    "load_training_view_config",
    "save_training_view_config",
    "training_view_config_from_dict",
    "training_view_config_to_dict",
]
