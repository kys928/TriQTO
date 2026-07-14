"""Strict configuration for deterministic held-out Phase 15 evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .constants import ABLATION_NAMES, EVALUATION_SCHEMA_VERSION, EVALUATION_TASKS
from triqto.training_views.config import HOLDOUT_AXES

EVALUATION_DESIGNS = ("iid_test", "ood_axis_holdout")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value.strip()


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


def _float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _ordered_subset(value: Any, name: str, universe: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_text(item, f"{name} entry") for item in value)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    unknown = set(normalized) - set(universe)
    if unknown:
        raise ValueError(f"Unknown {name}: {sorted(unknown)}")
    expected = tuple(item for item in universe if item in normalized)
    if normalized != expected:
        raise ValueError(f"{name} must follow fixed order {list(universe)}")
    return normalized


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    schema_version: str = EVALUATION_SCHEMA_VERSION
    run_name: str = "triqto_phase15"
    seed: int = 2026
    tasks: tuple[str, ...] = EVALUATION_TASKS
    ablations: tuple[str, ...] = ABLATION_NAMES
    batch_size: int = 16
    calibration_bins: int = 10
    checkpoint_selection: str = "best"
    require_test_items: bool = True
    include_baseline_comparison: bool = False
    max_items: int = 1_000_000
    device: str = "cpu"
    dtype: str = "float32"
    distribution_epsilon: float = 1e-12
    evaluation_design: str = "iid_test"
    holdout_axis: str | None = None
    holdout_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {EVALUATION_SCHEMA_VERSION!r}"
            )
        run_name = _text(self.run_name, "run_name")
        seed = _int(self.seed, "seed")
        tasks = _ordered_subset(self.tasks, "tasks", EVALUATION_TASKS)
        ablations = _ordered_subset(self.ablations, "ablations", ABLATION_NAMES)
        if "full" not in ablations:
            raise ValueError("ablations must include 'full'")
        batch_size = _int(self.batch_size, "batch_size", minimum=1)
        bins = _int(self.calibration_bins, "calibration_bins", minimum=2)
        checkpoint = _text(self.checkpoint_selection, "checkpoint_selection")
        if checkpoint not in {"best", "final"}:
            raise ValueError("checkpoint_selection must be 'best' or 'final'")
        require_test = _bool(self.require_test_items, "require_test_items")
        include_baselines = _bool(
            self.include_baseline_comparison,
            "include_baseline_comparison",
        )
        max_items = _int(self.max_items, "max_items", minimum=1)
        device = _text(self.device, "device")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        dtype = _text(self.dtype, "dtype")
        if dtype != "float32":
            raise ValueError("Phase 15 v2 supports dtype=float32 only")
        epsilon = _float(
            self.distribution_epsilon,
            "distribution_epsilon",
            minimum=1e-18,
        )
        design = _text(self.evaluation_design, "evaluation_design")
        if design not in EVALUATION_DESIGNS:
            raise ValueError(f"evaluation_design must be one of {EVALUATION_DESIGNS}")
        holdout_axis = self.holdout_axis
        if holdout_axis is not None:
            holdout_axis = _text(holdout_axis, "holdout_axis")
            if holdout_axis not in HOLDOUT_AXES:
                raise ValueError(f"holdout_axis must be one of {HOLDOUT_AXES}")
        if isinstance(self.holdout_values, (str, bytes)) or not isinstance(
            self.holdout_values,
            Sequence,
        ):
            raise TypeError("holdout_values must be a sequence")
        holdout_values = tuple(
            sorted(_text(value, "holdout_values entry") for value in self.holdout_values)
        )
        if len(set(holdout_values)) != len(holdout_values):
            raise ValueError("holdout_values must not contain duplicates")
        if design == "iid_test":
            if holdout_axis is not None or holdout_values:
                raise ValueError(
                    "iid_test does not accept holdout_axis/holdout_values"
                )
        elif holdout_axis is None or not holdout_values:
            raise ValueError(
                "ood_axis_holdout requires holdout_axis and holdout_values"
            )
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "ablations", ablations)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "calibration_bins", bins)
        object.__setattr__(self, "checkpoint_selection", checkpoint)
        object.__setattr__(self, "require_test_items", require_test)
        object.__setattr__(self, "include_baseline_comparison", include_baselines)
        object.__setattr__(self, "max_items", max_items)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "distribution_epsilon", epsilon)
        object.__setattr__(self, "evaluation_design", design)
        object.__setattr__(self, "holdout_axis", holdout_axis)
        object.__setattr__(self, "holdout_values", holdout_values)
        json.dumps(evaluation_config_to_dict(self), sort_keys=True, allow_nan=False)


def evaluation_config_to_dict(config: EvaluationConfig) -> dict[str, Any]:
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be EvaluationConfig")
    payload = asdict(config)
    payload["tasks"] = list(config.tasks)
    payload["ablations"] = list(config.ablations)
    payload["holdout_values"] = list(config.holdout_values)
    return payload


def evaluation_config_from_dict(payload: Mapping[str, Any]) -> EvaluationConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("evaluation config payload must be a mapping")
    allowed = set(EvaluationConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown evaluation config fields: {sorted(extra)}")
    return EvaluationConfig(**dict(payload))


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite evaluation config constant: {value}")


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    target = Path(path)
    text = target.read_text()
    if target.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(payload, Mapping):
        raise TypeError("evaluation config document must contain a mapping")
    return evaluation_config_from_dict(payload)


def save_evaluation_config(config: EvaluationConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            evaluation_config_to_dict(config),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return target


__all__ = [
    "EVALUATION_DESIGNS",
    "EvaluationConfig",
    "evaluation_config_from_dict",
    "evaluation_config_to_dict",
    "load_evaluation_config",
    "save_evaluation_config",
]
