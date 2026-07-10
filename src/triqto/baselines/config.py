"""Strict configuration for deterministic Phase 10 baseline comparisons."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .constants import BASELINE_NAMES, BASELINE_SCHEMA_VERSION


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _require_int(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_float(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _require_enabled(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("enabled_baselines must be a sequence of baseline names")
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"enabled_baselines[{index}] must be nonblank text")
        names.append(item.strip())
    if not names:
        raise ValueError("enabled_baselines must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("enabled_baselines must be unique")
    unknown = set(names) - set(BASELINE_NAMES)
    if unknown:
        raise ValueError(f"Unknown baseline names: {sorted(unknown)}")
    expected = tuple(name for name in BASELINE_NAMES if name in names)
    if tuple(names) != expected:
        raise ValueError(
            "enabled_baselines must follow the fixed Phase 10 baseline order: "
            f"{list(BASELINE_NAMES)}"
        )
    return tuple(names)


def _require_weights(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("metric_weights must be a three-value sequence")
    if len(value) != 3:
        raise ValueError("metric_weights must contain exactly three values")
    weights = tuple(
        _require_float(item, f"metric_weights[{index}]", nonnegative=True)
        for index, item in enumerate(value)
    )
    if sum(weights) <= 0.0:
        raise ValueError("At least one metric weight must be positive")
    return weights  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class BaselineSuiteConfig:
    """Scientific and operational choices for Phase 10 baseline evaluation.

    Optimizer-dimension and objective-evaluation ceilings are operational guardrails. They
    do not enter the scientific suite identity and fail rather than silently truncating.
    """

    schema_version: str = BASELINE_SCHEMA_VERSION
    enabled_baselines: tuple[str, ...] = BASELINE_NAMES
    random_seed: int = 2026
    random_include_no_op: bool = False
    random_allow_oracle: bool = False
    loss_only_allow_oracle: bool = False
    metric_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_abs_angle: float = math.pi
    improvement_atol: float = 1e-12
    spsa_iterations: int = 12
    spsa_a: float = 0.2
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    cobyla_maxiter: int = 80
    cobyla_initial_step: float = 0.2
    cobyla_tolerance: float = 1e-9
    transpiler_optimization_level: int = 3
    max_optimizer_dimensions: int = 128
    max_objective_evaluations: int = 20000

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be nonblank text")
        schema = self.schema_version.strip()
        if schema != BASELINE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported baseline schema {schema!r}; expected "
                f"{BASELINE_SCHEMA_VERSION!r}"
            )
        enabled = _require_enabled(self.enabled_baselines)
        seed = _require_int(self.random_seed, "random_seed", nonnegative=True)
        random_include_no_op = _require_bool(
            self.random_include_no_op, "random_include_no_op"
        )
        random_allow_oracle = _require_bool(
            self.random_allow_oracle, "random_allow_oracle"
        )
        loss_only_allow_oracle = _require_bool(
            self.loss_only_allow_oracle, "loss_only_allow_oracle"
        )
        weights = _require_weights(self.metric_weights)
        max_abs_angle = _require_float(
            self.max_abs_angle, "max_abs_angle", positive=True
        )
        improvement_atol = _require_float(
            self.improvement_atol, "improvement_atol", nonnegative=True
        )
        spsa_iterations = _require_int(
            self.spsa_iterations, "spsa_iterations", positive=True
        )
        spsa_a = _require_float(self.spsa_a, "spsa_a", positive=True)
        spsa_c = _require_float(self.spsa_c, "spsa_c", positive=True)
        spsa_alpha = _require_float(self.spsa_alpha, "spsa_alpha", positive=True)
        spsa_gamma = _require_float(self.spsa_gamma, "spsa_gamma", positive=True)
        cobyla_maxiter = _require_int(
            self.cobyla_maxiter, "cobyla_maxiter", positive=True
        )
        cobyla_initial_step = _require_float(
            self.cobyla_initial_step, "cobyla_initial_step", positive=True
        )
        cobyla_tolerance = _require_float(
            self.cobyla_tolerance, "cobyla_tolerance", positive=True
        )
        level = _require_int(
            self.transpiler_optimization_level,
            "transpiler_optimization_level",
            nonnegative=True,
        )
        if level > 3:
            raise ValueError("transpiler_optimization_level must be in [0, 3]")
        max_dimensions = _require_int(
            self.max_optimizer_dimensions,
            "max_optimizer_dimensions",
            positive=True,
        )
        max_evaluations = _require_int(
            self.max_objective_evaluations,
            "max_objective_evaluations",
            positive=True,
        )

        object.__setattr__(self, "schema_version", schema)
        object.__setattr__(self, "enabled_baselines", enabled)
        object.__setattr__(self, "random_seed", seed)
        object.__setattr__(self, "random_include_no_op", random_include_no_op)
        object.__setattr__(self, "random_allow_oracle", random_allow_oracle)
        object.__setattr__(self, "loss_only_allow_oracle", loss_only_allow_oracle)
        object.__setattr__(self, "metric_weights", weights)
        object.__setattr__(self, "max_abs_angle", max_abs_angle)
        object.__setattr__(self, "improvement_atol", improvement_atol)
        object.__setattr__(self, "spsa_iterations", spsa_iterations)
        object.__setattr__(self, "spsa_a", spsa_a)
        object.__setattr__(self, "spsa_c", spsa_c)
        object.__setattr__(self, "spsa_alpha", spsa_alpha)
        object.__setattr__(self, "spsa_gamma", spsa_gamma)
        object.__setattr__(self, "cobyla_maxiter", cobyla_maxiter)
        object.__setattr__(self, "cobyla_initial_step", cobyla_initial_step)
        object.__setattr__(self, "cobyla_tolerance", cobyla_tolerance)
        object.__setattr__(self, "transpiler_optimization_level", level)
        object.__setattr__(self, "max_optimizer_dimensions", max_dimensions)
        object.__setattr__(self, "max_objective_evaluations", max_evaluations)
        json.dumps(baseline_config_to_dict(self), sort_keys=True, allow_nan=False)


def baseline_config_to_dict(config: BaselineSuiteConfig) -> dict[str, Any]:
    if not isinstance(config, BaselineSuiteConfig):
        raise TypeError("config must be BaselineSuiteConfig")
    payload = asdict(config)
    payload["enabled_baselines"] = list(config.enabled_baselines)
    payload["metric_weights"] = list(config.metric_weights)
    return payload


def baseline_config_from_dict(payload: Mapping[str, Any]) -> BaselineSuiteConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("baseline config payload must be a mapping")
    allowed = set(BaselineSuiteConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown baseline config fields: {sorted(extra)}")
    return BaselineSuiteConfig(**dict(payload))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite baseline config constant: {value}")


def load_baseline_config(path: str | Path) -> BaselineSuiteConfig:
    target = Path(path)
    text = target.read_text()
    if target.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text, parse_constant=_reject_json_constant)
    if not isinstance(payload, Mapping):
        raise TypeError("baseline config document must contain a mapping")
    return baseline_config_from_dict(payload)


def save_baseline_config(config: BaselineSuiteConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            baseline_config_to_dict(config),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return target


__all__ = [
    "BaselineSuiteConfig",
    "baseline_config_from_dict",
    "baseline_config_to_dict",
    "load_baseline_config",
    "save_baseline_config",
]
