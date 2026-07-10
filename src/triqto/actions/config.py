"""Strict configuration for deterministic Phase 9 action generation and ranking."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from .constants import ACTION_SCHEMA_VERSION


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer and not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_finite_float(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON numeric value and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _require_magnitudes(value: Any, max_abs_angle: float) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("candidate_magnitudes must be a sequence of JSON numeric values")
    magnitudes: list[float] = []
    for index, item in enumerate(value):
        magnitude = _require_finite_float(
            item,
            f"candidate_magnitudes[{index}]",
            positive=True,
        )
        if magnitude > max_abs_angle:
            raise ValueError(
                "candidate_magnitudes entries must not exceed max_abs_angle"
            )
        magnitudes.append(magnitude)
    if not magnitudes:
        raise ValueError("candidate_magnitudes must not be empty")
    if len(set(magnitudes)) != len(magnitudes):
        raise ValueError("candidate_magnitudes must be unique")
    if magnitudes != sorted(magnitudes):
        raise ValueError("candidate_magnitudes must be sorted in ascending order")
    return tuple(magnitudes)


@dataclass(frozen=True, slots=True)
class ActionEngineConfig:
    """Configuration for Phase 9 deterministic candidate generation and validation.

    Candidate magnitudes and reward weights define the scientific action universe. The
    maximum candidate/edit counts are operational guardrails and do not enter action IDs.
    """

    schema_version: str = ACTION_SCHEMA_VERSION
    candidate_magnitudes: tuple[float, ...] = (0.05, 0.1, 0.2)
    include_no_op: bool = True
    include_blind_candidates: bool = True
    include_oracle_inverse: bool = True
    observed_edges_only: bool = True
    max_candidates_per_sample: int = 4096
    max_edits_per_action: int = 1024
    max_abs_angle: float = math.pi
    reward_total_variation_weight: float = 1.0
    reward_jensen_shannon_weight: float = 1.0
    reward_hellinger_weight: float = 1.0
    depth_penalty_weight: float = 1e-3
    gate_penalty_weight: float = 1e-3
    edit_penalty_weight: float = 1e-4
    risk_penalty_weight: float = 5e-2
    improvement_atol: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a nonblank string")
        schema_version = self.schema_version.strip()
        if schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Phase 9 action schema {schema_version!r}; "
                f"expected {ACTION_SCHEMA_VERSION!r}"
            )

        max_abs_angle = _require_finite_float(
            self.max_abs_angle,
            "max_abs_angle",
            positive=True,
        )
        magnitudes = _require_magnitudes(self.candidate_magnitudes, max_abs_angle)
        include_no_op = _require_bool(self.include_no_op, "include_no_op")
        include_blind = _require_bool(
            self.include_blind_candidates,
            "include_blind_candidates",
        )
        include_oracle = _require_bool(
            self.include_oracle_inverse,
            "include_oracle_inverse",
        )
        observed_edges_only = _require_bool(
            self.observed_edges_only,
            "observed_edges_only",
        )
        if not observed_edges_only:
            raise ValueError(
                "Phase 9 v1 supports observed_edges_only=True; physical coupling "
                "or all-to-all entangling proposals are not implemented"
            )
        if not any((include_no_op, include_blind, include_oracle)):
            raise ValueError("At least one candidate source must be enabled")

        max_candidates = _require_positive_int(
            self.max_candidates_per_sample,
            "max_candidates_per_sample",
        )
        max_edits = _require_positive_int(
            self.max_edits_per_action,
            "max_edits_per_action",
        )

        weight_fields = (
            "reward_total_variation_weight",
            "reward_jensen_shannon_weight",
            "reward_hellinger_weight",
            "depth_penalty_weight",
            "gate_penalty_weight",
            "edit_penalty_weight",
            "risk_penalty_weight",
            "improvement_atol",
        )
        normalized: dict[str, float] = {}
        for name in weight_fields:
            normalized[name] = _require_finite_float(
                getattr(self, name),
                name,
                nonnegative=True,
            )
        if (
            normalized["reward_total_variation_weight"]
            + normalized["reward_jensen_shannon_weight"]
            + normalized["reward_hellinger_weight"]
            <= 0.0
        ):
            raise ValueError("At least one primary reward metric weight must be positive")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "candidate_magnitudes", magnitudes)
        object.__setattr__(self, "include_no_op", include_no_op)
        object.__setattr__(self, "include_blind_candidates", include_blind)
        object.__setattr__(self, "include_oracle_inverse", include_oracle)
        object.__setattr__(self, "observed_edges_only", observed_edges_only)
        object.__setattr__(self, "max_candidates_per_sample", max_candidates)
        object.__setattr__(self, "max_edits_per_action", max_edits)
        object.__setattr__(self, "max_abs_angle", max_abs_angle)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

        json.dumps(action_config_to_dict(self), sort_keys=True, allow_nan=False)


def action_config_to_dict(config: ActionEngineConfig) -> dict[str, Any]:
    """Return the complete strict JSON-compatible configuration payload."""
    if not isinstance(config, ActionEngineConfig):
        raise TypeError("config must be ActionEngineConfig")
    payload = asdict(config)
    payload["candidate_magnitudes"] = list(config.candidate_magnitudes)
    return payload


def action_config_from_dict(payload: Mapping[str, Any]) -> ActionEngineConfig:
    """Construct a config while rejecting unknown fields and implicit coercion."""
    if not isinstance(payload, Mapping):
        raise TypeError("action config payload must be a mapping")
    allowed = set(ActionEngineConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown action config fields: {sorted(extra)}")
    return ActionEngineConfig(**dict(payload))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite JSON constant in action config: {value}")


def load_action_config(path: str | Path) -> ActionEngineConfig:
    """Load strict JSON action configuration."""
    try:
        payload = json.loads(
            Path(path).read_text(),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed action config JSON: {path}") from exc
    return action_config_from_dict(payload)


def save_action_config(config: ActionEngineConfig, path: str | Path) -> Path:
    """Persist the exact action configuration used by a conversion."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            action_config_to_dict(config),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return target


__all__ = [
    "ActionEngineConfig",
    "action_config_from_dict",
    "action_config_to_dict",
    "load_action_config",
    "save_action_config",
]
