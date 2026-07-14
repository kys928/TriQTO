"""Typed Phase 15 item, aggregate, and baseline-comparison manifest rows."""
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


def _optional_text(value: Any, name: str) -> None:
    if value is not None:
        _text(value, name)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


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


def _restore_parquet_numeric_map(row: JsonMap, name: str) -> None:
    """Remove null struct members introduced for heterogeneous metric maps."""
    value = row.get(name)
    if isinstance(value, Mapping):
        row[name] = {
            key: item
            for key, item in value.items()
            if item is not None
            and not (isinstance(item, float) and math.isnan(item))
        }


@dataclass(slots=True)
class EvaluationItemRecordV1(ManifestRecordMixin):
    required_fields: ClassVar[tuple[str, ...]] = (
        "evaluation_item_id",
        "evaluation_run_id",
        "view_item_id",
        "entity_id",
        "task",
        "split",
        "ablation",
        "artifact_ref",
        "content_hash",
    )

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
    distortion_type: str | None
    backend_id: str | None
    predicted_action_id: str | None
    target_action_id: str | None
    target_action_rank: int | None
    metrics: JsonMap = field(default_factory=dict)
    calibration: JsonMap = field(default_factory=dict)
    artifact_ref: str = ""
    content_hash: str = ""
    metadata: JsonMap = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: JsonMap) -> "EvaluationItemRecordV1":
        """Restore the optional integer rank after Parquet null promotion.

        Pandas promotes a column containing both integer ranks and nulls to
        floating point on Parquet readback.  Accept only exactly integral,
        finite values at this storage boundary; validation remains strict for
        every other value.
        """
        restored = dict(row)
        _restore_parquet_numeric_map(restored, "metrics")
        _restore_parquet_numeric_map(restored, "calibration")
        rank = restored.get("target_action_rank")
        if (
            isinstance(rank, float)
            and math.isfinite(rank)
            and rank.is_integer()
        ):
            restored["target_action_rank"] = int(rank)
        return ManifestRecordMixin.from_dict.__func__(cls, restored)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _text(getattr(self, name), name)
        if self.split != "test":
            raise ValueError("Phase 15 item rows must be held-out test rows")
        _optional_text(self.family, "family")
        _optional_text(self.distortion_id, "distortion_id")
        _optional_text(self.distortion_type, "distortion_type")
        _optional_text(self.backend_id, "backend_id")
        _optional_text(self.predicted_action_id, "predicted_action_id")
        _optional_text(self.target_action_id, "target_action_id")
        _nonnegative_int(self.n_qubits, "n_qubits")
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.target_action_rank is not None:
            _nonnegative_int(self.target_action_rank, "target_action_rank")
        _numeric_map(self.metrics, "metrics")
        _numeric_map(self.calibration, "calibration")
        _safe_ref(self.artifact_ref, "artifact_ref")
        _hash(self.content_hash, "content_hash")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@dataclass(slots=True)
class EvaluationAggregateRecordV1(ManifestRecordMixin):
    required_fields: ClassVar[tuple[str, ...]] = (
        "evaluation_aggregate_id",
        "evaluation_run_id",
        "task",
        "ablation",
        "group_dimension",
        "group_value",
    )

    evaluation_aggregate_id: str
    evaluation_run_id: str
    task: str
    ablation: str
    group_dimension: str
    group_value: str
    item_count: int
    metrics: JsonMap = field(default_factory=dict)
    calibration: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: JsonMap) -> "EvaluationAggregateRecordV1":
        restored = dict(row)
        _restore_parquet_numeric_map(restored, "metrics")
        _restore_parquet_numeric_map(restored, "calibration")
        return ManifestRecordMixin.from_dict.__func__(cls, restored)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _text(getattr(self, name), name)
        _nonnegative_int(self.item_count, "item_count")
        if self.item_count <= 0:
            raise ValueError("item_count must be positive")
        _numeric_map(self.metrics, "metrics")
        _numeric_map(self.calibration, "calibration")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@dataclass(slots=True)
class EvaluationBaselineRecordV1(ManifestRecordMixin):
    required_fields: ClassVar[tuple[str, ...]] = (
        "evaluation_baseline_id",
        "evaluation_run_id",
        "sample_id",
        "task",
        "baseline_name",
        "learned_action_id",
    )

    evaluation_baseline_id: str
    evaluation_run_id: str
    sample_id: str
    task: str
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
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _text(getattr(self, name), name)
        _optional_text(self.baseline_action_id, "baseline_action_id")
        before = _finite(self.objective_before, "objective_before")
        learned = _finite(self.learned_objective_after, "learned_objective_after")
        baseline = _finite(self.baseline_objective_after, "baseline_objective_after")
        delta = _finite(self.learned_minus_baseline, "learned_minus_baseline")
        if min(before, learned, baseline) < 0.0:
            raise ValueError("baseline objectives must be nonnegative")
        if not math.isclose(delta, learned - baseline, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("learned_minus_baseline must equal learned-baseline")
        for name in ("learned_success", "baseline_success", "baseline_privileged"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


__all__ = [
    "EvaluationAggregateRecordV1",
    "EvaluationBaselineRecordV1",
    "EvaluationItemRecordV1",
]
