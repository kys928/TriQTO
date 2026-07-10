"""Typed Phase 10 baseline-result manifest record."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar

from .schema import JsonMap, ManifestRecordMixin

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _safe_ref(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a nonempty relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed == PurePosixPath("."):
        raise ValueError(f"{name} must be relative")
    if any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{name} contains traversal or is not normalized")
    if parsed.as_posix() != value:
        raise ValueError(f"{name} must be normalized POSIX text")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must have format sha256:<64 lowercase hex>")
    return value


@dataclass(slots=True)
class BaselineResultRecord(ManifestRecordMixin):
    """Manifest row for one Phase 10 baseline/sample comparison."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "baseline_result_id",
        "baseline_suite_id",
        "sample_id",
        "graph_pair_id",
        "baseline_name",
        "source_circuit_id",
        "clean_target_run_id",
        "artifact_ref",
        "content_hash",
    )

    baseline_result_id: str
    baseline_suite_id: str
    sample_id: str
    graph_pair_id: str
    baseline_name: str
    source_circuit_id: str
    clean_target_run_id: str
    selected_action_id: str | None
    artifact_ref: str
    content_hash: str
    objective_before: float
    objective_after: float
    objective_improvement: float
    success: bool
    evaluations: int
    iterations: int
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        if self.selected_action_id is not None:
            _nonblank(self.selected_action_id, "selected_action_id")
        _safe_ref(self.artifact_ref, "artifact_ref")
        _hash(self.content_hash, "content_hash")
        before = _finite(self.objective_before, "objective_before")
        after = _finite(self.objective_after, "objective_after")
        improvement = _finite(self.objective_improvement, "objective_improvement")
        if before < 0.0 or after < 0.0:
            raise ValueError("baseline objectives must be nonnegative")
        if not math.isclose(improvement, before - after, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("objective_improvement must equal objective_before-after")
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        _nonnegative_int(self.evaluations, "evaluations")
        _nonnegative_int(self.iterations, "iterations")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


__all__ = ["BaselineResultRecord"]
