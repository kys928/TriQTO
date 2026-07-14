"""Identifiability-aware diagnosis evaluation boundaries.

This module is deliberately small and is used by the Phase 15 evaluator. It
prevents headline diagnosis metrics from silently including
targets that cannot be inferred from the allowed observable evidence.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from triqto.data_generation.identifiability import IDENTIFIABILITY_STATUSES

T = TypeVar("T")


def _value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"evaluation row is missing {name}")
        return record[name]
    if not hasattr(record, name):
        raise ValueError(f"evaluation record is missing {name}")
    return getattr(record, name)


def _metadata(record: Any) -> Mapping[str, Any]:
    value = (
        record.get("metadata", {})
        if isinstance(record, Mapping)
        else getattr(record, "metadata", {})
    )
    if not isinstance(value, Mapping):
        raise TypeError("evaluation record metadata must be a mapping")
    return value


def _validated_status(record: Any) -> tuple[str, str | None, bool]:
    status = _value(record, "identifiability_status")
    reason = _value(record, "identifiability_reason")
    supervised = _value(record, "diagnosis_supervision_mask")
    if status not in IDENTIFIABILITY_STATUSES:
        raise ValueError(f"invalid identifiability_status {status!r}")
    if status == "identifiable":
        if reason is not None:
            raise ValueError("identifiable evaluation rows must not have a reason")
    elif not isinstance(reason, str) or not reason:
        raise ValueError("non-identifiable evaluation rows require a reason")
    if not isinstance(supervised, bool):
        raise TypeError("diagnosis_supervision_mask must be bool")
    if status == "unidentifiable" and supervised:
        if _metadata(record).get("unidentifiable_supervision_override") is not True:
            raise ValueError(
                "supervised unidentifiable evaluation rows require an explicit override"
            )
    return status, reason, supervised


@dataclass(frozen=True, slots=True)
class IdentifiabilityEvaluationReport:
    """Coverage that must accompany diagnosis metrics and rankings."""

    total_count: int
    default_scored_count: int
    default_excluded_count: int
    explicit_override_count: int
    status_counts: dict[str, int]
    reason_counts: dict[str, int]

    @property
    def default_scored_coverage(self) -> float:
        return self.default_scored_count / self.total_count if self.total_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "default_scored_count": self.default_scored_count,
            "default_excluded_count": self.default_excluded_count,
            "explicit_override_count": self.explicit_override_count,
            "default_scored_coverage": self.default_scored_coverage,
            "status_counts": dict(self.status_counts),
            "reason_counts": dict(self.reason_counts),
            "unidentifiable_rows_in_headline_metrics": False,
        }


def filter_diagnosis_evaluation_rows(
    records: Sequence[T] | Iterable[T],
    *,
    include_explicit_unidentifiable_overrides: bool = False,
) -> list[T]:
    """Return only rows eligible for diagnosis metrics under the chosen policy.

    Unidentifiable rows are excluded even when a generation-time override made
    them supervised.  Including those rows in an evaluation requires a second,
    explicit evaluation-time opt-in and should never be used for headline
    results.
    """
    if not isinstance(include_explicit_unidentifiable_overrides, bool):
        raise TypeError("include_explicit_unidentifiable_overrides must be bool")
    selected: list[T] = []
    for record in records:
        status, _, supervised = _validated_status(record)
        if not supervised:
            continue
        if status != "unidentifiable" or include_explicit_unidentifiable_overrides:
            selected.append(record)
    return selected


def build_identifiability_evaluation_report(
    records: Sequence[Any] | Iterable[Any],
) -> IdentifiabilityEvaluationReport:
    """Build mandatory diagnosis-scoring coverage and exclusion counts."""
    rows = list(records)
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    default_scored = 0
    overrides = 0
    for record in rows:
        status, reason, supervised = _validated_status(record)
        statuses[status] += 1
        if reason is not None:
            reasons[reason] += 1
        if supervised and status != "unidentifiable":
            default_scored += 1
        if supervised and status == "unidentifiable":
            overrides += 1
    return IdentifiabilityEvaluationReport(
        total_count=len(rows),
        default_scored_count=default_scored,
        default_excluded_count=len(rows) - default_scored,
        explicit_override_count=overrides,
        status_counts=dict(sorted(statuses.items())),
        reason_counts=dict(sorted(reasons.items())),
    )


__all__ = [
    "IdentifiabilityEvaluationReport",
    "build_identifiability_evaluation_report",
    "filter_diagnosis_evaluation_rows",
]
