"""Baseline comparison identities for Phase 15 evaluation artifacts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from triqto.core.ids import make_deterministic_id


@dataclass(frozen=True, slots=True)
class BaselineComparisonKey:
    run_id: str
    sample_id: str
    baseline_id: str
    task: str
    view_id: str
    execution_mode: str
    ablation_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "sample_id", "baseline_id", "task", "view_id", "execution_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonblank")
        if self.ablation_id is not None and (not isinstance(self.ablation_id, str) or not self.ablation_id.strip()):
            raise ValueError("ablation_id must be nonblank when provided")

    @property
    def comparison_id(self) -> str:
        return make_deterministic_id("baselinecmp", self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "baseline_id": self.baseline_id,
            "task": self.task,
            "view_id": self.view_id,
            "execution_mode": self.execution_mode,
            "ablation_id": self.ablation_id,
            "identity_schema": "triqto.phase15.baseline_comparison.v2",
        }


def comparison_id(*, run_id: str, sample_id: str, baseline_id: str, task: str, view_id: str, execution_mode: str, ablation_id: str | None = None) -> str:
    return BaselineComparisonKey(run_id, sample_id, baseline_id, task, view_id, execution_mode, ablation_id).comparison_id


def validate_unique_comparisons(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate comparison IDs unless records are byte-for-byte semantic duplicates."""
    seen: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        cid = record.get("comparison_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("comparison record missing comparison_id")
        previous = seen.get(cid)
        if previous is None:
            seen[cid] = record
        elif previous != record:
            raise ValueError(f"conflicting duplicate comparison_id: {cid}")
        else:
            raise ValueError(f"duplicate comparison_id is not allowed: {cid}")


def build_comparison_records(*, run_id: str, sample_id: str, baselines: Sequence[str], tasks: Sequence[str], view_id: str, execution_mode: str, ablation_id: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for baseline in baselines:
        for task in tasks:
            key = BaselineComparisonKey(run_id, sample_id, baseline, task, view_id, execution_mode, ablation_id)
            records.append({"comparison_id": key.comparison_id, **key.to_payload()})
    validate_unique_comparisons(records)
    return records


__all__ = ["BaselineComparisonKey", "build_comparison_records", "comparison_id", "validate_unique_comparisons"]
