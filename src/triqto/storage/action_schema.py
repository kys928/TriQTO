"""Typed Phase 9 action-candidate and rollout manifest records."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar

from .schema import JsonMap, ManifestRecordMixin

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_GENERATION_SOURCES = {
    "blind_physics_prior",
    "no_op",
    "oracle_inverse",
}


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer and not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a nonnegative integer and not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _safe_ref(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a nonempty relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed == PurePosixPath("."):
        raise ValueError(f"{name} must be a relative file path")
    if any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{name} must be normalized and remain inside the dataset")
    if parsed.as_posix() != value:
        raise ValueError(f"{name} must be a normalized POSIX path")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must have format sha256:<64 lowercase hex characters>")
    return value


@dataclass(slots=True)
class ActionCandidateRecordV1(ManifestRecordMixin):
    """Manifest row for one validated Phase 9 candidate circuit edit."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "action_id",
        "sample_id",
        "graph_pair_id",
        "source_circuit_id",
        "source_run_id",
        "distortion_id",
        "candidate_circuit_id",
        "action_ref",
        "circuit_ref",
        "content_hash",
        "circuit_hash",
    )

    action_id: str
    sample_id: str
    graph_pair_id: str
    source_circuit_id: str
    source_run_id: str
    distortion_id: str
    candidate_circuit_id: str
    generation_sources: list[str]
    action_ref: str
    circuit_ref: str
    content_hash: str
    circuit_hash: str
    edit_count: int
    validity_mask: bool
    risk_score: float
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        if not isinstance(self.generation_sources, list) or not self.generation_sources:
            raise TypeError("generation_sources must be a nonempty list")
        if any(
            not isinstance(value, str) or not value
            for value in self.generation_sources
        ):
            raise TypeError("generation_sources entries must be nonblank strings")
        if self.generation_sources != sorted(self.generation_sources):
            raise ValueError("generation_sources must be sorted")
        if len(set(self.generation_sources)) != len(self.generation_sources):
            raise ValueError("generation_sources must be unique")
        unknown = set(self.generation_sources) - _ALLOWED_GENERATION_SOURCES
        if unknown:
            raise ValueError(f"Unknown generation_sources: {sorted(unknown)}")
        _safe_ref(self.action_ref, "action_ref")
        _safe_ref(self.circuit_ref, "circuit_ref")
        _hash(self.content_hash, "content_hash")
        _hash(self.circuit_hash, "circuit_hash")
        _nonnegative_int(self.edit_count, "edit_count")
        if not isinstance(self.validity_mask, bool):
            raise TypeError("validity_mask must be bool")
        risk = _finite(self.risk_score, "risk_score")
        if risk < 0.0 or risk > 1.0:
            raise ValueError("risk_score must be in [0, 1]")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@dataclass(slots=True)
class ActionRolloutRecord(ManifestRecordMixin):
    """Manifest row for one exact ideal-simulator action validation rollout."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "rollout_id",
        "action_id",
        "sample_id",
        "graph_pair_id",
        "candidate_circuit_id",
        "clean_target_run_id",
        "scientific_config_id",
        "rollout_ref",
        "content_hash",
    )

    rollout_id: str
    action_id: str
    sample_id: str
    graph_pair_id: str
    candidate_circuit_id: str
    clean_target_run_id: str
    scientific_config_id: str
    rollout_ref: str
    content_hash: str
    rank: int
    reward: float
    risk_score: float
    dominates_baseline: bool
    primary_metric_nonworsening: bool
    selected: bool
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        _safe_ref(self.rollout_ref, "rollout_ref")
        _hash(self.content_hash, "content_hash")
        _positive_int(self.rank, "rank")
        _finite(self.reward, "reward")
        risk = _finite(self.risk_score, "risk_score")
        if risk < 0.0 or risk > 1.0:
            raise ValueError("risk_score must be in [0, 1]")
        for name in (
            "dominates_baseline",
            "primary_metric_nonworsening",
            "selected",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


__all__ = ["ActionCandidateRecordV1", "ActionRolloutRecord"]
