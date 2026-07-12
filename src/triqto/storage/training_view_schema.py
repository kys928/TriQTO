"""Typed Phase 12 training-view definition and item manifest records."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar

from .schema import JsonMap, ManifestRecordMixin

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value


def _nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
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


def _string_list(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{name} must contain nonblank strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must be unique")
    return value


@dataclass(slots=True)
class TrainingViewDefinitionRecordV1(ManifestRecordMixin):
    """One task-view definition and its split/item coverage."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "training_view_id",
        "training_view_dataset_id",
        "task",
        "mask_policy",
        "split_policy",
    )

    training_view_id: str
    training_view_dataset_id: str
    task: str
    input_groups: list[str]
    target_groups: list[str]
    mask_policy: str
    split_policy: str
    item_count: int
    split_counts: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        _string_list(self.input_groups, "input_groups")
        _string_list(self.target_groups, "target_groups")
        _nonnegative_int(self.item_count, "item_count")
        if not isinstance(self.split_counts, Mapping):
            raise TypeError("split_counts must be a mapping")
        total = 0
        for split, count in self.split_counts.items():
            _nonblank(split, "split_counts key")
            total += _nonnegative_int(count, f"split_counts[{split}]")
        if total != self.item_count:
            raise ValueError("split_counts must sum to item_count")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@dataclass(slots=True)
class TrainingViewItemRecordV1(ManifestRecordMixin):
    """One materialized task-specific view item."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "view_item_id",
        "training_view_id",
        "training_view_dataset_id",
        "task",
        "split",
        "split_group_id",
        "entity_id",
        "artifact_ref",
        "content_hash",
    )

    view_item_id: str
    training_view_id: str
    training_view_dataset_id: str
    task: str
    split: str
    split_group_id: str
    entity_id: str
    input_groups: list[str]
    target_groups: list[str]
    artifact_ref: str
    content_hash: str
    hilbert_available_mask: bool
    topology_available_mask: bool
    privileged_target_available_mask: bool
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        _string_list(self.input_groups, "input_groups")
        _string_list(self.target_groups, "target_groups")
        _safe_ref(self.artifact_ref, "artifact_ref")
        _hash(self.content_hash, "content_hash")
        for name in (
            "hilbert_available_mask",
            "topology_available_mask",
            "privileged_target_available_mask",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


__all__ = ["TrainingViewDefinitionRecordV1", "TrainingViewItemRecordV1"]
