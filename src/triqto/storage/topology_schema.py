"""Typed Phase 11 topology-group manifest record."""
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


@dataclass(slots=True)
class TopologyGroupRecordV1(ManifestRecordMixin):
    """Manifest row for one aligned tri-manifold persistent-homology audit group."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "topology_group_id",
        "topology_audit_id",
        "group_kind",
        "group_key",
        "artifact_ref",
        "content_hash",
    )

    topology_group_id: str
    topology_audit_id: str
    group_kind: str
    group_key: str
    point_count: int
    homology_dimensions: list[int]
    manifolds: list[str]
    artifact_ref: str
    content_hash: str
    hilbert_available: bool
    latent_available: bool
    topology_feature_dim: int
    alignment_feature_dim: int
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        _nonnegative_int(self.point_count, "point_count", positive=True)
        _nonnegative_int(self.topology_feature_dim, "topology_feature_dim")
        _nonnegative_int(self.alignment_feature_dim, "alignment_feature_dim")
        if not isinstance(self.homology_dimensions, list) or not self.homology_dimensions:
            raise ValueError("homology_dimensions must be a nonempty list")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.homology_dimensions
        ):
            raise TypeError("homology_dimensions must contain nonnegative integers")
        if self.homology_dimensions != sorted(set(self.homology_dimensions)):
            raise ValueError("homology_dimensions must be sorted and unique")
        if not isinstance(self.manifolds, list) or not self.manifolds:
            raise ValueError("manifolds must be a nonempty list")
        if any(not isinstance(value, str) or not value for value in self.manifolds):
            raise TypeError("manifolds must contain nonblank strings")
        if len(set(self.manifolds)) != len(self.manifolds):
            raise ValueError("manifolds must be unique")
        if not isinstance(self.hilbert_available, bool):
            raise TypeError("hilbert_available must be bool")
        if not isinstance(self.latent_available, bool):
            raise TypeError("latent_available must be bool")
        if self.latent_available:
            raise ValueError("Phase 11 cannot claim latent topology before a model exists")
        if self.hilbert_available != ("hilbert" in self.manifolds):
            raise ValueError("hilbert_available must match manifolds")
        _safe_ref(self.artifact_ref, "artifact_ref")
        _hash(self.content_hash, "content_hash")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


__all__ = ["TopologyGroupRecordV1"]
