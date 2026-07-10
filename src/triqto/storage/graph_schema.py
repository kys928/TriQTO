"""Phase 8 graph manifest record schemas.

These records are kept in a dedicated module so the Phase 1--7 storage schema remains
backward compatible while Phase 8 can evolve its circuit-level provenance contract.
"""
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
        raise ValueError(f"{name} must be nonblank")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a nonnegative integer and not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _safe_ref(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a nonempty relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed == PurePosixPath("."):
        raise ValueError(f"{name} must be a relative file path")
    if any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{name} must be normalized and must not escape the root")
    if parsed.as_posix() != value:
        raise ValueError(f"{name} must be normalized POSIX path")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must have format sha256:<64 lowercase hex characters>"
        )
    return value


@dataclass(slots=True)
class GraphRecord(ManifestRecordMixin):
    """Manifest row for one circuit/run-level Phase 8 graph artifact."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "graph_id",
        "circuit_id",
        "source_run_id",
        "role",
        "family",
        "graph_schema_version",
        "graph_ref",
        "content_hash",
    )

    graph_id: str
    circuit_id: str
    source_run_id: str
    role: str
    family: str
    graph_schema_version: str
    graph_ref: str
    content_hash: str
    n_nodes: int
    n_edges: int
    n_gate_events: int
    node_feature_dim: int
    edge_feature_dim: int
    gate_feature_dim: int
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        if self.role not in {"clean", "distorted"}:
            raise ValueError("GraphRecord.role must be clean or distorted")
        for name in (
            "n_nodes",
            "n_edges",
            "n_gate_events",
            "node_feature_dim",
            "edge_feature_dim",
            "gate_feature_dim",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.n_nodes == 0:
            raise ValueError("GraphRecord.n_nodes must be positive")
        _safe_ref(self.graph_ref, "graph_ref")
        _hash(self.content_hash, "content_hash")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("GraphRecord.metadata must be a mapping")


@dataclass(slots=True)
class GraphPairRecord(ManifestRecordMixin):
    """Manifest row linking one Phase 7 sample to clean/distorted graph artifacts."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "graph_pair_id",
        "sample_id",
        "clean_graph_id",
        "distorted_graph_id",
        "distortion_id",
        "metric_id",
        "pair_ref",
        "content_hash",
    )

    graph_pair_id: str
    sample_id: str
    clean_graph_id: str
    distorted_graph_id: str
    distortion_id: str
    metric_id: str
    pair_ref: str
    content_hash: str
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        for name in self.required_fields:
            _nonblank(getattr(self, name), name)
        _safe_ref(self.pair_ref, "pair_ref")
        _hash(self.content_hash, "content_hash")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("GraphPairRecord.metadata must be a mapping")


__all__ = ["GraphPairRecord", "GraphRecord"]
