"""Strict configuration for deterministic Phase 8 graph conversion."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import GRAPH_SCHEMA_VERSION
from .utils import (
    json_copy,
    require_exact_bool,
    require_nonblank,
    require_positive_int,
    strict_json_load,
    write_strict_json,
)


@dataclass(frozen=True, slots=True)
class GraphConversionConfig:
    schema_version: str = GRAPH_SCHEMA_VERSION
    max_gate_events: int = 100_000
    max_probability_outcomes: int = 1_000_000
    include_supplemental_counts: bool = True
    reject_conditioned_operations: bool = True

    def __post_init__(self) -> None:
        schema_version = require_nonblank(self.schema_version, "schema_version")
        if schema_version != GRAPH_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported graph schema_version {schema_version!r}; "
                f"Phase 8 v1 supports only {GRAPH_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "max_gate_events",
            require_positive_int(self.max_gate_events, "max_gate_events"),
        )
        object.__setattr__(
            self,
            "max_probability_outcomes",
            require_positive_int(
                self.max_probability_outcomes,
                "max_probability_outcomes",
            ),
        )
        object.__setattr__(
            self,
            "include_supplemental_counts",
            require_exact_bool(
                self.include_supplemental_counts,
                "include_supplemental_counts",
            ),
        )
        reject_conditioned = require_exact_bool(
            self.reject_conditioned_operations,
            "reject_conditioned_operations",
        )
        if reject_conditioned is not True:
            raise ValueError(
                "Phase 8 v1 cannot preserve classical conditions; "
                "reject_conditioned_operations must be True"
            )
        object.__setattr__(self, "reject_conditioned_operations", True)
        json_copy(asdict(self))


def graph_config_to_dict(config: GraphConversionConfig) -> dict[str, Any]:
    if not isinstance(config, GraphConversionConfig):
        raise TypeError("config must be GraphConversionConfig")
    return json_copy(asdict(config))


def graph_config_from_dict(payload: Mapping[str, Any]) -> GraphConversionConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("graph config payload must be a mapping")
    allowed = set(GraphConversionConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown graph config fields: {sorted(unknown)}")
    return GraphConversionConfig(**dict(payload))


def load_graph_config(path: str | Path) -> GraphConversionConfig:
    payload = strict_json_load(path)
    return graph_config_from_dict(payload)


def save_graph_config(config: GraphConversionConfig, path: str | Path) -> Path:
    return write_strict_json(path, graph_config_to_dict(config))


__all__ = [
    "GraphConversionConfig",
    "graph_config_from_dict",
    "graph_config_to_dict",
    "load_graph_config",
    "save_graph_config",
]
