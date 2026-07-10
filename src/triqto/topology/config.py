"""Strict configuration for the Phase 11 persistent-homology audit."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    BORN_DISTANCE_NAMES,
    DEFAULT_BETTI_GRID_SIZE,
    DEFAULT_HOMOLOGY_DIMENSIONS,
    DEFAULT_TOP_K_LIFETIMES,
    GROUP_KINDS,
    TOPOLOGY_SCHEMA_VERSION,
)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _require_int(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_float(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _require_group_kinds(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("group_kinds must be a sequence of strings")
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"group_kinds[{index}] must be nonblank text")
        names.append(item.strip())
    if not names:
        raise ValueError("group_kinds must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("group_kinds must be unique")
    unknown = set(names) - set(GROUP_KINDS)
    if unknown:
        raise ValueError(f"Unknown topology group kinds: {sorted(unknown)}")
    expected = tuple(name for name in GROUP_KINDS if name in names)
    if tuple(names) != expected:
        raise ValueError(
            "group_kinds must follow the fixed Phase 11 order: "
            f"{list(GROUP_KINDS)}"
        )
    return tuple(names)


def _require_homology_dimensions(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("homology_dimensions must be a sequence of integers")
    dimensions: list[int] = []
    for index, item in enumerate(value):
        dimension = _require_int(
            item,
            f"homology_dimensions[{index}]",
            nonnegative=True,
        )
        if dimension > 2:
            raise ValueError("Phase 11 supports homology dimensions 0, 1, and optional 2")
        dimensions.append(dimension)
    if not dimensions:
        raise ValueError("homology_dimensions must not be empty")
    if dimensions != sorted(set(dimensions)):
        raise ValueError("homology_dimensions must be sorted and unique")
    if 0 not in dimensions or 1 not in dimensions:
        raise ValueError("H0 and H1 must remain active in Phase 11")
    return tuple(dimensions)


@dataclass(frozen=True, slots=True)
class TopologyAuditConfig:
    """Scientific and operational choices for Phase 11 topology auditing.

    Group/point/statevector ceilings are operational guardrails. They never subsample or
    alter a scientific topology result; exceeding them raises instead.
    """

    schema_version: str = TOPOLOGY_SCHEMA_VERSION
    group_kinds: tuple[str, ...] = GROUP_KINDS
    min_points: int = 3
    homology_dimensions: tuple[int, ...] = DEFAULT_HOMOLOGY_DIMENSIONS
    include_hilbert: bool = True
    born_distance: str = "hellinger"
    normalize_distance_matrices: bool = True
    raw_parameter_weight: float = 0.1
    born_pullback_weight: float = 1.0
    hilbert_pullback_weight: float = 1.0
    betti_grid_size: int = DEFAULT_BETTI_GRID_SIZE
    top_k_lifetimes: int = DEFAULT_TOP_K_LIFETIMES
    max_filtration: float = 1.0
    max_points_per_group: int = 256
    max_groups: int = 4096
    max_statevector_amplitudes: int = 65536

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be nonblank text")
        schema = self.schema_version.strip()
        if schema != TOPOLOGY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported topology schema {schema!r}; expected "
                f"{TOPOLOGY_SCHEMA_VERSION!r}"
            )
        group_kinds = _require_group_kinds(self.group_kinds)
        min_points = _require_int(self.min_points, "min_points", positive=True)
        if min_points < 2:
            raise ValueError("min_points must be at least two")
        dimensions = _require_homology_dimensions(self.homology_dimensions)
        include_hilbert = _require_bool(self.include_hilbert, "include_hilbert")
        if not isinstance(self.born_distance, str):
            raise TypeError("born_distance must be text")
        born_distance = self.born_distance.strip()
        if born_distance not in BORN_DISTANCE_NAMES:
            raise ValueError(
                f"Unsupported born_distance {born_distance!r}; expected one of "
                f"{list(BORN_DISTANCE_NAMES)}"
            )
        normalize = _require_bool(
            self.normalize_distance_matrices,
            "normalize_distance_matrices",
        )
        raw_weight = _require_float(
            self.raw_parameter_weight,
            "raw_parameter_weight",
            nonnegative=True,
        )
        born_weight = _require_float(
            self.born_pullback_weight,
            "born_pullback_weight",
            nonnegative=True,
        )
        hilbert_weight = _require_float(
            self.hilbert_pullback_weight,
            "hilbert_pullback_weight",
            nonnegative=True,
        )
        if raw_weight + born_weight + (hilbert_weight if include_hilbert else 0.0) <= 0.0:
            raise ValueError("At least one active parameter-distance component must be positive")
        betti_grid_size = _require_int(
            self.betti_grid_size,
            "betti_grid_size",
            positive=True,
        )
        if betti_grid_size < 2:
            raise ValueError("betti_grid_size must be at least two")
        top_k = _require_int(
            self.top_k_lifetimes,
            "top_k_lifetimes",
            positive=True,
        )
        max_filtration = _require_float(
            self.max_filtration,
            "max_filtration",
            positive=True,
        )
        max_points = _require_int(
            self.max_points_per_group,
            "max_points_per_group",
            positive=True,
        )
        if max_points < min_points:
            raise ValueError("max_points_per_group must be at least min_points")
        max_groups = _require_int(self.max_groups, "max_groups", positive=True)
        max_statevector = _require_int(
            self.max_statevector_amplitudes,
            "max_statevector_amplitudes",
            positive=True,
        )

        object.__setattr__(self, "schema_version", schema)
        object.__setattr__(self, "group_kinds", group_kinds)
        object.__setattr__(self, "min_points", min_points)
        object.__setattr__(self, "homology_dimensions", dimensions)
        object.__setattr__(self, "include_hilbert", include_hilbert)
        object.__setattr__(self, "born_distance", born_distance)
        object.__setattr__(self, "normalize_distance_matrices", normalize)
        object.__setattr__(self, "raw_parameter_weight", raw_weight)
        object.__setattr__(self, "born_pullback_weight", born_weight)
        object.__setattr__(self, "hilbert_pullback_weight", hilbert_weight)
        object.__setattr__(self, "betti_grid_size", betti_grid_size)
        object.__setattr__(self, "top_k_lifetimes", top_k)
        object.__setattr__(self, "max_filtration", max_filtration)
        object.__setattr__(self, "max_points_per_group", max_points)
        object.__setattr__(self, "max_groups", max_groups)
        object.__setattr__(self, "max_statevector_amplitudes", max_statevector)
        json.dumps(topology_config_to_dict(self), sort_keys=True, allow_nan=False)


def topology_config_to_dict(config: TopologyAuditConfig) -> dict[str, Any]:
    if not isinstance(config, TopologyAuditConfig):
        raise TypeError("config must be TopologyAuditConfig")
    payload = asdict(config)
    payload["group_kinds"] = list(config.group_kinds)
    payload["homology_dimensions"] = list(config.homology_dimensions)
    return payload


def topology_config_from_dict(payload: Mapping[str, Any]) -> TopologyAuditConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("topology config payload must be a mapping")
    allowed = set(TopologyAuditConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown topology config fields: {sorted(extra)}")
    return TopologyAuditConfig(**dict(payload))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite topology config constant: {value}")


def load_topology_config(path: str | Path) -> TopologyAuditConfig:
    target = Path(path)
    text = target.read_text()
    if target.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text, parse_constant=_reject_json_constant)
    if not isinstance(payload, Mapping):
        raise TypeError("topology config document must contain a mapping")
    return topology_config_from_dict(payload)


def save_topology_config(config: TopologyAuditConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            topology_config_to_dict(config),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return target


__all__ = [
    "TopologyAuditConfig",
    "load_topology_config",
    "save_topology_config",
    "topology_config_from_dict",
    "topology_config_to_dict",
]
