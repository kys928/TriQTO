"""Structured in-memory result records for TriQTO Born metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BornMetricResult:
    """A single Born-probability metric value and its interpretation metadata."""

    metric_name: str
    metric_family: str
    value: float
    lower_is_better: bool
    symmetric: bool
    bounded: bool
    value_range: tuple[float | None, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BornMetricBundle:
    """A collection of Born metrics computed over a shared aligned support."""

    metric_family: str
    metrics: dict[str, BornMetricResult]
    support: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
