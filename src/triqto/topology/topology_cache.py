"""Explicit non-global cache utilities for repeated topology computations."""
from __future__ import annotations

from .models import TopologyCache


def make_topology_cache() -> TopologyCache:
    """Return a fresh cache instance; Phase 11 never uses global mutable cache state."""
    return TopologyCache()


__all__ = ["TopologyCache", "make_topology_cache"]
