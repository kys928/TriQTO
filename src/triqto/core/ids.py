"""Deterministic identifier helpers for TriQTO records.

Phase 2 will replace these lightweight stubs with canonical hashing contracts.
"""
from __future__ import annotations


def make_circuit_id(payload: object) -> str:
    """Return a future deterministic circuit identifier placeholder."""
    raise NotImplementedError("Phase 2 will define deterministic circuit IDs.")


def make_run_id(payload: object) -> str:
    """Return a future deterministic simulation or experiment run ID placeholder."""
    raise NotImplementedError("Phase 2 will define deterministic run IDs.")


def make_sample_id(payload: object) -> str:
    """Return a future deterministic data-lake sample ID placeholder."""
    raise NotImplementedError("Phase 2 will define deterministic sample IDs.")


def make_topology_group_id(payload: object) -> str:
    """Return a future deterministic topology group ID placeholder."""
    raise NotImplementedError("Phase 2 will define deterministic topology group IDs.")
