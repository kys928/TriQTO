"""Read-only validation of the Phase 7/8/9 source chain used by topology."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from triqto.baselines import load_baseline_sources
from triqto.graph import snapshot_managed_files


def load_topology_sources(
    phase7_root: str | Path,
    graph_root: str | Path,
    action_root: str | Path,
) -> Any:
    """Reuse the fully validated Phase 7/8/9 loader established before Phase 11."""
    return load_baseline_sources(phase7_root, graph_root, action_root)


def verify_topology_source_snapshots(sources: Any) -> None:
    """Prove no managed source file changed during Phase 11 work."""
    checks = (
        (
            "Phase 7",
            sources.phase7.source_root,
            sources.phase7.source_snapshot,
        ),
        ("Phase 8", sources.graph.root, sources.graph.snapshot),
        ("Phase 9", sources.action.root, sources.action.snapshot),
    )
    for name, root, expected in checks:
        actual = snapshot_managed_files(
            root,
            tuple(entry.reference for entry in expected.entries),
        )
        if actual != expected:
            raise RuntimeError(f"{name} managed source files changed during Phase 11")


__all__ = ["load_topology_sources", "verify_topology_source_snapshots"]
