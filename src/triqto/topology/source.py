"""Read-only validation of the Phase 7/8/9 source chain used by topology."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from triqto.actions import load_action_engine_sources, load_lazy_action_dataset
from triqto.baselines.models import BaselineSources
from triqto.graph import snapshot_managed_files


def load_topology_sources(
    phase7_root: str | Path,
    graph_root: str | Path,
    action_root: str | Path,
) -> Any:
    """Load Phase 7/8 normally and Phase 9 through a durable per-sample lazy index."""
    earlier = load_action_engine_sources(phase7_root, graph_root)
    action_path = Path(action_root)
    action = load_lazy_action_dataset(
        action_path,
        phase7=earlier.phase7,
        graph=earlier.graph,
        checkpoint_root=action_path.parent / ".phase9-lazy-index",
        label="Phase 11",
    )
    return BaselineSources(
        phase7=earlier.phase7,
        graph=earlier.graph,
        action=action,
    )


def verify_topology_source_snapshots(sources: Any) -> None:
    """Prove no source control file or accessed artifact inventory changed."""
    checks = (
        (
            "Phase 7",
            sources.phase7.source_root,
            sources.phase7.source_snapshot,
        ),
        ("Phase 8", sources.graph.root, sources.graph.snapshot),
    )
    for name, root, expected in checks:
        actual = snapshot_managed_files(
            root,
            tuple(entry.reference for entry in expected.entries),
        )
        if actual != expected:
            raise RuntimeError(f"{name} managed source files changed during Phase 11")
    if getattr(sources.action, "is_lazy", False):
        sources.action.verify_source()
    else:
        actual = snapshot_managed_files(
            sources.action.root,
            tuple(entry.reference for entry in sources.action.snapshot.entries),
        )
        if actual != sources.action.snapshot:
            raise RuntimeError("Phase 9 managed source files changed during Phase 11")


__all__ = ["load_topology_sources", "verify_topology_source_snapshots"]
