"""Read-only validation of the Phase 7/8/9 source chain used by topology."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any
import uuid

from triqto.actions import load_action_engine_sources, load_lazy_action_dataset
from triqto.baselines.models import BaselineSources
from triqto.graph import snapshot_managed_files
from triqto.graph.utils import write_strict_json


def _write_status(root: Path, stage: str, **details: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "progress.json"
    temporary = root / f".progress.tmp-{uuid.uuid4().hex}.json"
    payload = {
        "schema": "triqto.phase15_6.phase11_progress.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **details,
    }
    try:
        write_strict_json(temporary, payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _log(message: str) -> None:
    print(f"[Phase 11][startup] {message}", flush=True)


def load_topology_sources(
    phase7_root: str | Path,
    graph_root: str | Path,
    action_root: str | Path,
) -> Any:
    """Load Phase 7/8 normally and Phase 9 through a durable per-sample lazy index."""
    action_path = Path(action_root)
    status_root = action_path.parent / ".phase11-checkpoints"

    started = time.monotonic()
    _write_status(
        status_root,
        "loading_phase7_phase8",
        detail="validating and materializing completed Phase 7/8 source datasets",
    )
    _log("loading Phase 7 and Phase 8 validated sources")
    earlier = load_action_engine_sources(phase7_root, graph_root)
    _log(
        "Phase 7/8 sources ready | "
        f"samples={len(earlier.phase7.samples):,} | "
        f"graphs={len(earlier.graph.graph_records):,} | "
        f"pairs={len(earlier.graph.pair_records):,} | "
        f"elapsed={(time.monotonic()-started)/60.0:.2f}m"
    )

    _write_status(
        status_root,
        "opening_lazy_phase9_index",
        detail="validating Phase 9 control plane and reusing/building bounded SQLite index",
    )
    _log("opening the durable Phase 9 lazy index; full artifact hydration is disabled")
    action = load_lazy_action_dataset(
        action_path,
        phase7=earlier.phase7,
        graph=earlier.graph,
        checkpoint_root=action_path.parent / ".phase9-lazy-index",
        label="Phase 11",
    )
    _write_status(
        status_root,
        "sources_ready",
        phase7_samples=len(earlier.phase7.samples),
        phase8_graphs=len(earlier.graph.graph_records),
        phase9_candidates=action.candidate_count,
        phase9_rollouts=action.rollout_count,
    )
    _log(
        "all topology sources ready | "
        f"Phase 9 candidates={action.candidate_count:,} | "
        f"rollouts={action.rollout_count:,}"
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
