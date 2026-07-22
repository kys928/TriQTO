#!/usr/bin/env python3
"""Finish Phase 12 separately with fast checkpoint reuse and parallel shards."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triqto.phase15_6.action_ranking_projection import (
    install_action_ranking_projection,
)
from triqto.phase15_6.fast_phase12_publication import (
    write_training_view_dataset_fast,
)
import triqto.phase15_6.phase12_only as _phase12_only


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _rss_gib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _count(root: Path) -> int:
    try:
        return sum(1 for path in root.rglob("*.json") if path.is_file())
    except Exception:
        return 0


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Heartbeat:
    def __init__(self, workspace: Path, interval: int) -> None:
        self.workspace = workspace
        self.interval = interval
        self.started = time.monotonic()
        self.last_wall = self.started
        self.last_cpu = time.process_time()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.interval > 0:
            self.thread.start()

    def stop(self) -> None:
        if self.interval > 0:
            self.stop_event.set()
            self.thread.join(timeout=self.interval + 2)

    def _stage(self) -> str:
        data = self.workspace / "data"
        phase12 = _read_json(data / ".phase12-checkpoints" / "progress.json")
        lazy = _read_json(data / ".phase9-lazy-index" / "progress.json")
        if phase12 is not None and phase12.get("stage") == "logical_shards":
            eta = phase12.get("eta_seconds")
            eta_text = "unknown" if eta is None else _duration(float(eta))
            return (
                f"phase12 task={phase12.get('task')} "
                f"shards={phase12.get('completed_shards')}/"
                f"{phase12.get('total_shards')} "
                f"entities={phase12.get('completed_entities')}/"
                f"{phase12.get('total_entities')} ETA≈{eta_text}"
            )
        if lazy is not None and lazy.get("stage") == "hydrate_sample":
            return (
                f"phase12 hydrate sample={lazy.get('sample_id')} "
                f"artifacts={lazy.get('completed')}/{lazy.get('total')}"
            )
        if phase12 is not None:
            return f"phase12 {phase12.get('stage', 'startup')}"
        return "phase12-startup"

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            now = time.monotonic()
            cpu_now = time.process_time()
            cores = max(
                0.0,
                (cpu_now - self.last_cpu) / max(now - self.last_wall, 1e-9),
            )
            self.last_wall = now
            self.last_cpu = cpu_now
            data = self.workspace / "data"
            stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            print(
                "[Phase 12 heartbeat] "
                f"time={stamp} | alive={_duration(now-self.started)} | "
                f"stage={self._stage()} | cpu≈{cores:.2f} cores | "
                f"RSS≈{_rss_gib():.2f} GiB | "
                f"item_markers={_count(data/'.phase12-checkpoints'/'markers'):,}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--phase12-workers", type=int, default=4)
    parser.add_argument("--phase12-shards", type=int, default=256)
    parser.add_argument(
        "--resume-mode",
        choices=("strict", "repair", "off"),
        default="strict",
    )
    parser.add_argument(
        "--checkpoint-retention",
        choices=("phase", "campaign", "always"),
        default="campaign",
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=15)
    args = parser.parse_args()
    if args.heartbeat_seconds < 0:
        parser.error("--heartbeat-seconds must be nonnegative")
    os.environ["TRIQTO_RESUME_MODE"] = args.resume_mode
    os.environ.setdefault(
        "TRIQTO_PHASE12_PUBLICATION_WORKERS",
        str(args.phase12_workers),
    )
    install_action_ranking_projection()
    _phase12_only.write_training_view_dataset_resumable = write_training_view_dataset_fast
    print(
        "[Phase 12] action-ranking projection enabled | "
        "candidate_qpy_skipped=true | rollout_arrays_skipped=true | "
        "policy=lossless_all_candidates",
        flush=True,
    )
    print(
        "[Phase 12] fast publication enabled | "
        "redundant_npz_reloads_skipped=true | "
        f"publication_workers={os.environ['TRIQTO_PHASE12_PUBLICATION_WORKERS']}",
        flush=True,
    )
    workspace = Path(args.workspace).expanduser().resolve()
    heartbeat = Heartbeat(workspace, args.heartbeat_seconds)
    heartbeat.start()
    try:
        result = _phase12_only.run_phase12_only(
            workspace=workspace,
            phase12_workers=args.phase12_workers,
            phase12_shards=args.phase12_shards,
            resume_mode=args.resume_mode,
            checkpoint_retention=args.checkpoint_retention,
        )
    finally:
        heartbeat.stop()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
