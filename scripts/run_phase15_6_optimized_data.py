#!/usr/bin/env python3
"""Run the resumable, parallel, sharded Phase 15.6 data stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triqto.phase15_6.optimized_data import run_optimized_data_stage


def _format_seconds(value: float) -> str:
    total = max(0, int(value))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _current_rss_gib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _count_markers(root: Path) -> int:
    try:
        return sum(1 for path in root.rglob("*.json") if path.is_file())
    except Exception:
        return 0


def _lazy_index_db_state(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = dict(connection.execute("SELECT key,value FROM index_state").fetchall())
        finally:
            connection.close()
        if rows.get("ready") == "1":
            return "lazy-index-ready"
        candidate_rows = int(rows.get("candidates_rows", "0"))
        rollout_rows = int(rows.get("rollouts_rows", "0"))
        if rollout_rows:
            return f"indexing-rollouts rows={rollout_rows:,}"
        if candidate_rows:
            return f"indexing-candidates rows={candidate_rows:,}"
    except Exception:
        return None
    return None


class _Heartbeat:
    def __init__(self, workspace: Path, interval_seconds: int) -> None:
        self.workspace = workspace
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self._last_wall = self.started
        self._last_cpu = time.process_time()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="triqto-pipeline-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        if self.interval_seconds > 0:
            self._thread.start()

    def stop(self) -> None:
        if self.interval_seconds <= 0:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))

    def _stage(self) -> str:
        data = self.workspace / "data"
        if (data / "phase15_6_data_complete.json").is_file():
            return "data-complete"
        progress = _read_json_object(data / ".phase9-lazy-index" / "progress.json")
        if progress is not None:
            stage = str(progress.get("stage", "lazy-index"))
            details: list[str] = []
            if "batch" in progress and "total_batches" in progress:
                details.append(f"batch={progress['batch']}/{progress['total_batches']}")
            if "rows" in progress and "total_rows" in progress:
                details.append(f"rows={int(progress['rows']):,}/{int(progress['total_rows']):,}")
            if "sample_id" in progress:
                details.append(f"sample={progress['sample_id']}")
            if "completed" in progress and "total" in progress:
                details.append(f"items={progress['completed']}/{progress['total']}")
            return stage + (" " + " ".join(details) if details else "")
        db_state = _lazy_index_db_state(data / ".phase9-lazy-index" / "action_index.sqlite3")
        if db_state is not None:
            return db_state
        if (data / "phase11" / "topology_complete.json").is_file():
            return "phase12"
        if (data / "phase9" / "action_complete.json").is_file():
            return "phase11-startup"
        return "pipeline-startup"

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            now = time.monotonic()
            cpu_now = time.process_time()
            wall_delta = max(now - self._last_wall, 1e-9)
            cpu_cores = max(0.0, (cpu_now - self._last_cpu) / wall_delta)
            self._last_wall = now
            self._last_cpu = cpu_now
            data = self.workspace / "data"
            phase11_markers = _count_markers(data / ".phase11-checkpoints" / "units")
            phase12_markers = _count_markers(data / ".phase12-checkpoints" / "markers")
            timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            print(
                "[Pipeline heartbeat] "
                f"time={timestamp} | alive={_format_seconds(now-self.started)} | "
                f"stage={self._stage()} | cpu≈{cpu_cores:.2f} cores | "
                f"RSS≈{_current_rss_gib():.2f} GiB | "
                f"phase11_markers={phase11_markers:,} | "
                f"phase12_item_markers={phase12_markers:,}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Bounded Phase 9 worker count; default is min(8, CPU count)",
    )
    parser.add_argument(
        "--phase11-workers",
        type=int,
        default=1,
        help=(
            "Bounded Phase 11 topology-group workers. Groups with at least 512 "
            "points run exclusively; default 1 is safest for memory."
        ),
    )
    parser.add_argument(
        "--phase12-shards",
        type=int,
        default=256,
        help="Deterministic hash-shard count per Phase 12 logical task",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("strict", "repair", "off"),
        default="strict",
        help=(
            "strict validates and reuses checkpoints; repair quarantines invalid "
            "checkpoints and recomputes them; off clears checkpoints for incomplete phases"
        ),
    )
    parser.add_argument(
        "--checkpoint-retention",
        choices=("phase", "campaign", "always"),
        default="campaign",
        help=(
            "phase removes checkpoints after each published phase; campaign keeps "
            "them for the wider campaign; always never requests automatic cleanup"
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Print an alive/stage/CPU/RAM/checkpoint heartbeat every N seconds; 0 disables",
    )
    args = parser.parse_args()
    if args.heartbeat_seconds < 0:
        parser.error("--heartbeat-seconds must be nonnegative")
    os.environ["TRIQTO_RESUME_MODE"] = args.resume_mode
    workspace = Path(args.workspace).expanduser().resolve()
    heartbeat = _Heartbeat(workspace, args.heartbeat_seconds)
    heartbeat.start()
    try:
        result = run_optimized_data_stage(
            workspace=workspace,
            workers=args.workers,
            phase11_workers=args.phase11_workers,
            phase12_shards=args.phase12_shards,
            resume_mode=args.resume_mode,
            checkpoint_retention=args.checkpoint_retention,
        )
    finally:
        heartbeat.stop()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
