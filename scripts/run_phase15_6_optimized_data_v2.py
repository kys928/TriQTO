#!/usr/bin/env python3
"""Run Phase 15.6 with lossless Phase 11 group expansion and live telemetry."""
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

from triqto.phase15_6.optimized_data_v2 import run_optimized_data_stage


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
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


def _format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_stage(progress: dict[str, Any]) -> str:
    stage = str(progress.get("stage", "phase11"))
    details: list[str] = []
    if progress.get("detail"):
        details.append(str(progress["detail"]))
    if "batch" in progress and "total_batches" in progress:
        details.append(f"batch={progress['batch']}/{progress['total_batches']}")
    if "rows" in progress and "total_rows" in progress:
        details.append(f"rows={int(progress['rows']):,}/{int(progress['total_rows']):,}")
    if "current_group_index" in progress and "total_groups" in progress:
        details.append(
            f"group={int(progress['current_group_index']):,}/"
            f"{int(progress['total_groups']):,}"
        )
    if "point_count" in progress:
        details.append(f"points={int(progress['point_count']):,}")
    if progress.get("group_kind"):
        details.append(str(progress["group_kind"]))
    if progress.get("stage_status"):
        details.append(str(progress["stage_status"]))
    if "eta_seconds" in progress:
        details.append(f"ETA≈{_format_duration(float(progress['eta_seconds']))}")
    return stage + (" | " + " | ".join(details) if details else "")


def _lazy_db_state(path: Path) -> str | None:
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
        if int(rows.get("rollouts_rows", "0")):
            return f"indexing-rollouts rows={int(rows['rollouts_rows']):,}"
        if int(rows.get("candidates_rows", "0")):
            return f"indexing-candidates rows={int(rows['candidates_rows']):,}"
    except Exception:
        return None
    return None


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
        lazy = _read_json(data / ".phase9-lazy-index" / "progress.json")
        if lazy is not None and str(lazy.get("stage")) in {
            "index_candidates", "index_rollouts", "finalize_index", "hydrate_sample"
        }:
            return _format_stage(lazy)
        phase11 = _read_json(data / ".phase11-checkpoints" / "progress.json")
        if phase11 is not None:
            return _format_stage(phase11)
        if lazy is not None:
            return _format_stage(lazy)
        return _lazy_db_state(data / ".phase9-lazy-index" / "action_index.sqlite3") or "pipeline-startup"

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            now = time.monotonic()
            cpu = time.process_time()
            cores = max(0.0, (cpu - self.last_cpu) / max(now - self.last_wall, 1e-9))
            self.last_wall, self.last_cpu = now, cpu
            data = self.workspace / "data"
            stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            print(
                "[Pipeline heartbeat] "
                f"time={stamp} | alive={_format_duration(now-self.started)} | "
                f"stage={self._stage()} | cpu≈{cores:.2f} cores | RSS≈{_rss_gib():.2f} GiB | "
                f"phase11_markers={_count(data/'.phase11-checkpoints'/'units'):,} | "
                f"phase12_item_markers={_count(data/'.phase12-checkpoints'/'markers'):,}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--phase11-workers", type=int, default=1)
    parser.add_argument("--phase12-shards", type=int, default=256)
    parser.add_argument("--resume-mode", choices=("strict", "repair", "off"), default="strict")
    parser.add_argument(
        "--checkpoint-retention", choices=("phase", "campaign", "always"), default="campaign"
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.heartbeat_seconds < 0:
        parser.error("--heartbeat-seconds must be nonnegative")
    os.environ["TRIQTO_RESUME_MODE"] = args.resume_mode
    workspace = Path(args.workspace).expanduser().resolve()
    heartbeat = Heartbeat(workspace, args.heartbeat_seconds)
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
