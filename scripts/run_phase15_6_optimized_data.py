#!/usr/bin/env python3
"""Run the resumable, parallel, sharded Phase 15.6 data stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triqto.phase15_6.optimized_data import run_optimized_data_stage


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
    args = parser.parse_args()
    result = run_optimized_data_stage(
        workspace=args.workspace,
        workers=args.workers,
        phase11_workers=args.phase11_workers,
        phase12_shards=args.phase12_shards,
        resume_mode=args.resume_mode,
        checkpoint_retention=args.checkpoint_retention,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
