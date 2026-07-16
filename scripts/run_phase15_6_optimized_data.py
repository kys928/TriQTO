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
    args = parser.parse_args()
    result = run_optimized_data_stage(
        workspace=args.workspace,
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
