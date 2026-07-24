#!/usr/bin/env python3
"""Attach explicit deployable Born query coordinates to a model-ready product."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from triqto.phase15_6.born_query_attachment import attach_born_query_coordinates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    result = attach_born_query_coordinates(
        source_root=args.source_root,
        output_parent=args.output_parent,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
