#!/usr/bin/env python
"""Build a deterministic Phase 9 action-candidate dataset."""
from __future__ import annotations

import argparse

from triqto.actions import (
    build_action_engine_result,
    load_action_config,
    write_action_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-source", required=True)
    parser.add_argument("--graph-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_action_config(args.config) if args.config else None
    result = build_action_engine_result(
        args.phase7_source,
        args.graph_source,
        config,
    )
    written = write_action_dataset(result, args.output)
    print(
        f"source_scientific_generation_id: "
        f"{result.source_scientific_generation_id}"
    )
    print(f"graph_conversion_id: {result.graph_conversion_id}")
    print(f"action_engine_id: {result.action_engine_id}")
    print(f"action_schema_id: {result.action_schema_id}")
    print(f"candidate_count: {written.candidate_count}")
    print(f"rollout_count: {written.rollout_count}")
    print(
        f"improving_rollout_count: "
        f"{result.summary['improving_rollout_count']}"
    )
    print(
        f"source_immutability_verified: "
        f"{result.summary['source_immutability_verified']}"
    )
    print(f"output: {written.output_root}")


if __name__ == "__main__":
    main()
