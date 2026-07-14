#!/usr/bin/env python3
"""Build and persist deterministic Phase 12 task-specific training views."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from triqto.training_views import (
    build_training_view_result,
    load_training_view_config,
    write_training_view_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-source", required=True)
    parser.add_argument("--graph-source", required=True)
    parser.add_argument("--action-source", required=True)
    parser.add_argument("--topology-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_training_view_config(args.config) if args.config else None
    result = build_training_view_result(
        args.phase7_source,
        args.graph_source,
        args.action_source,
        args.topology_source,
        config,
    )
    written = write_training_view_dataset(result, args.output)
    print(f"source_scientific_generation_id: {result.source_scientific_generation_id}")
    print(f"graph_conversion_id: {result.graph_conversion_id}")
    print(f"action_engine_id: {result.action_engine_id}")
    print(f"topology_audit_id: {result.topology_audit_id}")
    print(f"training_view_dataset_id: {result.training_view_dataset_id}")
    print(f"training_view_schema_id: {result.training_view_schema_id}")
    print(f"view_count: {written.view_count}")
    print(f"item_count: {written.item_count}")
    print(f"task_item_counts: {result.summary['task_item_counts']}")
    print(f"split_item_counts: {result.summary['split_item_counts']}")
    print(f"topology_loss_weight: {result.summary['topology_loss_weight']}")
    print(f"training_executed: {result.summary['training_executed']}")
    print(f"source_immutability_verified: {result.summary['source_immutability_verified']}")
    print(f"output: {written.output_root}")


if __name__ == "__main__":
    main()
