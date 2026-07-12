#!/usr/bin/env python3
"""Build and persist the deterministic Phase 11 topology audit dataset."""
from __future__ import annotations

import argparse

from triqto.topology import (
    build_topology_audit_result,
    load_topology_config,
    write_topology_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-source", required=True)
    parser.add_argument("--graph-source", required=True)
    parser.add_argument("--action-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_topology_config(args.config) if args.config else None
    result = build_topology_audit_result(
        args.phase7_source,
        args.graph_source,
        args.action_source,
        config,
    )
    written = write_topology_dataset(result, args.output)
    print(f"source_scientific_generation_id: {result.source_scientific_generation_id}")
    print(f"graph_conversion_id: {result.graph_conversion_id}")
    print(f"action_engine_id: {result.action_engine_id}")
    print(f"topology_audit_id: {result.topology_audit_id}")
    print(f"topology_schema_id: {result.topology_schema_id}")
    print(f"group_count: {written.group_count}")
    print(f"total_group_point_count: {written.point_count}")
    print(f"hilbert_group_count: {result.summary['hilbert_group_count']}")
    print(f"topology_loss_weight: {result.summary['topology_loss_weight']}")
    print(f"source_immutability_verified: {result.summary['source_immutability_verified']}")
    print(f"output: {written.output_root}")


if __name__ == "__main__":
    main()
