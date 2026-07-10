#!/usr/bin/env python
"""Convert a completed Phase 7 dataset into immutable Phase 8 graph artifacts."""
from __future__ import annotations

import argparse

from triqto.graph import (
    convert_completed_dataset_to_graphs,
    load_graph_config,
    write_graph_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Completed Phase 7 dataset root")
    parser.add_argument("--output", required=True, help="New Phase 8 graph dataset root")
    parser.add_argument("--config", help="Optional strict Phase 8 graph config JSON")
    arguments = parser.parse_args()

    config = load_graph_config(arguments.config) if arguments.config else None
    result = convert_completed_dataset_to_graphs(arguments.source, config)
    written = write_graph_dataset(result, arguments.output)

    print(f"source_scientific_generation_id: {result.source_scientific_generation_id}")
    print(f"graph_conversion_id: {result.graph_conversion_id}")
    print(f"graph_schema_id: {result.graph_schema_id}")
    print(f"graph_count: {written.graph_count}")
    print(f"pair_count: {written.pair_count}")
    print(f"total_nodes: {result.summary['total_nodes']}")
    print(f"total_directed_edges: {result.summary['total_directed_edges']}")
    print(f"total_gate_events: {result.summary['total_gate_events']}")
    print(
        "source_immutability_verified: "
        f"{result.summary['source_immutability_verified']}"
    )
    print(f"output: {written.output_root}")


if __name__ == "__main__":
    main()
