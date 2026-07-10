#!/usr/bin/env python
"""Convert a completed Phase 7 dataset into Phase 8 graph artifacts."""
from __future__ import annotations
import argparse
from triqto.graph import convert_completed_dataset_to_graphs, load_graph_config, write_graph_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    args = parser.parse_args()
    cfg = load_graph_config(args.config) if args.config else None
    result = convert_completed_dataset_to_graphs(args.source, cfg)
    write = write_graph_dataset(result, args.output)
    print(f"source_scientific_generation_id: {result.source_scientific_generation_id}")
    print(f"graph_conversion_id: {result.graph_conversion_id}")
    print(f"graph_count: {write.graph_count}")
    print(f"pair_count: {write.pair_count}")
    print(f"total_nodes: {result.summary['total_nodes']}")
    print(f"total_directed_edges: {result.summary['total_directed_edges']}")
    print(f"total_gate_events: {result.summary['total_gate_events']}")
    print(f"output: {write.output_root}")

if __name__ == '__main__':
    main()
