#!/usr/bin/env python3
"""Run and persist the deterministic Phase 10 TriQTO baseline suite."""
from __future__ import annotations

import argparse

from triqto.baselines import (
    load_baseline_config,
    run_baseline_suite,
    write_baseline_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-source", required=True)
    parser.add_argument("--graph-source", required=True)
    parser.add_argument("--action-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_baseline_config(args.config) if args.config else None
    result = run_baseline_suite(
        args.phase7_source,
        args.graph_source,
        args.action_source,
        config,
    )
    written = write_baseline_dataset(result, args.output)
    print(f"source_scientific_generation_id: {result.source_scientific_generation_id}")
    print(f"graph_conversion_id: {result.graph_conversion_id}")
    print(f"action_engine_id: {result.action_engine_id}")
    print(f"baseline_suite_id: {result.baseline_suite_id}")
    print(f"baseline_schema_id: {result.baseline_schema_id}")
    print(f"sample_count: {written.sample_count}")
    print(f"result_count: {written.result_count}")
    print(f"source_immutability_verified: {result.summary['source_immutability_verified']}")
    print(f"triqto_model_compared: {result.summary['triqto_model_compared']}")
    print(f"output: {written.output_root}")


if __name__ == "__main__":
    main()
