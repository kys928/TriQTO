#!/usr/bin/env python3
"""Evaluate a trained TriQTO checkpoint on the untouched Phase 12 test split."""
from __future__ import annotations

import argparse
import json

from triqto.evaluation import (
    load_evaluation_config,
    run_evaluation,
    write_evaluation_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view-root", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--eval-config",
        default="configs/eval/phase15_base.yaml",
    )
    parser.add_argument("--phase7-root")
    parser.add_argument("--graph-root")
    parser.add_argument("--action-root")
    parser.add_argument("--baseline-root")
    args = parser.parse_args()

    config = load_evaluation_config(args.eval_config)
    result = run_evaluation(
        training_view_root=args.training_view_root,
        training_run_root=args.training_run_root,
        output_root=args.output,
        evaluation_config=config,
        phase7_root=args.phase7_root,
        graph_root=args.graph_root,
        action_root=args.action_root,
        baseline_root=args.baseline_root,
    )
    output = write_evaluation_dataset(result, args.output)
    print(
        json.dumps(
            {**result.summary, "output_root": str(output)},
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
