#!/usr/bin/env python3
"""Train the Phase 13 TriQTO model from a completed Phase 12 view dataset."""
from __future__ import annotations

import argparse
import json

from triqto.model import load_model_config
from triqto.training import load_training_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-config", default="configs/train/phase14_base.yaml")
    parser.add_argument("--model-config", default="configs/model/triqto_base.yaml")
    parser.add_argument("--phase7-root")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()

    result = run_training(
        training_view_root=args.training_view_root,
        output_root=args.output,
        training_config=load_training_config(args.train_config),
        model_config=load_model_config(args.model_config),
        phase7_root=args.phase7_root,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result.summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
