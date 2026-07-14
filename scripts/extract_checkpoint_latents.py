#!/usr/bin/env python3
"""Extract deterministic latent coordinates from a trained Phase 14 checkpoint."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import argparse
import json
from triqto.training.latent_extraction import extract_checkpoint_latents, load_latent_extraction_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/eval/latent_extraction_smoke.yaml")
    parser.add_argument("--phase7-root")
    args = parser.parse_args()
    result = extract_checkpoint_latents(training_view_root=args.training_view_root, training_root=args.training_root, checkpoint=args.checkpoint, output_root=args.output, config=load_latent_extraction_config(args.config), phase7_root=args.phase7_root)
    print(json.dumps({"latent_extraction_id": result["metadata"]["latent_extraction_id"], "point_count": result["metadata"]["point_count"], "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
