#!/usr/bin/env python3
"""Generate diagnostic persistent homology from a validated latent extraction."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import argparse
import json
from triqto.topology.latent import load_latent_topology_config
from triqto.topology.checkpoint_latent import run_checkpoint_bound_latent_topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-extraction-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/eval/latent_topology_smoke.yaml")
    args = parser.parse_args()
    result = run_checkpoint_bound_latent_topology(latent_extraction_root=args.latent_extraction_root, output_root=args.output, config=load_latent_topology_config(args.config))
    print(json.dumps({"latent_topology_id": result["result"]["latent_topology_id"], "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
