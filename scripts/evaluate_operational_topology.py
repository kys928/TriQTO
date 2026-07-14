#!/usr/bin/env python3
"""Run Phase 15 with separate operational-action and latent-topology reporting."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import argparse
import json
from triqto.evaluation import load_phase15_config
from triqto.evaluation.integrated import load_integrated_phase15_config, run_integrated_phase15_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--operational-action-root", required=True)
    parser.add_argument("--latent-topology-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/eval/phase15_operational_topology_smoke.yaml")
    parser.add_argument("--phase7-root")
    args = parser.parse_args()
    integrated = load_integrated_phase15_config(args.config)
    result = run_integrated_phase15_evaluation(training_view_root=args.training_view_root, training_root=args.training_root, checkpoint=args.checkpoint, output_root=args.output, core_config=load_phase15_config(integrated.core_config), integration_config=integrated, operational_action_root=args.operational_action_root, latent_topology_root=args.latent_topology_root, phase7_root=args.phase7_root)
    print(json.dumps({"integrated_phase15_run_id": result["summary"]["integrated_phase15_run_id"], "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
