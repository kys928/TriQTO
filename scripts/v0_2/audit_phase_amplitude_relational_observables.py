#!/usr/bin/env python3
"""Audit relational Z/X/Y observables before running the development probes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "train_phase_amplitude_relational_observable_probes.py"
MODULE_NAME = "triqto_relational_observable_probe_runner"


def load_runner():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    runner = load_runner()
    base = runner.load_base()
    root = args.product_dir.expanduser().resolve()
    complete = base.read_json(root / "generation_complete.json")
    if complete.get("test_split_accessed") is not False:
        raise RuntimeError("Pilot did not certify historical-test isolation")

    examples = base.load_examples(root)
    for index, item in enumerate(examples, start=1):
        runner.derived_observables(str(item.artifact), item.n_qubits)
        if index % args.progress_every == 0 or index == len(examples):
            print(f"Audited {index}/{len(examples)} entities", flush=True)

    if len(runner.AUDIT["verified_entities"]) != len(examples):
        raise RuntimeError(
            f"Audit covered {len(runner.AUDIT['verified_entities'])}/"
            f"{len(examples)} entities"
        )

    report = {
        key: (len(value) if key == "verified_entities" else float(value))
        for key, value in runner.AUDIT.items()
    }
    report["verification_atol"] = runner.VERIFY_ATOL
    report["status"] = "PASS"
    report["historical_v0_1_test_accessed"] = False

    print()
    print("=" * 78)
    print("TRIQTO RELATIONAL OBSERVABLE AUDIT PASS")
    print("=" * 78)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("The historical v0.1 test split was not accessed.")


if __name__ == "__main__":
    main()
