#!/usr/bin/env python3
"""Audit all relational observables using dtype-aware tolerances."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "train_phase_amplitude_relational_observable_probes_v2.py"
MODULE_NAME = "triqto_v0_2_relational_observable_probe_v2"


def load_runner():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module.load_patched_runner()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    config = parser.parse_args()

    runner = load_runner()
    base = runner.load_base()
    root = config.product_dir.expanduser().resolve()
    complete = base.read_json(root / "generation_complete.json")
    if complete.get("test_split_accessed") is not False:
        raise RuntimeError("Pilot did not certify historical-test isolation")

    examples = base.load_examples(root)
    for index, item in enumerate(examples, start=1):
        runner.derived_observables(str(item.artifact), item.n_qubits)
        if config.progress_every > 0 and (
            index % config.progress_every == 0 or index == len(examples)
        ):
            print(f"Audited {index}/{len(examples)} entities", flush=True)

    report = {
        key: (len(value) if key == "verified_entities" else float(value))
        for key, value in runner.AUDIT.items()
    }
    report.update(
        {
            "status": "PASS",
            "product_id": complete["product_id"],
            "historical_v0_1_test_accessed": False,
            "tolerance_policy": {
                "stored_z_float32_absolute": runner.AUDIT[
                    "stored_z_float32_atol"
                ],
                "float64_observables_absolute": runner.AUDIT[
                    "float64_observable_atol"
                ],
            },
        }
    )

    output = root / "reports" / "relational_observable_audit_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("TRIQTO RELATIONAL OBSERVABLE AUDIT V2 PASS")
    print("=" * 78)
    print(f"verified_entities: {report['verified_entities']}")
    print(
        "stored-Z float32 tolerance: "
        f"{report['stored_z_float32_atol']:.1e}"
    )
    print(
        "float64 observable tolerance: "
        f"{report['float64_observable_atol']:.1e}"
    )
    print(
        "max stored-Z error: "
        f"{report['max_distorted_z_probability_abs_error']:.3e}"
    )
    print(f"Report: {output}")
    print("The historical v0.1 test split was not accessed.")


if __name__ == "__main__":
    main()
