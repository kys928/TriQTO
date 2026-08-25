#!/usr/bin/env python3
"""Directory-compatible wrapper for the frozen Step-13 ideal replay.

This consumes the original Step-12 run directory directly, verifies the file
hashes recorded by step12_complete.json, and performs the same no-QPU replay as
run_step13_ideal_counterfactual_replay.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import run_step11_exploratory_ibm_transfer_pilot as step11
import run_step12_independent_phase_generalization as step12
import run_step13_ideal_counterfactual_replay as base
from triqto.hardware.diagnostic_acquisition import build_step7_model_batch_from_counts
from triqto.hardware.dry_run import predict_frozen_ensemble

EXPECTED_PLAN_ID = "step12plan_24b2631b5cd94c58af951492"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG)
    p.add_argument("--step10c-benchmark-dir", type=Path, default=base.DEFAULT_STEP10C)
    p.add_argument("--step10d-benchmark-dir", type=Path, default=base.DEFAULT_STEP10D)
    p.add_argument("--step9a-bundle-dir", type=Path, default=base.DEFAULT_STEP9A)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.expanduser().resolve()
    complete = json.loads((run_dir / "step12_complete.json").read_text())
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("Step-12 run directory is not COMPLETE")
    if complete.get("plan_id") != EXPECTED_PLAN_ID:
        raise RuntimeError("unexpected Step-12 plan identity")
    for name, expected in complete.get("file_hashes", {}).items():
        path = run_dir / str(name)
        if not path.is_file():
            raise RuntimeError(f"missing frozen Step-12 artifact: {name}")
        actual = "sha256:" + sha256_file(path)
        if actual != str(expected):
            raise RuntimeError(f"Step-12 artifact hash mismatch: {name}")
    plan = json.loads((run_dir / "generalization_plan.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "case_results.jsonl").read_text().splitlines() if line.strip()]
    return plan, rows


def main() -> None:
    args = parse_args()
    device = base._device(args.device)
    config = json.loads(args.config.expanduser().resolve().read_text())
    step12._assert_frozen_contract(config)
    step11._verify_step10d_reference(args.step10d_benchmark_dir, config)
    primary = step11._load_step10c_primary(args.step10c_benchmark_dir, config, device=device)
    baseline = step11._load_step9a_baseline(args.step9a_bundle_dir, config, device=device)
    plan, frozen_rows = load_frozen_run(args.run_dir)
    chain = tuple(int(v) for v in plan["identity"]["physical_chain"])
    cases = step12.build_generalization_cases(config, chain)
    frozen_by_id = {str(row["case_id"]): row for row in frozen_rows}

    modes = ("hardware", "ideal_all", "ideal_local_only", "ideal_pair_parity_only")
    replay_rows: list[dict[str, Any]] = []
    hardware_reproduction_ok = True

    for case in cases:
        frozen = frozen_by_id[case.case_id]
        counts = frozen["counts_by_program"]
        reference_counts = {basis: counts[f"reference_{basis}"] for basis in ("Z", "X", "Y")}
        observed_counts = {basis: counts[f"observed_{basis}"] for basis in ("Z", "X", "Y")}
        hardware_batch = build_step7_model_batch_from_counts(
            case.reference_circuit,
            case.physical_layout,
            reference_counts,
            observed_counts,
            device=device,
        )
        ideal_local, ideal_pair, ideal_parity = base._ideal_diagnostic_tensors(case, device=device)
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "family": case.family,
            "strength": case.strength,
            "expected_effect": case.expected_effect,
            "expected_mechanism": case.expected_mechanism,
        }
        for mode in modes:
            batch = base._variant_batch(
                hardware_batch,
                ideal_local=ideal_local,
                ideal_pair=ideal_pair,
                ideal_parity=ideal_parity,
                mode=mode,
            )
            row[f"step10c__{mode}"] = predict_frozen_ensemble(primary, batch)
            row[f"step9a__{mode}"] = predict_frozen_ensemble(baseline, batch)
        if row["step10c__hardware"]["mechanism_prediction"] != frozen["step10c_prediction"]["mechanism_prediction"]:
            hardware_reproduction_ok = False
        if row["step9a__hardware"]["mechanism_prediction"] != frozen["step9a_prediction"]["mechanism_prediction"]:
            hardware_reproduction_ok = False
        replay_rows.append(row)

    result: dict[str, Any] = {
        "schema": "triqto.v0_2.step13_ideal_counterfactual_replay.v1",
        "status": "COMPLETE_NO_QPU_NO_TRAINING",
        "source_run_dir": str(args.run_dir.expanduser().resolve()),
        "source_plan_id": EXPECTED_PLAN_ID,
        "hardware_prediction_reproduction": "PASS" if hardware_reproduction_ok else "FAIL",
        "primary": {},
        "baseline_report_only": {},
        "rows": replay_rows,
    }
    for mode in modes:
        result["primary"][mode] = base._metrics(replay_rows, f"step10c__{mode}")
        result["baseline_report_only"][mode] = base._metrics(replay_rows, f"step9a__{mode}")
    if not hardware_reproduction_ok:
        raise RuntimeError("counterfactual replay did not reproduce frozen hardware predictions")

    hardware_correct = int(result["primary"]["hardware"]["mechanism_correct"])
    ideal_correct = int(result["primary"]["ideal_all"]["mechanism_correct"])
    if ideal_correct <= hardware_correct + 2:
        conclusion = "IDEAL_DIAGNOSTICS_DO_NOT_RESCUE_MODEL__TRAINING_OR_GRAPH_CONTEXT_GENERALIZATION_PRIMARY"
    elif ideal_correct >= 14:
        conclusion = "IDEAL_DIAGNOSTICS_SUBSTANTIALLY_RESCUE_MODEL__HARDWARE_DOMAIN_SHIFT_MATERIALLY_CONTRIBUTES"
    else:
        conclusion = "MIXED_IDEAL_REPLAY__BOTH_MODEL_GENERALIZATION_AND_HARDWARE_DOMAIN_SHIFT_PLAUSIBLE"
    result["diagnostic_conclusion"] = conclusion
    result["interpretation_boundary"] = {
        "diagnostic_only": True,
        "qpu_access": False,
        "training": False,
        "weights_changed": False,
        "thresholds_changed": False,
        "step12_outcome_rewritten": False,
    }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Wrote: {output}")

    print("\nTRIQTO STEP 13 IDEAL-DIAGNOSTIC COUNTERFACTUAL REPLAY COMPLETE")
    print("Hardware reproduction:", result["hardware_prediction_reproduction"])
    for mode in modes:
        print(f"Step10C {mode}: {result['primary'][mode]['mechanism_correct']}/18")
    print("Conclusion:", conclusion)


if __name__ == "__main__":
    main()
