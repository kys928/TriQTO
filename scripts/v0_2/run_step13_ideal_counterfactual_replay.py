#!/usr/bin/env python3
"""No-QPU counterfactual replay of the frozen Step-12 cases through frozen models.

The script reconstructs the exact Step-12 graph/layout inputs and compares four
frozen-inference conditions on the already-acquired cases:

1. hardware diagnostics (must reproduce the frozen Step-12 predictions),
2. exact ideal statevector diagnostics,
3. ideal local diagnostics + hardware pair/parity diagnostics,
4. hardware local diagnostics + ideal pair/parity diagnostics.

No weights, thresholds, labels, or QPU settings are changed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any, Mapping

import numpy as np
import torch
from qiskit.quantum_info import Pauli, Statevector

import run_step11_exploratory_ibm_transfer_pilot as step11
import run_step12_independent_phase_generalization as step12
from triqto.hardware.diagnostic_acquisition import (
    all_pair_indices,
    build_step7_model_batch_from_counts,
)
from triqto.hardware.dry_run import predict_frozen_ensemble

EXPECTED_BUNDLE_SHA256 = "9c9a4a726e0403a466960669917a394ef7df73c2f6fb12239dbdf799f818d7ff"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "v0_2" / "step12_independent_phase_generalization.json"
DEFAULT_STEP10C = Path("/workspace/triqto-data/step10c_crashsafe_long_horizon/benchmark_f9478da45d68795655259054")
DEFAULT_STEP10D = Path("/workspace/triqto-data/step10d_final_simulator_lr_refinement/benchmark_1455864a09de8804a7e7958a")
DEFAULT_STEP9A = Path("/workspace/triqto-data/step9a_deployment_bundle/deploy_ac536a74b2f8dd571d353a12")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--step10c-benchmark-dir", type=Path, default=DEFAULT_STEP10C)
    p.add_argument("--step10d-benchmark-dir", type=Path, default=DEFAULT_STEP10D)
    p.add_argument("--step9a-bundle-dir", type=Path, default=DEFAULT_STEP9A)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _bundle(bundle: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = bundle.expanduser().resolve()
    observed = sha256_file(bundle)
    if observed != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError(f"Step-12 source-bundle SHA256 mismatch: {observed}")
    with zipfile.ZipFile(bundle) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Step-12 source ZIP failed CRC validation")
        roots = {name.split("/", 1)[0] for name in zf.namelist() if "/" in name}
        if len(roots) != 1:
            raise RuntimeError("expected one Step-12 bundle root")
        root = next(iter(roots)) + "/"
        plan = json.loads(zf.read(root + "generalization_plan.json"))
        rows = [json.loads(line) for line in zf.read(root + "case_results.jsonl").decode().splitlines() if line.strip()]
    if plan.get("plan_id") != "step12plan_24b2631b5cd94c58af951492":
        raise RuntimeError("unexpected Step-12 plan identity")
    return plan, rows


def _expectation(state: Statevector, basis: str, qubits: tuple[int, ...], n: int) -> float:
    label = ["I"] * n
    for q in qubits:
        label[n - 1 - int(q)] = basis
    return float(np.real(state.expectation_value(Pauli("".join(label)))))


def _ideal_diagnostic_tensors(case: step12.PilotCase, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = Statevector.from_instruction(case.reference_circuit)
    obs = Statevector.from_instruction(case.observed_circuit)
    n = int(case.reference_circuit.num_qubits)
    pairs = all_pair_indices(n)
    local = np.zeros((n, 3), dtype=np.float32)
    pair = np.zeros((len(pairs), 3), dtype=np.float32)
    parity = np.zeros((1, 3), dtype=np.float32)
    for basis_index, basis in enumerate(("Z", "X", "Y")):
        for q in range(n):
            local[q, basis_index] = _expectation(obs, basis, (q,), n) - _expectation(ref, basis, (q,), n)
        for pair_index, (left, right) in enumerate(pairs.tolist()):
            pair[pair_index, basis_index] = (
                _expectation(obs, basis, (int(left), int(right)), n)
                - _expectation(ref, basis, (int(left), int(right)), n)
            )
        parity[0, basis_index] = (
            _expectation(obs, basis, tuple(range(n)), n)
            - _expectation(ref, basis, tuple(range(n)), n)
        )
    return (
        torch.as_tensor(local, dtype=torch.float32, device=device),
        torch.as_tensor(pair, dtype=torch.float32, device=device),
        torch.as_tensor(parity, dtype=torch.float32, device=device),
    )


def _variant_batch(hardware_batch: Any, *, ideal_local: torch.Tensor, ideal_pair: torch.Tensor, ideal_parity: torch.Tensor, mode: str) -> Any:
    d = hardware_batch.diagnostic
    if mode == "hardware":
        return hardware_batch
    if mode == "ideal_all":
        local, pair, parity = ideal_local, ideal_pair, ideal_parity
    elif mode == "ideal_local_only":
        local, pair, parity = ideal_local, d.pair_values, d.global_parity
    elif mode == "ideal_pair_parity_only":
        local, pair, parity = d.local_values, ideal_pair, ideal_parity
    else:
        raise ValueError(mode)
    diagnostic = replace(d, local_values=local, pair_values=pair, global_parity=parity)
    batch = replace(hardware_batch, diagnostic=diagnostic)
    diagnostic.validate(batch.graph)
    return batch


def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    distorted = [r for r in rows if bool(r["expected_effect"])]
    correct = sum(str(r[key]["mechanism_prediction"]) == str(r["expected_mechanism"]) for r in distorted)
    by_motif: dict[str, dict[str, int]] = {}
    by_mechanism: dict[str, dict[str, int]] = {}
    for motif in sorted({str(r["family"]) for r in distorted}):
        subset = [r for r in distorted if str(r["family"]) == motif]
        by_motif[motif] = {"correct": sum(str(r[key]["mechanism_prediction"]) == str(r["expected_mechanism"]) for r in subset), "count": len(subset)}
    for mechanism in ("rz_drift", "rx_overrotation", "ry_overrotation"):
        subset = [r for r in distorted if str(r["expected_mechanism"]) == mechanism]
        by_mechanism[mechanism] = {"correct": sum(str(r[key]["mechanism_prediction"]) == mechanism for r in subset), "count": len(subset)}
    return {"mechanism_correct": int(correct), "case_count": len(distorted), "by_motif": by_motif, "by_mechanism": by_mechanism}


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    config = json.loads(args.config.expanduser().resolve().read_text())
    step12._assert_frozen_contract(config)
    step11._verify_step10d_reference(args.step10d_benchmark_dir, config)
    primary = step11._load_step10c_primary(args.step10c_benchmark_dir, config, device=device)
    baseline = step11._load_step9a_baseline(args.step9a_bundle_dir, config, device=device)
    plan, frozen_rows = _bundle(args.bundle)
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
        ideal_local, ideal_pair, ideal_parity = _ideal_diagnostic_tensors(case, device=device)
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "family": case.family,
            "strength": case.strength,
            "expected_effect": case.expected_effect,
            "expected_mechanism": case.expected_mechanism,
        }
        for mode in modes:
            batch = _variant_batch(
                hardware_batch,
                ideal_local=ideal_local,
                ideal_pair=ideal_pair,
                ideal_parity=ideal_parity,
                mode=mode,
            )
            p10 = predict_frozen_ensemble(primary, batch)
            p9 = predict_frozen_ensemble(baseline, batch)
            row[f"step10c__{mode}"] = p10
            row[f"step9a__{mode}"] = p9
        if row["step10c__hardware"]["mechanism_prediction"] != frozen["step10c_prediction"]["mechanism_prediction"]:
            hardware_reproduction_ok = False
        if row["step9a__hardware"]["mechanism_prediction"] != frozen["step9a_prediction"]["mechanism_prediction"]:
            hardware_reproduction_ok = False
        replay_rows.append(row)

    result: dict[str, Any] = {
        "schema": "triqto.v0_2.step13_ideal_counterfactual_replay.v1",
        "status": "COMPLETE_NO_QPU_NO_TRAINING",
        "source_bundle_sha256": "sha256:" + EXPECTED_BUNDLE_SHA256,
        "hardware_prediction_reproduction": "PASS" if hardware_reproduction_ok else "FAIL",
        "primary": {},
        "baseline_report_only": {},
        "interpretation_boundary": {
            "diagnostic_only": true,
            "qpu_access": false,
            "training": false,
            "weights_changed": false,
            "thresholds_changed": false,
            "step12_outcome_rewritten": false
        },
        "rows": replay_rows,
    }
    for mode in modes:
        result["primary"][mode] = _metrics(replay_rows, f"step10c__{mode}")
        result["baseline_report_only"][mode] = _metrics(replay_rows, f"step9a__{mode}")

    if not hardware_reproduction_ok:
        raise RuntimeError("counterfactual replay did not reproduce frozen hardware predictions")

    h = int(result["primary"]["hardware"]["mechanism_correct"])
    ideal = int(result["primary"]["ideal_all"]["mechanism_correct"])
    if ideal <= h + 2:
        conclusion = "IDEAL_DIAGNOSTICS_DO_NOT_RESCUE_MODEL__TRAINING_OR_GRAPH_CONTEXT_GENERALIZATION_PRIMARY"
    elif ideal >= 14:
        conclusion = "IDEAL_DIAGNOSTICS_SUBSTANTIALLY_RESCUE_MODEL__HARDWARE_DOMAIN_SHIFT_MATERIALLY_CONTRIBUTES"
    else:
        conclusion = "MIXED_IDEAL_REPLAY__BOTH_MODEL_GENERALIZATION_AND_HARDWARE_DOMAIN_SHIFT_PLAUSIBLE"
    result["diagnostic_conclusion"] = conclusion

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
