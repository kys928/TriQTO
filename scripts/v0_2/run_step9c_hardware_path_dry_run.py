#!/usr/bin/env python3
"""Run the frozen Step-9C local hardware-path dry run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any

import numpy as np
import torch

from triqto.hardware.diagnostic_acquisition import acquire_paired_diagnostics
from triqto.hardware.dry_run import (
    assert_training_hardware_batch_equivalent,
    build_dry_run_cases,
    build_local_backend,
    build_local_sampler,
    load_frozen_deployment_ensemble,
    predict_frozen_ensemble,
    read_json,
    sha256_file,
    verify_acquisition_structure,
    verify_step9b_contract,
)
from triqto.step7.graph_adapter import batch_from_step5_examples
from triqto.hardware.diagnostic_acquisition import (
    BASIS_ORDER,
    paired_diagnostic_arrays,
    serialize_intended_graph,
)

SCHEMA = "triqto.v0_2.step9c_hardware_path_dry_run.v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v0_2/step9c_hardware_path_dry_run.json"
DEFAULT_STEP9B_CONFIG = ROOT / "configs/v0_2/step9b_hardware_acquisition.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step9c_hardware_path_dry_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--step9b-config", type=Path, default=DEFAULT_STEP9B_CONFIG)
    parser.add_argument("--deployment-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args()


def git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    payload = f"blob {len(content)}\0".encode("ascii") + content
    return hashlib.sha1(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _training_adapter_batch(case: Any, acquisition: Any, device: torch.device) -> Any:
    reference_counts = {
        basis: acquisition.counts_by_program[f"reference_{basis}"] for basis in BASIS_ORDER
    }
    observed_counts = {
        basis: acquisition.counts_by_program[f"observed_{basis}"] for basis in BASIS_ORDER
    }
    example: dict[str, np.ndarray] = {}
    example.update(serialize_intended_graph(case.reference_circuit, case.initial_layout))
    example.update(
        paired_diagnostic_arrays(
            reference_counts,
            observed_counts,
            case.reference_circuit.num_qubits,
        )
    )
    example["y__effect_present_target"] = np.asarray([0], dtype=np.int8)
    example["y__mechanism_target"] = np.asarray([0], dtype=np.int8)
    example["y__mechanism_loss_mask"] = np.asarray([False], dtype=np.bool_)
    batch, _targets = batch_from_step5_examples([example], device=device)
    return batch


def _diagnostic_summary(batch: Any) -> dict[str, Any]:
    local = batch.diagnostic.local_values.detach().cpu().numpy()
    pair = batch.diagnostic.pair_values.detach().cpu().numpy()
    parity = batch.diagnostic.global_parity.detach().cpu().numpy()
    pieces = [local.reshape(-1), pair.reshape(-1), parity.reshape(-1)]
    flat = np.concatenate([piece for piece in pieces if piece.size])
    return {
        "local_values": local.tolist(),
        "pair_values": pair.tolist(),
        "global_parity": parity.tolist(),
        "total_rms": float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    step9b_path = args.step9b_config.expanduser().resolve()
    bundle_dir = args.deployment_bundle_dir.expanduser().resolve()
    config = read_json(config_path)
    step9b = read_json(step9b_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_DRY_RUN_EXECUTION":
        raise RuntimeError("Step-9C dry-run contract is not frozen")
    if config["scientific_boundaries"]["physical_qpu_execution"]:
        raise RuntimeError("Step-9C may not execute physical QPU hardware")
    verify_step9b_contract(step9b, config)
    if git_blob_sha(step9b_path) != config["source_step9b"]["config_git_blob_sha"]:
        raise RuntimeError("Step-9B config Git-blob identity changed")

    device = resolve_device(args.device)
    deployment = config["deployment_bundle"]
    ensemble = load_frozen_deployment_ensemble(bundle_dir, deployment, device=device)
    backend_config = config["simulator_backend"]
    backend = build_local_backend(backend_config)
    sampler = build_local_sampler(backend, backend_config)
    shots = int(backend_config["shots"])

    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": SCHEMA,
        "config_sha256": sha256_file(config_path),
        "step9b_config_git_blob_sha": git_blob_sha(step9b_path),
        "deployment_bundle_id": ensemble.bundle_id,
        "checkpoint_hashes": ensemble.checkpoint_hashes,
        "backend_seed": int(backend_config["backend_seed"]),
        "sampler_seed": int(backend_config["sampler_seed"]),
        "shots": shots,
        "cases": [str(row["case_id"]) for row in config["dry_run_cases"]],
    }
    run_id = "dryrun_" + hashlib.sha256(
        (canonical_json(identity) + uuid.uuid4().hex).encode("utf-8")
    ).hexdigest()[:24]
    output = output_parent / run_id
    staging = output_parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    case_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    try:
        cases = build_dry_run_cases(config)
        for case in cases:
            print(f"\nStep 9C executing {case.case_id} through frozen Step-9B path", flush=True)
            acquisition = acquire_paired_diagnostics(
                case.reference_circuit,
                case.observed_circuit,
                backend,
                sampler,
                initial_layout=case.initial_layout,
                shots=shots,
                optimization_level=int(backend_config["optimization_level"]),
                seed_transpiler=int(backend_config["seed_transpiler"]),
                device=device,
            )
            verify_acquisition_structure(acquisition, requested_shots=shots)
            if tuple(acquisition.initial_layout) != tuple(case.initial_layout):
                raise RuntimeError(f"Step-9C layout contract mismatch for {case.case_id}")

            training_batch = _training_adapter_batch(case, acquisition, device)
            assert_training_hardware_batch_equivalent(acquisition.model_batch, training_batch)
            prediction = predict_frozen_ensemble(ensemble, acquisition.model_batch)
            diagnostic = _diagnostic_summary(acquisition.model_batch)

            case_row = {
                "case_id": case.case_id,
                "family": case.family,
                "n_qubits": case.reference_circuit.num_qubits,
                "initial_layout": list(case.initial_layout),
                "injected_mechanism_report_only": case.injected_mechanism_report_only,
                "strength": case.strength,
                "requested_shots_per_program": shots,
                "program_order": list(acquisition.counts_by_program),
                "realized_shots_by_program": {
                    key: int(sum(value.values()))
                    for key, value in acquisition.counts_by_program.items()
                },
                "counts_by_program": acquisition.counts_by_program,
                "tensor_equivalence_with_step7_training_adapter": True,
                "meas_register_survived_transpilation": True,
                "diagnostic": diagnostic,
                "prediction_report_only": prediction,
            }
            case_rows.append(case_row)
            prediction_rows.append(
                {
                    "case_id": case.case_id,
                    "injected_mechanism_report_only": case.injected_mechanism_report_only,
                    "mean_effect_logit": prediction["mean_effect_logit"],
                    "effect_probability": prediction["effect_probability"],
                    "effect_present": prediction["effect_present"],
                    "mechanism_prediction": prediction["mechanism_prediction"],
                    "mechanism_code": prediction["mechanism_code"],
                    "prediction_correctness_is_gate": False,
                }
            )
            print(
                f"{case.case_id}: acquisition PASS | tensor equivalence PASS | "
                f"effect={prediction['effect_present']} mechanism={prediction['mechanism_prediction']} (report only)",
                flush=True,
            )

        with (staging / "case_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in case_rows:
                handle.write(canonical_json(row) + "\n")
        with (staging / "inference_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)

        gates = {
            "step9b_identity_verified": True,
            "deployment_bundle_hashes_verified": True,
            "all_six_programs_execute_per_case": True,
            "meas_register_survives_transpilation": True,
            "realized_shots_equal_requested_per_program": True,
            "hardware_path_matches_step7_training_adapter_tensors": True,
            "all_three_frozen_checkpoints_load": len(ensemble.models) == 3,
            "all_checkpoint_parameter_counts_match": True,
            "all_model_outputs_finite": True,
            "prediction_correctness_is_not_a_gate": True,
        }
        if gates != config["pass_gates"]:
            raise RuntimeError("Step-9C realized gate set differs from frozen pass-gate contract")

        summary = {
            "schema": SCHEMA,
            "status": "STEP9C_HARDWARE_PATH_DRY_RUN_PASS",
            "run_id": run_id,
            "identity": identity,
            "device": str(device),
            "backend_class": type(backend).__name__,
            "sampler_class": type(sampler).__name__,
            "case_count": len(case_rows),
            "pass_gates": gates,
            "prediction_correctness_used_for_selection": False,
            "physical_qpu_executed": False,
            "new_training_performed": False,
            "model_weights_changed": False,
            "deployment_threshold_changed": False,
        }
        atomic_json(staging / "dry_run_summary.json", summary)
        files = ["case_results.jsonl", "inference_predictions.csv", "dry_run_summary.json"]
        complete = {
            **summary,
            "file_hashes": {name: sha256_file(staging / name) for name in files},
            "required_files_present": True,
            "step9d_exploratory_qpu_pilot_unlocked": True,
        }
        atomic_json(staging / "dry_run_complete.json", complete)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 9C HARDWARE-PATH DRY RUN PASS\n")
    print(f"Run: {run_id}")
    print(f"Deployment bundle: {ensemble.bundle_id}")
    print(f"Cases: {len(case_rows)}")
    print("Six-program acquisition: PASS")
    print("Step-7 training/hardware tensor equivalence: PASS")
    print("Frozen checkpoint loading/inference: PASS")
    print("Prediction correctness used as gate: NO")
    print("Physical QPU executed: NO")
    print("Step 9D exploratory QPU pilot unlocked: YES")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
