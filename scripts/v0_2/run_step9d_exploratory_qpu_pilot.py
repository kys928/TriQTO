#!/usr/bin/env python3
"""Plan or explicitly execute the frozen Step-9D exploratory IBM-QPU pilot."""
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
from qiskit import qpy

from triqto.hardware.dry_run import load_frozen_deployment_ensemble, read_json, sha256_file
from triqto.hardware.qpu_pilot import (
    batch_and_prediction_from_case_counts,
    best_connected_three_qubit_chain,
    build_pilot_cases,
    compile_pilot_programs,
    descriptive_pilot_metrics,
    select_backend_and_chain,
    split_sampler_results,
)

SCHEMA = "triqto.v0_2.step9d_exploratory_qpu_pilot.v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v0_2/step9d_exploratory_qpu_pilot.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step9d_exploratory_qpu_pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deployment-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--backend-name", type=str)
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--execute-physical-qpu", action="store_true")
    parser.add_argument("--confirmation-token", type=str)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def backend_snapshot(backend: Any, candidate: Any, ranking: list[dict[str, Any]]) -> dict[str, Any]:
    status = backend.status()
    properties = None
    try:
        properties = backend.properties()
    except Exception:
        properties = None
    return {
        "backend_name": str(backend.name),
        "backend_version": str(getattr(backend, "backend_version", "unknown")),
        "num_qubits": int(getattr(backend, "num_qubits", 0)),
        "processor_type": getattr(backend, "processor_type", None),
        "status": {
            "operational": bool(getattr(status, "operational", False)),
            "status_msg": str(getattr(status, "status_msg", "")),
            "pending_jobs": int(getattr(status, "pending_jobs", 0)),
        },
        "properties_last_update_date": _iso(getattr(properties, "last_update_date", None)),
        "selected_chain": candidate.as_dict(),
        "candidate_ranking": ranking,
    }


def _diagnostic_rms(batch: Any) -> float:
    pieces = [
        batch.diagnostic.local_values.detach().cpu().numpy().reshape(-1),
        batch.diagnostic.pair_values.detach().cpu().numpy().reshape(-1),
        batch.diagnostic.global_parity.detach().cpu().numpy().reshape(-1),
    ]
    flat = np.concatenate([piece for piece in pieces if piece.size])
    return float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0


def _make_service() -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        return QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(
            "IBM Qiskit Runtime account is not configured. Configure QiskitRuntimeService before Step 9D."
        ) from exc


def _verify_execution_boundaries(config: dict[str, Any], args: argparse.Namespace) -> None:
    boundaries = config["scientific_boundaries"]
    forbidden = (
        "confirmatory_claim",
        "new_training",
        "model_weight_change",
        "architecture_change",
        "threshold_change",
        "checkpoint_selection",
        "shot_count_selection_from_qpu_results",
        "backend_selection_from_model_predictions",
        "measurement_mitigation_adaptation",
        "step8_confirmatory_reuse",
    )
    if any(bool(boundaries[name]) for name in forbidden):
        raise RuntimeError("Step-9D scientific boundary was relaxed")
    if not bool(boundaries["exploratory_only"]):
        raise RuntimeError("Step-9D must remain explicitly exploratory")
    if args.execute_physical_qpu:
        expected = str(config["execution"]["explicit_confirmation_token"])
        if args.confirmation_token != expected:
            raise RuntimeError(
                f"physical QPU execution requires --confirmation-token {expected}"
            )


def _plan_identity(config_path: Path, ensemble: Any, snapshot: dict[str, Any], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "config_sha256": sha256_file(config_path),
        "deployment_bundle_id": ensemble.bundle_id,
        "checkpoint_hashes": ensemble.checkpoint_hashes,
        "backend_name": snapshot["backend_name"],
        "backend_version": snapshot["backend_version"],
        "properties_last_update_date": snapshot["properties_last_update_date"],
        "physical_chain": snapshot["selected_chain"]["physical_chain"],
        "compiled_program_metadata_sha256": "sha256:" + hashlib.sha256(canonical_json(metadata).encode()).hexdigest(),
    }


def make_plan(
    *,
    config_path: Path,
    config: dict[str, Any],
    bundle_dir: Path,
    output_parent: Path,
    backend_name: str | None,
    device: torch.device,
) -> Path:
    service = _make_service()
    ensemble = load_frozen_deployment_ensemble(bundle_dir, config["deployment_bundle"], device=device)
    backend, candidate, ranking = select_backend_and_chain(
        service,
        backend_name=backend_name,
        gate_preference=config["backend_selection"]["two_qubit_gate_preference"],
    )
    snapshot = backend_snapshot(backend, candidate, ranking)
    cases = build_pilot_cases(config, candidate.physical_chain)
    programs, metadata = compile_pilot_programs(
        cases,
        backend,
        optimization_level=int(config["execution"]["optimization_level"]),
        seed_transpiler=int(config["execution"]["seed_transpiler"]),
        require_no_routing_permutation=bool(config["execution"]["require_no_routing_permutation"]),
    )
    if len(programs) != int(config["execution"]["total_programs"]):
        raise RuntimeError("Step-9D compiled-program count drift")
    identity = _plan_identity(config_path, ensemble, snapshot, metadata)
    plan_id = "qpuplan_" + hashlib.sha256(
        (canonical_json(identity) + uuid.uuid4().hex).encode()
    ).hexdigest()[:24]
    output = output_parent / plan_id
    output.mkdir(parents=True)
    plan = {
        "schema": SCHEMA,
        "status": "STEP9D_QPU_PLAN_READY_NOT_SUBMITTED",
        "plan_id": plan_id,
        "identity": identity,
        "shots_per_program": int(config["execution"]["shots_per_program"]),
        "case_count": len(cases),
        "program_count": len(programs),
        "total_executions": int(config["execution"]["total_executions"]),
        "single_sampler_job": True,
        "physical_qpu_submitted": False,
        "compiled_program_metadata": metadata,
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "condition": case.condition,
                "physical_layout": list(case.physical_layout),
                "affected_logical_qubit": case.affected_logical_qubit,
                "strength": case.strength,
                "expected_effect_report_only": case.expected_effect,
                "expected_mechanism_report_only": case.expected_mechanism,
            }
            for case in cases
        ],
    }
    atomic_json(output / "backend_snapshot.json", snapshot)
    atomic_json(output / "pilot_plan.json", plan)
    print("\nTRIQTO STEP 9D EXPLORATORY QPU PLAN READY — NO QPU SUBMITTED\n")
    print(f"Plan: {plan_id}")
    print(f"Backend: {snapshot['backend_name']} version={snapshot['backend_version']}")
    print(f"Physical chain: {snapshot['selected_chain']['physical_chain']}")
    print(f"2Q gate: {snapshot['selected_chain']['two_qubit_gate']}")
    print(f"2Q errors: {snapshot['selected_chain']['edge_errors']}")
    print(f"Readout errors: {snapshot['selected_chain']['readout_errors']}")
    print(f"Cases: {len(cases)} | programs: {len(programs)} | shots/program: {config['execution']['shots_per_program']}")
    print(f"Total Sampler executions: {config['execution']['total_executions']}")
    print("Physical QPU submitted: NO")
    print(f"Plan file: {output / 'pilot_plan.json'}")
    return output / "pilot_plan.json"


def execute_plan(
    *,
    config_path: Path,
    config: dict[str, Any],
    bundle_dir: Path,
    plan_file: Path,
    device: torch.device,
) -> Path:
    plan_file = plan_file.expanduser().resolve()
    output = plan_file.parent
    if (output / "physical_access_started.json").exists():
        raise RuntimeError("this Step-9D plan has already started physical-QPU access; create a new plan for a rerun")
    plan = read_json(plan_file)
    snapshot = read_json(output / "backend_snapshot.json")
    if plan.get("status") != "STEP9D_QPU_PLAN_READY_NOT_SUBMITTED":
        raise RuntimeError("Step-9D plan is not in ready/not-submitted state")

    service = _make_service()
    backend = service.backend(str(snapshot["backend_name"]))
    current_candidate = best_connected_three_qubit_chain(
        backend,
        gate_preference=config["backend_selection"]["two_qubit_gate_preference"],
    )
    if list(current_candidate.physical_chain) != list(snapshot["selected_chain"]["physical_chain"]):
        raise RuntimeError("best calibrated physical chain changed since planning; create a fresh Step-9D plan")
    current_snapshot = backend_snapshot(backend, current_candidate, [current_candidate.as_dict()])
    if current_snapshot["backend_version"] != snapshot["backend_version"]:
        raise RuntimeError("backend version changed since planning; create a fresh Step-9D plan")
    if current_snapshot["properties_last_update_date"] != snapshot["properties_last_update_date"]:
        raise RuntimeError("backend calibration timestamp changed since planning; create a fresh Step-9D plan")

    ensemble = load_frozen_deployment_ensemble(bundle_dir, config["deployment_bundle"], device=device)
    cases = build_pilot_cases(config, current_candidate.physical_chain)
    programs, metadata = compile_pilot_programs(
        cases,
        backend,
        optimization_level=int(config["execution"]["optimization_level"]),
        seed_transpiler=int(config["execution"]["seed_transpiler"]),
        require_no_routing_permutation=bool(config["execution"]["require_no_routing_permutation"]),
    )
    metadata_sha = "sha256:" + hashlib.sha256(canonical_json(metadata).encode()).hexdigest()
    if metadata_sha != plan["identity"]["compiled_program_metadata_sha256"]:
        raise RuntimeError("compiled circuits changed since Step-9D planning; create a fresh plan")

    with (output / "submitted_circuits.qpy").open("wb") as handle:
        qpy.dump(list(programs), handle)
    access_marker = {
        "schema": SCHEMA,
        "status": "STEP9D_PHYSICAL_QPU_ACCESS_STARTED",
        "plan_id": plan["plan_id"],
        "backend_name": str(backend.name),
        "program_count": len(programs),
        "shots_per_program": int(config["execution"]["shots_per_program"]),
        "total_executions": int(config["execution"]["total_executions"]),
        "exploratory_only": True,
    }
    atomic_json(output / "physical_access_started.json", access_marker)

    try:
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=backend)
        sampler.options.max_execution_time = int(config["execution"]["max_execution_time_seconds"])
        print("\nSTEP 9D PHYSICAL QPU ACCESS STARTED — exploratory pilot, frozen model, no tuning.\n", flush=True)
        job = sampler.run(list(programs), shots=int(config["execution"]["shots_per_program"]))
        job_id = str(job.job_id())
        print(f"IBM Runtime job: {job_id}", flush=True)
        result = job.result()
        counts = split_sampler_results(result, cases)

        rows: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for case in cases:
            case_counts = counts[case.case_id]
            for program, values in case_counts.items():
                realized = sum(int(value) for value in values.values())
                if realized != int(config["execution"]["shots_per_program"]):
                    raise RuntimeError(f"realized shot mismatch for {case.case_id}/{program}")
            batch, prediction = batch_and_prediction_from_case_counts(
                case, case_counts, ensemble, device=device
            )
            row = {
                "case_id": case.case_id,
                "family": case.family,
                "condition": case.condition,
                "physical_layout": list(case.physical_layout),
                "affected_logical_qubit": case.affected_logical_qubit,
                "strength": case.strength,
                "expected_effect": case.expected_effect,
                "expected_mechanism": case.expected_mechanism,
                "counts_by_program": case_counts,
                "diagnostic_rms": _diagnostic_rms(batch),
                "prediction": prediction,
            }
            rows.append(row)
            predictions.append(
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "condition": case.condition,
                    "expected_effect": case.expected_effect,
                    "expected_mechanism": case.expected_mechanism,
                    "effect_probability": prediction["effect_probability"],
                    "effect_present": prediction["effect_present"],
                    "mechanism_prediction": prediction["mechanism_prediction"],
                    "diagnostic_rms": row["diagnostic_rms"],
                }
            )

        with (output / "case_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)

        metrics = descriptive_pilot_metrics(rows, ensemble.mechanism_classes)
        job_metrics = None
        try:
            job_metrics = job.metrics()
        except Exception:
            job_metrics = None
        summary = {
            "schema": SCHEMA,
            "status": "STEP9D_EXPLORATORY_QPU_PILOT_COMPLETE",
            "plan_id": plan["plan_id"],
            "job_id": job_id,
            "backend_name": str(backend.name),
            "backend_version": str(getattr(backend, "backend_version", "unknown")),
            "physical_chain": list(current_candidate.physical_chain),
            "case_count": len(cases),
            "program_count": len(programs),
            "shots_per_program": int(config["execution"]["shots_per_program"]),
            "total_executions": int(config["execution"]["total_executions"]),
            "descriptive_metrics": metrics,
            "job_metrics": job_metrics,
            "exploratory_only": True,
            "confirmatory_claim": False,
            "model_weights_changed": False,
            "threshold_changed": False,
            "new_training_performed": False,
        }
        atomic_json(output / "pilot_summary.json", summary)
        file_names = [
            "pilot_plan.json",
            "backend_snapshot.json",
            "physical_access_started.json",
            "submitted_circuits.qpy",
            "case_results.jsonl",
            "predictions.csv",
            "pilot_summary.json",
        ]
        complete = {
            **summary,
            "file_hashes": {name: sha256_file(output / name) for name in file_names},
            "physical_qpu_executed": True,
            "result_may_not_be_described_as_confirmatory": True,
        }
        atomic_json(output / "pilot_complete.json", complete)
    except Exception as exc:
        atomic_json(
            output / "pilot_failed.json",
            {
                "schema": SCHEMA,
                "status": "STEP9D_EXPLORATORY_QPU_ATTEMPT_FAILED_AFTER_ACCESS_START",
                "plan_id": plan["plan_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exploratory_attempt_remains_in_audit_trail": True,
            },
        )
        raise

    print("\nTRIQTO STEP 9D EXPLORATORY QPU PILOT COMPLETE\n")
    print(f"Plan: {plan['plan_id']}")
    print(f"IBM Runtime job: {job_id}")
    print(f"Backend: {backend.name}")
    print(f"Physical chain: {list(current_candidate.physical_chain)}")
    print(f"Cases: {len(cases)} | programs: {len(programs)}")
    print(f"Descriptive distorted effect detection: {metrics['distorted_effect_detection_count']}/{metrics['distorted_case_count']}")
    print(f"Descriptive distorted mechanism accuracy: {metrics['distorted_mechanism_correct_count']}/{metrics['distorted_case_count']}")
    print(f"Clean effect false positives: {metrics['clean_effect_false_positive_count']}/{metrics['clean_case_count']}")
    print("Confirmatory interpretation: NO")
    print(f"Output: {output}")
    return output


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_PHYSICAL_QPU_EXECUTION":
        raise RuntimeError("Step-9D QPU pilot contract is not frozen")
    _verify_execution_boundaries(config, args)
    device = resolve_device(args.device)
    bundle_dir = args.deployment_bundle_dir.expanduser().resolve()
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)

    if args.execute_physical_qpu:
        if args.plan_file is None:
            raise RuntimeError("physical QPU execution requires --plan-file from a prior Step-9D plan")
        execute_plan(
            config_path=config_path,
            config=config,
            bundle_dir=bundle_dir,
            plan_file=args.plan_file,
            device=device,
        )
    else:
        if args.plan_file is not None:
            raise RuntimeError("--plan-file is only valid with --execute-physical-qpu")
        make_plan(
            config_path=config_path,
            config=config,
            bundle_dir=bundle_dir,
            output_parent=output_parent,
            backend_name=args.backend_name,
            device=device,
        )


if __name__ == "__main__":
    main()
