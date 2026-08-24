#!/usr/bin/env python3
"""Plan or explicitly execute the frozen Step-11 exploratory IBM-QPU transfer pilot.

Step 11 is exploratory-only because the full simulator gate remains unmet.  The
primary model is fixed before QPU access to the Step-10C warm ensemble.  The
Step-9A ensemble is evaluated as a paired report-only baseline on the exact same
QPU counts and can never replace the primary candidate after seeing hardware.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from qiskit import qpy

from triqto.hardware.dry_run import (
    FrozenDeploymentEnsemble,
    load_frozen_deployment_ensemble,
    sha256_file,
)
from triqto.hardware.qpu_pilot import (
    batch_and_prediction_from_case_counts,
    best_connected_three_qubit_chain,
    build_pilot_cases,
    compile_pilot_programs,
    descriptive_pilot_metrics,
    select_backend_and_chain,
    split_sampler_results,
)
from triqto.step7.model import Step7DiagnosticModel


SCHEMA = "triqto.v0_2.step11_exploratory_ibm_transfer_pilot.v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "v0_2" / "step11_exploratory_ibm_transfer_pilot.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step11_exploratory_ibm_transfer_pilot")
DEFAULT_STEP10C = Path(
    "/workspace/triqto-data/step10c_crashsafe_long_horizon/benchmark_f9478da45d68795655259054"
)
DEFAULT_STEP10D = Path(
    "/workspace/triqto-data/step10d_final_simulator_lr_refinement/benchmark_1455864a09de8804a7e7958a"
)
DEFAULT_STEP9A = Path(
    "/workspace/triqto-data/step9a_deployment_bundle/deploy_ac536a74b2f8dd571d353a12"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--step10c-benchmark-dir", type=Path, default=DEFAULT_STEP10C)
    parser.add_argument("--step10d-benchmark-dir", type=Path, default=DEFAULT_STEP10D)
    parser.add_argument("--step9a-bundle-dir", type=Path, default=DEFAULT_STEP9A)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--backend-name", type=str)
    parser.add_argument("--instance-name", type=str)
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--execute-physical-qpu", action="store_true")
    parser.add_argument("--confirmation-token", type=str)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def verify_frozen_versions(config: Mapping[str, Any]) -> dict[str, str]:
    expected = config["software_environment"]
    actual = {
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        "qiskit_ibm_runtime": importlib.metadata.version("qiskit-ibm-runtime"),
    }
    for key, observed in actual.items():
        wanted = str(expected[key])
        if observed != wanted:
            raise RuntimeError(
                f"Step-11 software drift: {key}={observed}, expected {wanted}. "
                f"Use the frozen {expected['intended_environment']} environment."
            )
    return actual


def _assert_frozen_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected Step-11 schema")
    if config.get("status") != "FROZEN_BEFORE_PHYSICAL_QPU_EXECUTION":
        raise RuntimeError("Step-11 contract is not frozen before QPU execution")
    if config["source_step10d"]["primary_hardware_candidate"] != "step10c_warm_start":
        raise RuntimeError("Step-11 primary candidate drifted from Step-10D decision")
    if bool(config["source_step10d"]["further_simulator_tuning_before_hardware_permitted"]):
        raise RuntimeError("Step-11 cannot follow a contract that still permits simulator tuning")
    if config["primary_candidate"]["name"] != "step10c_warm_start":
        raise RuntimeError("Step-11 primary model must be Step-10C warm-start")
    if config["paired_baseline"]["name"] != "step9a_deployment_ensemble":
        raise RuntimeError("Step-11 paired baseline must remain Step-9A")
    if config["primary_candidate"]["architecture"] != "late_concat":
        raise RuntimeError("Step-11 primary architecture drift")
    if int(config["primary_candidate"]["expected_trainable_parameter_count"]) != 453829:
        raise RuntimeError("Step-11 primary parameter count drift")
    if list(config["primary_candidate"]["mechanism_classes"]) != [
        "rz_drift", "rx_overrotation", "ry_overrotation"
    ]:
        raise RuntimeError("Step-11 mechanism class order drift")
    if int(config["execution"]["case_count"]) != 12:
        raise RuntimeError("Step-11 case count must remain 12")
    if int(config["execution"]["programs_per_case"]) != 6:
        raise RuntimeError("Step-11 programs/case must remain 6")
    if int(config["execution"]["total_programs"]) != 72:
        raise RuntimeError("Step-11 total program count must remain 72")
    if int(config["execution"]["shots_per_program"]) != 4096:
        raise RuntimeError("Step-11 shots/program must remain 4096")
    if int(config["execution"]["max_execution_time_seconds"]) != 300:
        raise RuntimeError("Step-11 QPU execution cap must remain 300 seconds")
    for key in ("measurement_mitigation", "dynamical_decoupling", "twirling"):
        if bool(config["execution"][key]):
            raise RuntimeError(f"Step-11 frozen execution boundary relaxed: {key}")
    boundaries = config["scientific_boundaries"]
    forbidden_true = (
        "confirmatory_claim",
        "new_training",
        "model_weight_change",
        "architecture_change",
        "threshold_change",
        "checkpoint_selection",
        "shot_count_selection_from_qpu_results",
        "backend_selection_from_model_predictions",
        "measurement_mitigation_adaptation",
        "qpu_results_used_for_tuning",
        "qpu_results_used_for_retroactive_model_selection",
    )
    if any(bool(boundaries[name]) for name in forbidden_true):
        raise RuntimeError("Step-11 scientific boundary was relaxed")
    if not bool(boundaries["exploratory_only"]):
        raise RuntimeError("Step-11 must remain explicitly exploratory")
    if not bool(boundaries["primary_candidate_may_not_change_after_qpu"]):
        raise RuntimeError("Step-11 primary candidate must be immutable after QPU")


def _verify_step10d_reference(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    complete = read_json(path / "benchmark_complete.json")
    decision = read_json(path / "hardware_candidate_decision.json")
    expected = config["source_step10d"]
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("Step-10D reference benchmark is incomplete")
    if complete.get("benchmark_id") != expected["benchmark_id"]:
        raise RuntimeError("Step-10D benchmark identity mismatch")
    if complete.get("primary_hardware_candidate") != "step10c_warm_start":
        raise RuntimeError("Step-10D completion does not select Step-10C warm for hardware")
    if decision.get("primary_hardware_candidate") != "step10c_warm_start":
        raise RuntimeError("Step-10D decision does not select Step-10C warm for hardware")
    if bool(decision.get("further_simulator_tuning_before_hardware_permitted", True)):
        raise RuntimeError("Step-10D decision still permits simulator tuning")
    if bool(complete.get("qpu_executed", True)):
        raise RuntimeError("unexpected QPU execution in Step-10D reference")
    if bool(complete.get("step10c_outer_accessed", True)) or bool(complete.get("new_outer_accessed", True)):
        raise RuntimeError("Step-10D reference violated outer boundary")
    expected_decision_hash = complete.get("file_hashes", {}).get("hardware_candidate_decision.json")
    if expected_decision_hash is None or sha256_file(path / "hardware_candidate_decision.json") != expected_decision_hash:
        raise RuntimeError("Step-10D decision hash mismatch")
    return complete


def _load_step10c_primary(
    benchmark_dir: Path,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> FrozenDeploymentEnsemble:
    benchmark_dir = benchmark_dir.expanduser().resolve()
    complete = read_json(benchmark_dir / "benchmark_complete.json")
    candidate = config["primary_candidate"]
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("Step-10C reference benchmark is incomplete")
    if complete.get("benchmark_id") != candidate["benchmark_id"]:
        raise RuntimeError("Step-10C benchmark identity mismatch")
    if complete.get("architecture") != "late_concat":
        raise RuntimeError("Step-10C architecture mismatch")
    if int(complete.get("trainable_parameter_count", -1)) != 453829:
        raise RuntimeError("Step-10C parameter count mismatch")
    if bool(complete.get("qpu_executed", True)):
        raise RuntimeError("Step-10C reference unexpectedly executed a QPU")

    model_selection = benchmark_dir / "model_selection.json"
    if sha256_file(model_selection) != str(candidate["model_selection_sha256"]):
        raise RuntimeError("Step-10C model-selection hash mismatch")
    selection = read_json(model_selection)
    realized_threshold = float(selection["ensemble_effect_thresholds"]["warm_start"])
    if realized_threshold != float(candidate["effect_threshold"]):
        raise RuntimeError("Step-10C ensemble effect-threshold mismatch")

    models: dict[int, Step7DiagnosticModel] = {}
    hashes: dict[str, str] = {}
    for seed_value in candidate["seeds"]:
        seed = int(seed_value)
        filename = str(candidate["checkpoint_names"][str(seed)])
        path = benchmark_dir / filename
        expected_hash = str(candidate["checkpoint_sha256"][filename])
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Step-10C primary checkpoint hash mismatch: {filename}")
        if complete.get("file_hashes", {}).get(filename) != expected_hash:
            raise RuntimeError(f"Step-10C benchmark-complete hash mismatch: {filename}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("initialization") != "warm_start" or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"Step-10C checkpoint metadata mismatch: {filename}")
        if payload.get("architecture") != "late_concat":
            raise RuntimeError(f"Step-10C checkpoint architecture mismatch: {filename}")
        if not bool(payload.get("selected_checkpoint_retention_eligible")):
            raise RuntimeError(f"Step-10C selected checkpoint is not retention-eligible: {filename}")
        model = Step7DiagnosticModel(variant="late_concat", initialization_seed=seed)
        model.load_state_dict(payload["state_dict"], strict=True)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != int(candidate["expected_trainable_parameter_count"]):
            raise RuntimeError(f"Step-10C checkpoint parameter count mismatch: {filename}")
        model.to(device)
        model.eval()
        models[seed] = model
        hashes[filename] = actual_hash

    return FrozenDeploymentEnsemble(
        models=models,
        seeds=tuple(int(v) for v in candidate["seeds"]),
        effect_threshold=float(candidate["effect_threshold"]),
        mechanism_classes=tuple(str(v) for v in candidate["mechanism_classes"]),
        checkpoint_hashes=hashes,
        bundle_id="step10c_warm_start__" + str(candidate["benchmark_id"]),
    )


def _load_step9a_baseline(
    bundle_dir: Path,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> FrozenDeploymentEnsemble:
    return load_frozen_deployment_ensemble(
        bundle_dir.expanduser().resolve(),
        config["paired_baseline"],
        device=device,
    )


def _select_open_instance(instances: Sequence[Any], requested_name: str | None) -> dict[str, Any]:
    open_instances = [
        dict(row)
        for row in instances
        if isinstance(row, Mapping) and str(row.get("plan", "")).strip().lower() == "open"
    ]
    if requested_name:
        matches = [
            row for row in open_instances
            if requested_name in {str(row.get("name", "")), str(row.get("crn", ""))}
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"--instance-name {requested_name!r} did not identify exactly one Open Plan instance"
            )
        return matches[0]
    if len(open_instances) != 1:
        names = [str(row.get("name", row.get("crn", "unknown"))) for row in open_instances]
        raise RuntimeError(
            "Step-11 requires one explicit Open Plan instance; "
            f"found {len(open_instances)}: {names}. Re-run with --instance-name <name>."
        )
    return open_instances[0]


def _make_open_service(instance: Mapping[str, Any]) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    crn = str(instance.get("crn", "")).strip()
    if not crn:
        raise RuntimeError("Open Plan instance has no CRN")
    return QiskitRuntimeService(instance=crn)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _backend_snapshot(backend: Any, candidate: Any, ranking: list[dict[str, Any]], instance: Mapping[str, Any]) -> dict[str, Any]:
    status = backend.status()
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
        "instance": {
            "crn": str(instance.get("crn", "")),
            "name": str(instance.get("name", "")),
            "plan": str(instance.get("plan", "")),
        },
    }


def _diagnostic_rms(batch: Any) -> float:
    pieces = [
        batch.diagnostic.local_values.detach().cpu().numpy().reshape(-1),
        batch.diagnostic.pair_values.detach().cpu().numpy().reshape(-1),
        batch.diagnostic.global_parity.detach().cpu().numpy().reshape(-1),
    ]
    flat = np.concatenate([piece for piece in pieces if piece.size])
    return float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0


def _compiled_metadata_sha(metadata: Sequence[Mapping[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(list(metadata)).encode("utf-8")).hexdigest()


def _plan_identity(
    *,
    config_path: Path,
    versions: Mapping[str, str],
    primary: FrozenDeploymentEnsemble,
    baseline: FrozenDeploymentEnsemble,
    snapshot: Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "config_sha256": sha256_file(config_path),
        "software_versions": dict(versions),
        "primary_candidate": "step10c_warm_start",
        "primary_bundle_id": primary.bundle_id,
        "primary_checkpoint_hashes": dict(primary.checkpoint_hashes),
        "primary_effect_threshold": float(primary.effect_threshold),
        "baseline_candidate": "step9a_deployment_ensemble",
        "baseline_bundle_id": baseline.bundle_id,
        "baseline_checkpoint_hashes": dict(baseline.checkpoint_hashes),
        "baseline_effect_threshold": float(baseline.effect_threshold),
        "instance_crn": str(snapshot["instance"]["crn"]),
        "instance_name": str(snapshot["instance"]["name"]),
        "instance_plan": str(snapshot["instance"]["plan"]),
        "backend_name": str(snapshot["backend_name"]),
        "backend_version": str(snapshot["backend_version"]),
        "properties_last_update_date": snapshot["properties_last_update_date"],
        "physical_chain": list(snapshot["selected_chain"]["physical_chain"]),
        "compiled_program_metadata_sha256": _compiled_metadata_sha(metadata),
    }


def make_plan(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    step10c_dir: Path,
    step10d_dir: Path,
    step9a_dir: Path,
    output_parent: Path,
    backend_name: str | None,
    instance_name: str | None,
    device: torch.device,
) -> Path:
    _verify_step10d_reference(step10d_dir, config)
    primary = _load_step10c_primary(step10c_dir, config, device=device)
    baseline = _load_step9a_baseline(step9a_dir, config, device=device)
    versions = verify_frozen_versions(config)

    from qiskit_ibm_runtime import QiskitRuntimeService

    discovery = QiskitRuntimeService()
    instance = _select_open_instance(list(discovery.instances()), instance_name)
    if str(instance.get("plan", "")).strip().lower() != "open":
        raise RuntimeError("Step-11 paid-plan execution is forbidden")
    service = _make_open_service(instance)
    backend, chain, ranking = select_backend_and_chain(
        service,
        backend_name=backend_name,
        gate_preference=config["backend_selection"]["two_qubit_gate_preference"],
    )
    snapshot = _backend_snapshot(backend, chain, ranking, instance)
    cases = build_pilot_cases(config, chain.physical_chain)
    programs, metadata = compile_pilot_programs(
        cases,
        backend,
        optimization_level=int(config["execution"]["optimization_level"]),
        seed_transpiler=int(config["execution"]["seed_transpiler"]),
        require_no_routing_permutation=bool(config["execution"]["require_no_routing_permutation"]),
    )
    if len(cases) != int(config["execution"]["case_count"]):
        raise RuntimeError("Step-11 case count drift")
    if len(programs) != int(config["execution"]["total_programs"]):
        raise RuntimeError("Step-11 compiled-program count drift")

    identity = _plan_identity(
        config_path=config_path,
        versions=versions,
        primary=primary,
        baseline=baseline,
        snapshot=snapshot,
        metadata=metadata,
    )
    plan_id = "qpuplan_" + hashlib.sha256(
        (canonical_json(identity) + uuid.uuid4().hex).encode("utf-8")
    ).hexdigest()[:24]
    output = output_parent / plan_id
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": SCHEMA,
        "status": "STEP11_QPU_PLAN_READY_NOT_SUBMITTED",
        "plan_id": plan_id,
        "identity": identity,
        "primary_candidate": "step10c_warm_start",
        "paired_baseline": "step9a_deployment_ensemble",
        "shots_per_program": int(config["execution"]["shots_per_program"]),
        "case_count": len(cases),
        "program_count": len(programs),
        "total_executions": int(config["execution"]["total_executions"]),
        "max_execution_time_seconds": int(config["execution"]["max_execution_time_seconds"]),
        "single_sampler_job": True,
        "physical_qpu_submitted": False,
        "exploratory_only": True,
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

    print("\nTRIQTO STEP 11 EXPLORATORY IBM-QPU PLAN READY — NO QPU SUBMITTED\n")
    print(f"Plan: {plan_id}")
    print(f"IBM instance: {instance.get('name', '')} | plan={instance.get('plan', '')}")
    print(f"Backend: {snapshot['backend_name']} version={snapshot['backend_version']}")
    print(f"Physical chain: {snapshot['selected_chain']['physical_chain']}")
    print(f"2Q gate: {snapshot['selected_chain']['two_qubit_gate']}")
    print(f"2Q errors: {snapshot['selected_chain']['edge_errors']}")
    print(f"Readout errors: {snapshot['selected_chain']['readout_errors']}")
    print(f"Cases: {len(cases)} | programs: {len(programs)} | shots/program: {config['execution']['shots_per_program']}")
    print(f"Total circuit shots: {config['execution']['total_executions']}")
    print(f"Max QPU execution time: {config['execution']['max_execution_time_seconds']} s")
    print("Primary model: step10c_warm_start")
    print("Paired baseline: step9a_deployment_ensemble (same counts, report only)")
    print(f"Frozen software: {versions}")
    print("Physical QPU submitted: NO")
    print(f"Plan file: {output / 'pilot_plan.json'}")
    return output / "pilot_plan.json"


def _targeted_phase_result(rows: Sequence[Mapping[str, Any]], model_key: str) -> dict[str, Any]:
    phase = [
        row for row in rows
        if row["family"] == "phase_interference" and bool(row["expected_effect"])
    ]
    if len(phase) != 3:
        raise RuntimeError(f"expected exactly three distorted phase cases, found {len(phase)}")
    correct = sum(
        str(row[model_key]["mechanism_prediction"]) == str(row["expected_mechanism"])
        for row in phase
    )
    if correct == 3:
        interpretation = "strong exploratory targeted-repair transfer signal"
    elif correct == 2:
        interpretation = "partial exploratory targeted-repair transfer signal"
    else:
        interpretation = "weak_or_absent exploratory targeted-repair transfer signal"
    return {
        "case_count": 3,
        "mechanism_correct_count": int(correct),
        "mechanism_accuracy": float(correct / 3.0),
        "predeclared_interpretation": interpretation,
    }


def _prediction_csv_rows(rows: Sequence[Mapping[str, Any]], model_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = row[model_key]
        out.append(
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "condition": row["condition"],
                "expected_effect_report_only": row["expected_effect"],
                "expected_mechanism_report_only": row["expected_mechanism"],
                "effect_probability": pred["effect_probability"],
                "effect_threshold": pred["effect_threshold"],
                "effect_present": pred["effect_present"],
                "mechanism_prediction": pred["mechanism_prediction"],
                "diagnostic_rms": row["diagnostic_rms"],
            }
        )
    return out


def _metrics_rows(rows: Sequence[Mapping[str, Any]], model_key: str) -> list[dict[str, Any]]:
    return [
        {
            "expected_effect": row["expected_effect"],
            "expected_mechanism": row["expected_mechanism"],
            "prediction": row[model_key],
        }
        for row in rows
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def execute_plan(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    step10c_dir: Path,
    step10d_dir: Path,
    step9a_dir: Path,
    plan_file: Path,
    instance_name: str | None,
    device: torch.device,
) -> Path:
    plan_file = plan_file.expanduser().resolve()
    output = plan_file.parent
    if (output / "physical_access_started.json").exists():
        raise RuntimeError(
            "this Step-11 plan has already started physical-QPU access; create a new plan for any rerun"
        )
    plan = read_json(plan_file)
    snapshot = read_json(output / "backend_snapshot.json")
    if plan.get("schema") != SCHEMA or plan.get("status") != "STEP11_QPU_PLAN_READY_NOT_SUBMITTED":
        raise RuntimeError("Step-11 plan is not ready/not-submitted")
    if plan.get("primary_candidate") != "step10c_warm_start":
        raise RuntimeError("saved Step-11 plan primary candidate drift")

    _verify_step10d_reference(step10d_dir, config)
    primary = _load_step10c_primary(step10c_dir, config, device=device)
    baseline = _load_step9a_baseline(step9a_dir, config, device=device)
    versions = verify_frozen_versions(config)

    from qiskit_ibm_runtime import QiskitRuntimeService

    discovery = QiskitRuntimeService()
    instance = _select_open_instance(list(discovery.instances()), instance_name)
    if str(instance.get("crn", "")) != str(plan["identity"]["instance_crn"]):
        raise RuntimeError("active Open Plan instance CRN does not match saved Step-11 plan")
    if str(instance.get("plan", "")).strip().lower() != "open":
        raise RuntimeError("Step-11 execution is not bound to Open Plan")
    if plan["identity"]["software_versions"] != versions:
        raise RuntimeError("Step-11 software versions changed since planning")
    if plan["identity"]["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Step-11 config changed since planning; create a fresh plan")
    if plan["identity"]["primary_checkpoint_hashes"] != primary.checkpoint_hashes:
        raise RuntimeError("Step-11 primary model hashes changed since planning")
    if plan["identity"]["baseline_checkpoint_hashes"] != baseline.checkpoint_hashes:
        raise RuntimeError("Step-11 baseline model hashes changed since planning")

    service = _make_open_service(instance)
    backend = service.backend(str(snapshot["backend_name"]))
    current_chain = best_connected_three_qubit_chain(
        backend,
        gate_preference=config["backend_selection"]["two_qubit_gate_preference"],
    )
    if list(current_chain.physical_chain) != list(snapshot["selected_chain"]["physical_chain"]):
        raise RuntimeError("best calibrated physical chain changed since planning; create a fresh plan")
    current_snapshot = _backend_snapshot(backend, current_chain, [current_chain.as_dict()], instance)
    if current_snapshot["backend_version"] != snapshot["backend_version"]:
        raise RuntimeError("backend version changed since planning; create a fresh plan")
    if current_snapshot["properties_last_update_date"] != snapshot["properties_last_update_date"]:
        raise RuntimeError("backend calibration timestamp changed since planning; create a fresh plan")

    cases = build_pilot_cases(config, current_chain.physical_chain)
    programs, metadata = compile_pilot_programs(
        cases,
        backend,
        optimization_level=int(config["execution"]["optimization_level"]),
        seed_transpiler=int(config["execution"]["seed_transpiler"]),
        require_no_routing_permutation=bool(config["execution"]["require_no_routing_permutation"]),
    )
    if _compiled_metadata_sha(metadata) != plan["identity"]["compiled_program_metadata_sha256"]:
        raise RuntimeError("compiled circuits changed since Step-11 planning; create a fresh plan")

    with (output / "submitted_circuits.qpy").open("wb") as handle:
        qpy.dump(list(programs), handle)
    atomic_json(
        output / "physical_access_started.json",
        {
            "schema": SCHEMA,
            "status": "STEP11_PHYSICAL_QPU_ACCESS_STARTED",
            "plan_id": plan["plan_id"],
            "backend_name": str(backend.name),
            "program_count": len(programs),
            "shots_per_program": int(config["execution"]["shots_per_program"]),
            "total_executions": int(config["execution"]["total_executions"]),
            "primary_candidate": "step10c_warm_start",
            "exploratory_only": True,
            "started_unix": time.time(),
        },
    )

    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.max_execution_time = int(config["execution"]["max_execution_time_seconds"])
    print(
        "\nSTEP 11 PHYSICAL QPU ACCESS STARTED — exploratory transfer pilot; "
        "fixed Step-10C primary; no tuning.\n",
        flush=True,
    )
    job = sampler.run(list(programs), shots=int(config["execution"]["shots_per_program"]))
    job_id = str(job.job_id())
    print(f"IBM Runtime job: {job_id}", flush=True)
    result = job.result()
    counts = split_sampler_results(result, cases)

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_counts = counts[case.case_id]
        for program, values in case_counts.items():
            realized = sum(int(value) for value in values.values())
            if realized != int(config["execution"]["shots_per_program"]):
                raise RuntimeError(
                    f"realized shot mismatch for {case.case_id}/{program}: {realized}"
                )
        batch, primary_pred = batch_and_prediction_from_case_counts(
            case, case_counts, primary, device=device
        )
        _batch2, baseline_pred = batch_and_prediction_from_case_counts(
            case, case_counts, baseline, device=device
        )
        rows.append(
            {
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
                "step10c_prediction": primary_pred,
                "step9a_prediction": baseline_pred,
            }
        )

    with (output / "case_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")

    _write_csv(
        output / "predictions_step10c.csv",
        _prediction_csv_rows(rows, "step10c_prediction"),
    )
    _write_csv(
        output / "predictions_step9a.csv",
        _prediction_csv_rows(rows, "step9a_prediction"),
    )

    primary_metrics = descriptive_pilot_metrics(
        _metrics_rows(rows, "step10c_prediction"), primary.mechanism_classes
    )
    baseline_metrics = descriptive_pilot_metrics(
        _metrics_rows(rows, "step9a_prediction"), baseline.mechanism_classes
    )
    phase_primary = _targeted_phase_result(rows, "step10c_prediction")
    phase_baseline = _targeted_phase_result(rows, "step9a_prediction")

    distorted = [row for row in rows if bool(row["expected_effect"])]
    primary_better = 0
    baseline_better = 0
    both_correct = 0
    both_wrong = 0
    for row in distorted:
        truth = str(row["expected_mechanism"])
        p_ok = str(row["step10c_prediction"]["mechanism_prediction"]) == truth
        b_ok = str(row["step9a_prediction"]["mechanism_prediction"]) == truth
        if p_ok and not b_ok:
            primary_better += 1
        elif b_ok and not p_ok:
            baseline_better += 1
        elif p_ok and b_ok:
            both_correct += 1
        else:
            both_wrong += 1

    try:
        job_metrics = job.metrics()
    except Exception:
        job_metrics = None
    qpu_usage_seconds = None
    try:
        qpu_usage_seconds = float(job_metrics["usage"]["quantum_seconds"])
    except Exception:
        qpu_usage_seconds = None

    summary = {
        "schema": SCHEMA,
        "status": "STEP11_EXPLORATORY_IBM_TRANSFER_PILOT_COMPLETE",
        "plan_id": plan["plan_id"],
        "ibm_job_id": job_id,
        "primary_candidate": "step10c_warm_start",
        "paired_baseline": "step9a_deployment_ensemble",
        "exploratory_only": True,
        "simulator_full_gate_previously_met": False,
        "primary_targeted_phase": phase_primary,
        "baseline_targeted_phase_report_only": phase_baseline,
        "step10c_metrics": primary_metrics,
        "step9a_metrics_report_only": baseline_metrics,
        "paired_mechanism_correctness_on_9_distorted_cases": {
            "step10c_correct_step9a_wrong_count": int(primary_better),
            "step9a_correct_step10c_wrong_count": int(baseline_better),
            "both_correct_count": int(both_correct),
            "both_wrong_count": int(both_wrong),
        },
        "qpu_usage_seconds": qpu_usage_seconds,
        "job_metrics": job_metrics,
        "confirmatory_interpretation_allowed": False,
        "primary_candidate_change_after_qpu_allowed": False,
        "qpu_results_used_for_tuning": False,
    }
    atomic_json(output / "pilot_summary.json", summary)

    required = [str(v) for v in config["output"]["required_execution_files"] if str(v) != "pilot_complete.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Step-11 required output missing before completion: {missing}")
    hashes = {name: sha256_file(output / name) for name in required}
    complete = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "plan_id": plan["plan_id"],
        "ibm_job_id": job_id,
        "primary_candidate": "step10c_warm_start",
        "paired_baseline": "step9a_deployment_ensemble",
        "backend_name": str(backend.name),
        "physical_chain": list(current_chain.physical_chain),
        "qpu_usage_seconds": qpu_usage_seconds,
        "exploratory_only": True,
        "confirmatory_claim": False,
        "simulator_full_gate_previously_met": False,
        "qpu_results_used_for_tuning": False,
        "file_hashes": hashes,
    }
    atomic_json(output / "pilot_complete.json", complete)

    print("\nTRIQTO STEP 11 EXPLORATORY IBM TRANSFER PILOT COMPLETE\n")
    print(f"IBM Runtime job: {job_id}")
    print(f"Primary phase mechanism correctness: {phase_primary['mechanism_correct_count']}/3")
    print(f"Interpretation: {phase_primary['predeclared_interpretation']}")
    print(
        "Primary all-distorted mechanism accuracy: "
        f"{primary_metrics['distorted_mechanism_correct_count']}/{primary_metrics['distorted_case_count']} "
        f"= {primary_metrics['distorted_mechanism_accuracy']:.4f}"
    )
    print(
        "Paired mechanism cases — Step10C-only correct / Step9A-only correct: "
        f"{primary_better}/{baseline_better}"
    )
    print(f"QPU usage seconds: {qpu_usage_seconds}")
    print("Confirmatory claim allowed: NO")
    print("QPU-driven tuning allowed: NO")
    print(f"Output: {output}")
    return output / "pilot_complete.json"


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    _assert_frozen_contract(config)
    device = resolve_device(args.device)

    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)

    if args.execute_physical_qpu:
        expected = str(config["execution"]["explicit_confirmation_token"])
        if args.confirmation_token != expected:
            raise RuntimeError(
                f"physical QPU execution requires --confirmation-token {expected}"
            )
        if args.plan_file is None:
            raise RuntimeError("physical QPU execution requires --plan-file from a prior Step-11 plan")
        execute_plan(
            config_path=config_path,
            config=config,
            step10c_dir=args.step10c_benchmark_dir,
            step10d_dir=args.step10d_benchmark_dir,
            step9a_dir=args.step9a_bundle_dir,
            plan_file=args.plan_file,
            instance_name=args.instance_name,
            device=device,
        )
    else:
        if args.plan_file is not None:
            raise RuntimeError("--plan-file is only valid with --execute-physical-qpu")
        if args.confirmation_token is not None:
            raise RuntimeError("--confirmation-token is only valid with --execute-physical-qpu")
        make_plan(
            config_path=config_path,
            config=config,
            step10c_dir=args.step10c_benchmark_dir,
            step10d_dir=args.step10d_benchmark_dir,
            step9a_dir=args.step9a_bundle_dir,
            output_parent=output_parent,
            backend_name=args.backend_name,
            instance_name=args.instance_name,
            device=device,
        )


if __name__ == "__main__":
    main()
