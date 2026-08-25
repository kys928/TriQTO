#!/usr/bin/env python3
"""Plan or explicitly execute the frozen Step-12 independent phase-generalization QPU test.

Step 12 keeps the Step-10C primary model and Step-9A paired baseline immutable.
It asks a new question with new phase-sensitive motifs, two bridge-unseen strengths,
changed layouts, and a physical backend that must differ from Step 11.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector, partial_trace

import run_step11_exploratory_ibm_transfer_pilot as step11
from triqto.hardware.qpu_pilot import (
    PilotCase,
    batch_and_prediction_from_case_counts,
    best_connected_three_qubit_chain,
    compile_pilot_programs,
    split_sampler_results,
)


SCHEMA = "triqto.v0_2.step12_independent_phase_generalization.v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "v0_2" / "step12_independent_phase_generalization.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step12_independent_phase_generalization")
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


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _assert_frozen_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected Step-12 schema")
    if config.get("status") != "FROZEN_BEFORE_PHYSICAL_QPU_EXECUTION":
        raise RuntimeError("Step-12 contract is not frozen before physical-QPU execution")
    if config["primary_candidate"]["name"] != "step10c_warm_start":
        raise RuntimeError("Step-12 primary candidate must remain Step-10C warm")
    if config["paired_baseline"]["name"] != "step9a_deployment_ensemble":
        raise RuntimeError("Step-12 paired baseline must remain Step-9A")
    if bool(config["primary_candidate"]["model_change_after_step11_allowed"]):
        raise RuntimeError("Step-12 may not change the primary model after Step 11")
    design = config["generalization_design"]
    if [str(v) for v in design["mechanisms"]] != ["rz_drift", "rx_overrotation", "ry_overrotation"]:
        raise RuntimeError("Step-12 mechanism order drift")
    if [float(v) for v in design["strengths"]] != [0.13, 0.27]:
        raise RuntimeError("Step-12 strength schedule drift")
    if int(design["total_cases"]) != 21 or int(design["total_distorted_cases"]) != 18:
        raise RuntimeError("Step-12 case matrix drift")
    if bool(design["step9d_anchor_matrix_reused"]):
        raise RuntimeError("Step-12 may not reuse the Step-9D anchor matrix")
    if bool(design["step10_bridge_motif_names_reused"]):
        raise RuntimeError("Step-12 may not reuse Step-10 bridge motif names")
    if bool(design["model_predictions_used_to_select_motifs_before_freeze"]):
        raise RuntimeError("Step-12 motifs may not be pre-screened by model predictions")
    if not bool(design["no_model_inference_allowed_during_plan_generation"]):
        raise RuntimeError("Step-12 plan generation must remain model-inference-free")
    execution = config["execution"]
    expected = {
        "shots_per_program": 4096,
        "case_count": 21,
        "programs_per_case": 6,
        "total_programs": 126,
        "total_executions": 516096,
        "optimization_level": 1,
        "seed_transpiler": 17121,
        "max_execution_time_seconds": 300,
    }
    for key, value in expected.items():
        if int(execution[key]) != value:
            raise RuntimeError(f"Step-12 frozen execution drift: {key}")
    for key in ("measurement_mitigation", "dynamical_decoupling", "twirling"):
        if bool(execution[key]):
            raise RuntimeError(f"Step-12 frozen execution boundary relaxed: {key}")
    backend = config["backend_selection"]
    excluded = {str(v) for v in backend["excluded_backend_names"]}
    if "ibm_kingston" not in excluded or not bool(backend["require_backend_different_from_step11"]):
        raise RuntimeError("Step-12 must exclude the Step-11 backend")
    boundaries = config["scientific_boundaries"]
    forbidden_true = (
        "new_training", "model_weight_change", "architecture_change", "threshold_change",
        "checkpoint_selection", "qpu_results_used_for_tuning",
        "qpu_results_used_for_retroactive_model_selection",
        "shot_count_selection_from_qpu_results", "backend_selection_from_model_predictions",
        "measurement_mitigation_adaptation", "step11_result_replacement_allowed",
        "full_triqto_confirmatory_claim", "rerun_replaces_first_step12_outcome",
    )
    if any(bool(boundaries[name]) for name in forbidden_true):
        raise RuntimeError("Step-12 scientific boundary was relaxed")
    if not bool(boundaries["narrow_component_generalization_claim_only_if_predeclared_gate_passes"]):
        raise RuntimeError("Step-12 narrow claim must be gated by the predeclared rule")


def _verify_step11_freeze(config: Mapping[str, Any]) -> str:
    expected = config["source_step11"]
    path = ROOT / str(expected["result_manifest_path"])
    if not path.is_file():
        raise RuntimeError("Step-11 frozen result manifest is missing from the checkout")
    manifest = read_json(path)
    if manifest.get("status") != "FROZEN_POSTOUTCOME":
        raise RuntimeError("Step-11 result is not frozen postoutcome")
    if manifest.get("plan_id") != expected["plan_id"] or manifest.get("ibm_job_id") != expected["ibm_job_id"]:
        raise RuntimeError("Step-11 frozen plan/job identity mismatch")
    if manifest.get("primary_candidate") != "step10c_warm_start":
        raise RuntimeError("Step-11 frozen primary candidate mismatch")
    if bool(manifest.get("qpu_results_used_for_tuning", True)):
        raise RuntimeError("Step-11 freeze unexpectedly reports QPU-driven tuning")
    if int(manifest["primary_predeclared_targeted_result"]["mechanism_correct_count"]) != 3:
        raise RuntimeError("Step-11 frozen targeted result mismatch")
    if str(manifest["hardware_identity"]["backend_name"]) != str(expected["backend_name"]):
        raise RuntimeError("Step-11 frozen backend identity mismatch")
    return step11.sha256_file(path)


def _apply_injection(circuit: QuantumCircuit, mechanism: str | None, strength: float | None, qubit: int) -> None:
    if mechanism is None:
        return
    if strength is None:
        raise ValueError("distorted Step-12 case requires a strength")
    if mechanism == "rz_drift":
        circuit.rz(float(strength), qubit)
    elif mechanism == "rx_overrotation":
        circuit.rx(float(strength), qubit)
    elif mechanism == "ry_overrotation":
        circuit.ry(float(strength), qubit)
    else:
        raise ValueError(f"unknown Step-12 mechanism {mechanism!r}")


def _build_motif(name: str, mechanism: str | None = None, strength: float | None = None) -> tuple[QuantumCircuit, int]:
    if name == "cz_echo_ramsey":
        qc = QuantumCircuit(2, name=name)
        affected = 0
        qc.h(0)
        qc.ry(0.43, 1)
        qc.cz(0, 1)
        qc.rz(1.11, 0)
        _apply_injection(qc, mechanism, strength, affected)
        qc.rx(-0.31, 1)
        qc.h(0)
        qc.cz(0, 1)
        qc.ry(0.22, 0)
        qc.h(1)
        return qc, affected
    if name == "dual_arm_recombination":
        qc = QuantumCircuit(2, name=name)
        affected = 1
        qc.h(0)
        qc.ry(-0.47, 1)
        qc.cx(0, 1)
        qc.rz(-0.91, 1)
        qc.rx(0.36, 0)
        _apply_injection(qc, mechanism, strength, affected)
        qc.cz(0, 1)
        qc.h(1)
        qc.cx(1, 0)
        qc.rz(0.28, 0)
        return qc, affected
    if name == "three_qubit_phase_fanout":
        # Uses only nearest-neighbor logical edges (q0-q1 and q1-q2).  With the
        # frozen reverse-chain layout this remains nearest-neighbor on the selected
        # physical three-qubit chain and therefore needs no routing permutation.
        qc = QuantumCircuit(3, name=name)
        affected = 1
        qc.ry(0.93, 0)
        qc.ry(0.83, 1)
        qc.h(2)
        qc.cx(0, 1)
        qc.cz(1, 0)
        qc.ry(0.25, 1)
        qc.rz(1.04, 1)
        qc.ry(0.53, 1)
        _apply_injection(qc, mechanism, strength, affected)
        qc.rz(-0.30, 0)
        qc.cx(0, 1)
        qc.ry(1.05, 0)
        qc.cx(1, 2)
        qc.ry(-0.91, 2)
        qc.cz(0, 1)
        qc.rz(-0.13, 2)
        qc.rx(-0.93, 2)
        qc.h(0)
        return qc, affected
    raise ValueError(f"unknown Step-12 motif {name!r}")


def _operation_signature(circuit: QuantumCircuit) -> list[str]:
    out: list[str] = []
    for item in circuit.data:
        name = str(item.operation.name)
        q = [int(circuit.find_bit(bit).index) for bit in item.qubits]
        if len(q) == 1:
            out.append(f"{name}:q{q[0]}")
        else:
            out.append(name + ":" + "-".join(f"q{value}" for value in q))
    return out


def _contains_subsequence(sequence: Sequence[str], subsequence: Sequence[str]) -> bool:
    if not subsequence:
        return True
    width = len(subsequence)
    return any(list(sequence[i:i + width]) == list(subsequence) for i in range(len(sequence) - width + 1))


def _layout_for_motif(name: str, chain: Sequence[int]) -> tuple[int, ...]:
    left, center, right = [int(v) for v in chain]
    if name == "cz_echo_ramsey":
        return (left, center)
    if name == "dual_arm_recombination":
        return (right, center)
    if name == "three_qubit_phase_fanout":
        return (right, center, left)
    raise ValueError(name)


def build_generalization_cases(config: Mapping[str, Any], physical_chain: Sequence[int]) -> tuple[PilotCase, ...]:
    chain = tuple(int(v) for v in physical_chain)
    if len(chain) != 3 or len(set(chain)) != 3:
        raise ValueError("Step-12 requires one unique three-qubit physical chain")
    design = config["generalization_design"]
    mechanisms = [str(v) for v in design["mechanisms"]]
    strengths = [float(v) for v in design["strengths"]]
    cases: list[PilotCase] = []
    forbidden_core = ["h:q0", "rz:q0", "h:q0", "cx:q0-q1"]
    for raw in design["motifs"]:
        name = str(raw["name"])
        reference, affected = _build_motif(name)
        signature = _operation_signature(reference)
        if signature != [str(v) for v in raw["reference_operation_signature"]]:
            raise RuntimeError(f"Step-12 motif signature drift: {name}")
        if _contains_subsequence(signature, forbidden_core):
            raise RuntimeError(f"Step-12 motif reuses the targeted Step-9D/Step-10 H-RZ-H-CX core: {name}")
        layout = _layout_for_motif(name, chain)
        if int(reference.num_qubits) != int(raw["n_qubits"]):
            raise RuntimeError(f"Step-12 motif qubit-count drift: {name}")
        if affected != int(raw["affected_logical_qubit"]):
            raise RuntimeError(f"Step-12 motif affected-qubit drift: {name}")
        cases.append(PilotCase(
            case_id=f"{name}__clean",
            family=name,
            condition="clean",
            reference_circuit=reference.copy(name=f"{name}__reference"),
            observed_circuit=reference.copy(name=f"{name}__clean"),
            logical_layout=tuple(range(reference.num_qubits)),
            physical_layout=layout,
            affected_logical_qubit=None,
            strength=None,
            expected_effect=False,
            expected_mechanism=None,
        ))
        for strength in strengths:
            strength_tag = f"s{int(round(100 * strength)):02d}"
            for mechanism in mechanisms:
                observed, observed_affected = _build_motif(name, mechanism, strength)
                if observed_affected != affected:
                    raise RuntimeError("Step-12 observed affected-qubit drift")
                cases.append(PilotCase(
                    case_id=f"{name}__{mechanism}__{strength_tag}",
                    family=name,
                    condition=mechanism,
                    reference_circuit=reference.copy(name=f"{name}__reference"),
                    observed_circuit=observed,
                    logical_layout=tuple(range(reference.num_qubits)),
                    physical_layout=layout,
                    affected_logical_qubit=affected,
                    strength=float(strength),
                    expected_effect=True,
                    expected_mechanism=mechanism,
                ))
    if len(cases) != int(config["execution"]["case_count"]):
        raise RuntimeError(f"Step-12 case count drift: {len(cases)}")
    return tuple(cases)


def _bloch_vector(circuit: QuantumCircuit) -> np.ndarray:
    state = Statevector.from_instruction(circuit)
    values: list[float] = []
    n = int(circuit.num_qubits)
    for q in range(n):
        traced = [idx for idx in range(n) if idx != q]
        rho = partial_trace(state, traced) if traced else state.to_operator()
        matrix = np.asarray(rho.data, dtype=np.complex128)
        b = matrix[0, 1]
        values.extend([
            float(2.0 * np.real(b)),
            float(-2.0 * np.imag(b)),
            float(np.real(matrix[0, 0] - matrix[1, 1])),
        ])
    return np.asarray(values, dtype=np.float64)


def model_blind_identifiability_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    design = config["generalization_design"]
    audit_cfg = design["model_blind_identifiability_audit"]
    strengths = [float(v) for v in design["strengths"]]
    low_strength = min(strengths)
    records: list[dict[str, Any]] = []
    all_low_delta: list[float] = []
    all_low_pairwise: list[float] = []
    for raw in design["motifs"]:
        name = str(raw["name"])
        reference, _affected = _build_motif(name)
        ref = _bloch_vector(reference)
        for strength in strengths:
            deltas: dict[str, np.ndarray] = {}
            for mechanism in design["mechanisms"]:
                observed, _ = _build_motif(name, str(mechanism), strength)
                delta = _bloch_vector(observed) - ref
                deltas[str(mechanism)] = delta
                norm = float(np.linalg.norm(delta))
                records.append({
                    "motif": name,
                    "strength": float(strength),
                    "kind": "delta_norm",
                    "mechanism": str(mechanism),
                    "value": norm,
                })
                if strength == low_strength:
                    all_low_delta.append(norm)
            mechanisms = [str(v) for v in design["mechanisms"]]
            for i, left in enumerate(mechanisms):
                for right in mechanisms[i + 1:]:
                    distance = float(np.linalg.norm(deltas[left] - deltas[right]))
                    records.append({
                        "motif": name,
                        "strength": float(strength),
                        "kind": "pairwise_mechanism_distance",
                        "left": left,
                        "right": right,
                        "value": distance,
                    })
                    if strength == low_strength:
                        all_low_pairwise.append(distance)
    min_delta = float(min(all_low_delta))
    min_pairwise = float(min(all_low_pairwise))
    delta_ok = min_delta >= float(audit_cfg["minimum_low_strength_delta_norm"])
    pairwise_ok = min_pairwise >= float(audit_cfg["minimum_low_strength_pairwise_mechanism_distance"])
    return {
        "schema": SCHEMA,
        "status": "PASS" if delta_ok and pairwise_ok else "FAIL",
        "uses_model_predictions": False,
        "uses_statevector_only": True,
        "low_strength": low_strength,
        "minimum_observed_low_strength_delta_norm": min_delta,
        "minimum_required_low_strength_delta_norm": float(audit_cfg["minimum_low_strength_delta_norm"]),
        "minimum_observed_low_strength_pairwise_mechanism_distance": min_pairwise,
        "minimum_required_low_strength_pairwise_mechanism_distance": float(audit_cfg["minimum_low_strength_pairwise_mechanism_distance"]),
        "records": records,
    }


def _select_backend_and_chain_excluding(service: Any, config: Mapping[str, Any]) -> tuple[Any, Any, list[dict[str, Any]]]:
    policy = config["backend_selection"]
    excluded = {str(v) for v in policy["excluded_backend_names"]}
    backends = [
        backend for backend in service.backends(simulator=False, operational=True, min_num_qubits=3)
        if str(backend.name) not in excluded
    ]
    if not backends:
        raise RuntimeError("Step-12 found no allowed physical backend after excluding Step-11 backend")
    ranked: list[tuple[Any, Any]] = []
    rejected: list[dict[str, Any]] = []
    for backend in backends:
        try:
            candidate = best_connected_three_qubit_chain(
                backend,
                gate_preference=policy["two_qubit_gate_preference"],
            )
            ranked.append((candidate, backend))
        except Exception as exc:
            rejected.append({"backend_name": str(getattr(backend, "name", "unknown")), "reason": str(exc)})
    if not ranked:
        raise RuntimeError("Step-12 found no complete calibrated three-qubit chain on allowed backends")
    ranked.sort(key=lambda pair: pair[0].score)
    candidate, backend = ranked[0]
    ranking = [pair[0].as_dict() for pair in ranked]
    if rejected:
        ranking.extend({"rejected": row} for row in rejected)
    if str(backend.name) in excluded:
        raise RuntimeError("Step-12 backend exclusion failed")
    return backend, candidate, ranking


def _snapshot(backend: Any, candidate: Any, ranking: list[dict[str, Any]], instance: Mapping[str, Any]) -> dict[str, Any]:
    snap = step11._backend_snapshot(backend, candidate, ranking, instance)
    snap["excluded_backend_names"] = [str(v) for v in sorted(set(str(x) for x in ["ibm_kingston"]))]
    snap["cross_backend_relative_to_step11"] = str(backend.name) != "ibm_kingston"
    return snap


def _compiled_metadata_sha(metadata: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(list(metadata))


def _case_design_sha(cases: Sequence[PilotCase]) -> str:
    rows = []
    for case in cases:
        rows.append({
            "case_id": case.case_id,
            "family": case.family,
            "condition": case.condition,
            "physical_layout": list(case.physical_layout),
            "affected_logical_qubit": case.affected_logical_qubit,
            "strength": case.strength,
            "expected_effect": case.expected_effect,
            "expected_mechanism": case.expected_mechanism,
            "reference_signature": _operation_signature(case.reference_circuit),
            "observed_signature": _operation_signature(case.observed_circuit),
        })
    return _sha256_json(rows)


def make_plan(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    step10c_dir: Path,
    step10d_dir: Path,
    step9a_dir: Path,
    output_parent: Path,
    instance_name: str | None,
    device: torch.device,
) -> Path:
    _assert_frozen_contract(config)
    source_step11_sha = _verify_step11_freeze(config)
    step11._verify_step10d_reference(step10d_dir, config)
    primary = step11._load_step10c_primary(step10c_dir, config, device=device)
    baseline = step11._load_step9a_baseline(step9a_dir, config, device=device)
    versions = step11.verify_frozen_versions(config)

    identifiability = model_blind_identifiability_audit(config)
    if identifiability["status"] != "PASS":
        raise RuntimeError("Step-12 model-blind identifiability audit failed; protocol may not execute")

    from qiskit_ibm_runtime import QiskitRuntimeService
    discovery = QiskitRuntimeService()
    instance = step11._select_open_instance(list(discovery.instances()), instance_name)
    if str(instance.get("plan", "")).strip().lower() != "open":
        raise RuntimeError("Step-12 planning is not bound to IBM Open Plan")
    service = step11._make_open_service(instance)
    backend, candidate, ranking = _select_backend_and_chain_excluding(service, config)
    if str(backend.name) == str(config["source_step11"]["backend_name"]):
        raise RuntimeError("Step-12 selected the Step-11 backend; cross-backend independence failed")
    snapshot = _snapshot(backend, candidate, ranking, instance)

    cases = build_generalization_cases(config, candidate.physical_chain)
    programs, metadata = compile_pilot_programs(
        cases,
        backend,
        optimization_level=int(config["execution"]["optimization_level"]),
        seed_transpiler=int(config["execution"]["seed_transpiler"]),
        require_no_routing_permutation=bool(config["execution"]["require_no_routing_permutation"]),
    )
    if len(programs) != int(config["execution"]["total_programs"]):
        raise RuntimeError("Step-12 compiled program count drift")

    plan_id = "step12plan_" + uuid.uuid4().hex[:24]
    output = output_parent.expanduser().resolve() / plan_id
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "backend_snapshot.json", snapshot)
    atomic_json(output / "identifiability_audit.json", identifiability)
    plan = {
        "schema": SCHEMA,
        "status": "STEP12_QPU_PLAN_READY_NOT_SUBMITTED",
        "plan_id": plan_id,
        "physical_qpu_submitted": False,
        "primary_candidate": "step10c_warm_start",
        "paired_baseline": "step9a_deployment_ensemble",
        "claim_scope": "narrow_cross_backend_phase_generalization_only",
        "identity": {
            "config_sha256": step11.sha256_file(config_path),
            "source_step11_manifest_sha256": source_step11_sha,
            "primary_checkpoint_hashes": primary.checkpoint_hashes,
            "baseline_checkpoint_hashes": baseline.checkpoint_hashes,
            "primary_effect_threshold": primary.effect_threshold,
            "instance_crn": str(instance.get("crn", "")),
            "instance_name": str(instance.get("name", "")),
            "instance_plan": str(instance.get("plan", "")),
            "software_versions": versions,
            "backend_name": str(backend.name),
            "backend_version": snapshot["backend_version"],
            "properties_last_update_date": snapshot["properties_last_update_date"],
            "physical_chain": list(candidate.physical_chain),
            "compiled_program_metadata_sha256": _compiled_metadata_sha(metadata),
            "case_design_sha256": _case_design_sha(cases),
            "identifiability_audit_sha256": _sha256_json(identifiability),
        },
        "case_count": len(cases),
        "distorted_case_count": sum(bool(case.expected_effect) for case in cases),
        "program_count": len(programs),
        "shots_per_program": int(config["execution"]["shots_per_program"]),
        "total_executions": int(config["execution"]["total_executions"]),
        "predeclared_support_gate": config["predeclared_analysis"]["support_gate"],
        "compiled_program_metadata": metadata,
    }
    atomic_json(output / "generalization_plan.json", plan)

    print("\nTRIQTO STEP 12 INDEPENDENT PHASE-GENERALIZATION PLAN READY — NO QPU SUBMITTED\n")
    print(f"Plan: {plan_id}")
    print(f"IBM instance: {instance.get('name')} | plan={instance.get('plan')}")
    print(f"Backend: {backend.name} version={snapshot['backend_version']} (Step-11 backend excluded)")
    print(f"Physical chain: {list(candidate.physical_chain)}")
    print(f"2Q gate: {candidate.two_qubit_gate}")
    print(f"2Q errors: {list(candidate.edge_errors)}")
    print(f"Readout errors: {list(candidate.readout_errors)}")
    print(f"Cases: {len(cases)} | distorted: {sum(case.expected_effect for case in cases)}")
    print(f"Programs: {len(programs)} | shots/program: {config['execution']['shots_per_program']}")
    print(f"Total circuit shots: {config['execution']['total_executions']}")
    print(f"Identifiability audit: {identifiability['status']}")
    print("Primary model: step10c_warm_start (unchanged)")
    print("Paired baseline: step9a_deployment_ensemble (same counts; report only)")
    print("Physical QPU submitted: NO")
    print(f"Plan file: {output / 'generalization_plan.json'}")
    return output / "generalization_plan.json"


def _prediction_rows(rows: Sequence[Mapping[str, Any]], model_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = row[model_key]
        out.append({
            "case_id": row["case_id"],
            "motif": row["family"],
            "condition": row["condition"],
            "strength": row["strength"],
            "expected_effect_report_only": row["expected_effect"],
            "expected_mechanism_report_only": row["expected_mechanism"],
            "effect_probability": pred["effect_probability"],
            "effect_threshold": pred["effect_threshold"],
            "effect_present": pred["effect_present"],
            "mechanism_prediction": pred["mechanism_prediction"],
            "diagnostic_rms": row["diagnostic_rms"],
        })
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _model_metrics(rows: Sequence[Mapping[str, Any]], model_key: str, mechanism_classes: Sequence[str]) -> dict[str, Any]:
    clean = [row for row in rows if not bool(row["expected_effect"])]
    distorted = [row for row in rows if bool(row["expected_effect"])]
    correct = [
        row for row in distorted
        if str(row[model_key]["mechanism_prediction"]) == str(row["expected_mechanism"])
    ]
    detected = [row for row in distorted if bool(row[model_key]["effect_present"])]
    false_pos = [row for row in clean if bool(row[model_key]["effect_present"])]
    by_mechanism: dict[str, dict[str, int]] = {}
    for mechanism in mechanism_classes:
        subset = [row for row in distorted if str(row["expected_mechanism"]) == str(mechanism)]
        by_mechanism[str(mechanism)] = {
            "correct": sum(str(row[model_key]["mechanism_prediction"]) == str(mechanism) for row in subset),
            "count": len(subset),
        }
    by_strength: dict[str, dict[str, int]] = {}
    for strength in sorted({float(row["strength"]) for row in distorted}):
        subset = [row for row in distorted if float(row["strength"]) == strength]
        by_strength[str(strength)] = {
            "correct": sum(str(row[model_key]["mechanism_prediction"]) == str(row["expected_mechanism"]) for row in subset),
            "count": len(subset),
        }
    by_motif: dict[str, dict[str, int]] = {}
    for motif in sorted({str(row["family"]) for row in rows}):
        subset = [row for row in distorted if str(row["family"]) == motif]
        by_motif[motif] = {
            "correct": sum(str(row[model_key]["mechanism_prediction"]) == str(row["expected_mechanism"]) for row in subset),
            "count": len(subset),
        }
    classes = [str(v) for v in mechanism_classes]
    confusion = {truth: {pred: 0 for pred in classes} for truth in classes}
    for row in distorted:
        confusion[str(row["expected_mechanism"])][str(row[model_key]["mechanism_prediction"])] += 1
    return {
        "clean_case_count": len(clean),
        "clean_effect_false_positive_count": len(false_pos),
        "distorted_case_count": len(distorted),
        "distorted_effect_detection_count": len(detected),
        "distorted_mechanism_correct_count": len(correct),
        "distorted_mechanism_accuracy": float(len(correct) / len(distorted)),
        "by_mechanism": by_mechanism,
        "by_strength": by_strength,
        "by_motif": by_motif,
        "mechanism_confusion_matrix": confusion,
    }


def evaluate_predeclared_gate(
    primary_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["predeclared_analysis"]["support_gate"]
    criteria: dict[str, bool] = {}
    criteria["total_mechanism_correct"] = int(primary_metrics["distorted_mechanism_correct_count"]) >= int(gate["step10c_total_mechanism_correct_min"])
    criteria["each_mechanism_floor"] = all(
        int(row["correct"]) >= int(gate["step10c_each_mechanism_correct_min_out_of_6"])
        for row in primary_metrics["by_mechanism"].values()
    )
    criteria["each_strength_floor"] = all(
        int(row["correct"]) >= int(gate["step10c_each_strength_correct_min_out_of_9"])
        for row in primary_metrics["by_strength"].values()
    )
    criteria["effect_detection"] = int(primary_metrics["distorted_effect_detection_count"]) >= int(gate["step10c_distorted_effect_detection_min"])
    criteria["clean_false_positive"] = int(primary_metrics["clean_effect_false_positive_count"]) <= int(gate["step10c_clean_effect_false_positive_max"])
    advantage = int(primary_metrics["distorted_mechanism_correct_count"]) - int(baseline_metrics["distorted_mechanism_correct_count"])
    criteria["paired_advantage"] = advantage >= int(gate["step10c_minus_step9a_mechanism_correct_min"])
    passed = all(criteria.values())
    interpretations = config["predeclared_analysis"]["gate_interpretation"]
    return {
        "passed": passed,
        "criteria": criteria,
        "step10c_minus_step9a_mechanism_correct": advantage,
        "interpretation": interpretations["all_criteria_met"] if passed else interpretations["otherwise"],
    }


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
    _assert_frozen_contract(config)
    plan_file = plan_file.expanduser().resolve()
    output = plan_file.parent
    if (output / "physical_access_started.json").exists():
        raise RuntimeError("this Step-12 plan already started physical-QPU access; any rerun requires a new plan")
    plan = read_json(plan_file)
    snapshot = read_json(output / "backend_snapshot.json")
    identifiability = read_json(output / "identifiability_audit.json")
    if plan.get("schema") != SCHEMA or plan.get("status") != "STEP12_QPU_PLAN_READY_NOT_SUBMITTED":
        raise RuntimeError("Step-12 plan is not ready/not-submitted")
    if plan.get("primary_candidate") != "step10c_warm_start":
        raise RuntimeError("Step-12 saved primary candidate drift")

    source_step11_sha = _verify_step11_freeze(config)
    step11._verify_step10d_reference(step10d_dir, config)
    primary = step11._load_step10c_primary(step10c_dir, config, device=device)
    baseline = step11._load_step9a_baseline(step9a_dir, config, device=device)
    versions = step11.verify_frozen_versions(config)
    if plan["identity"]["source_step11_manifest_sha256"] != source_step11_sha:
        raise RuntimeError("Step-11 frozen source manifest changed since Step-12 planning")
    if plan["identity"]["config_sha256"] != step11.sha256_file(config_path):
        raise RuntimeError("Step-12 config changed since planning; create a fresh plan")
    if plan["identity"]["primary_checkpoint_hashes"] != primary.checkpoint_hashes:
        raise RuntimeError("Step-12 primary checkpoint hashes changed since planning")
    if plan["identity"]["baseline_checkpoint_hashes"] != baseline.checkpoint_hashes:
        raise RuntimeError("Step-12 baseline checkpoint hashes changed since planning")
    if plan["identity"]["software_versions"] != versions:
        raise RuntimeError("Step-12 software versions changed since planning")
    if _sha256_json(identifiability) != plan["identity"]["identifiability_audit_sha256"]:
        raise RuntimeError("Step-12 identifiability audit artifact changed since planning")
    current_ident = model_blind_identifiability_audit(config)
    if current_ident != identifiability or current_ident["status"] != "PASS":
        raise RuntimeError("Step-12 identifiability audit changed since planning")

    from qiskit_ibm_runtime import QiskitRuntimeService
    discovery = QiskitRuntimeService()
    instance = step11._select_open_instance(list(discovery.instances()), instance_name)
    if str(instance.get("crn", "")) != str(plan["identity"]["instance_crn"]):
        raise RuntimeError("active Open Plan instance CRN does not match Step-12 plan")
    service = step11._make_open_service(instance)
    backend, candidate, ranking = _select_backend_and_chain_excluding(service, config)
    if str(backend.name) != str(snapshot["backend_name"]):
        raise RuntimeError("best allowed Step-12 backend changed since planning; create a fresh plan")
    if list(candidate.physical_chain) != list(snapshot["selected_chain"]["physical_chain"]):
        raise RuntimeError("best Step-12 physical chain changed since planning; create a fresh plan")
    current_snapshot = _snapshot(backend, candidate, ranking, instance)
    if current_snapshot["backend_version"] != snapshot["backend_version"]:
        raise RuntimeError("Step-12 backend version changed since planning; create a fresh plan")
    if current_snapshot["properties_last_update_date"] != snapshot["properties_last_update_date"]:
        raise RuntimeError("Step-12 backend calibration timestamp changed since planning; create a fresh plan")

    cases = build_generalization_cases(config, candidate.physical_chain)
    if _case_design_sha(cases) != plan["identity"]["case_design_sha256"]:
        raise RuntimeError("Step-12 case design changed since planning")
    programs, metadata = compile_pilot_programs(
        cases,
        backend,
        optimization_level=int(config["execution"]["optimization_level"]),
        seed_transpiler=int(config["execution"]["seed_transpiler"]),
        require_no_routing_permutation=bool(config["execution"]["require_no_routing_permutation"]),
    )
    if _compiled_metadata_sha(metadata) != plan["identity"]["compiled_program_metadata_sha256"]:
        raise RuntimeError("Step-12 compiled circuits changed since planning; create a fresh plan")

    with (output / "submitted_circuits.qpy").open("wb") as handle:
        qpy.dump(list(programs), handle)
    atomic_json(output / "physical_access_started.json", {
        "schema": SCHEMA,
        "status": "STEP12_PHYSICAL_QPU_ACCESS_STARTED",
        "plan_id": plan["plan_id"],
        "backend_name": str(backend.name),
        "program_count": len(programs),
        "shots_per_program": int(config["execution"]["shots_per_program"]),
        "total_executions": int(config["execution"]["total_executions"]),
        "primary_candidate": "step10c_warm_start",
        "started_unix": time.time(),
    })

    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    sampler.options.max_execution_time = int(config["execution"]["max_execution_time_seconds"])
    print("\nSTEP 12 PHYSICAL QPU ACCESS STARTED — independent cross-backend phase-generalization test; no tuning.\n", flush=True)
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
                raise RuntimeError(f"realized shot mismatch for {case.case_id}/{program}: {realized}")
        batch, primary_pred = batch_and_prediction_from_case_counts(case, case_counts, primary, device=device)
        _batch2, baseline_pred = batch_and_prediction_from_case_counts(case, case_counts, baseline, device=device)
        rows.append({
            "case_id": case.case_id,
            "family": case.family,
            "condition": case.condition,
            "physical_layout": list(case.physical_layout),
            "affected_logical_qubit": case.affected_logical_qubit,
            "strength": case.strength,
            "expected_effect": case.expected_effect,
            "expected_mechanism": case.expected_mechanism,
            "counts_by_program": case_counts,
            "diagnostic_rms": step11._diagnostic_rms(batch),
            "step10c_prediction": primary_pred,
            "step9a_prediction": baseline_pred,
        })

    with (output / "case_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    _write_csv(output / "predictions_step10c.csv", _prediction_rows(rows, "step10c_prediction"))
    _write_csv(output / "predictions_step9a.csv", _prediction_rows(rows, "step9a_prediction"))

    primary_metrics = _model_metrics(rows, "step10c_prediction", primary.mechanism_classes)
    baseline_metrics = _model_metrics(rows, "step9a_prediction", baseline.mechanism_classes)
    gate_result = evaluate_predeclared_gate(primary_metrics, baseline_metrics, config)

    primary_better = baseline_better = both_correct = both_wrong = 0
    for row in [r for r in rows if bool(r["expected_effect"])]:
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
        "status": "STEP12_INDEPENDENT_PHASE_GENERALIZATION_COMPLETE",
        "plan_id": plan["plan_id"],
        "ibm_job_id": job_id,
        "backend_name": str(backend.name),
        "primary_candidate": "step10c_warm_start",
        "paired_baseline": "step9a_deployment_ensemble",
        "primary_metrics": primary_metrics,
        "baseline_metrics_report_only": baseline_metrics,
        "predeclared_gate": gate_result,
        "paired_mechanism_correctness": {
            "step10c_correct_step9a_wrong_count": primary_better,
            "step9a_correct_step10c_wrong_count": baseline_better,
            "both_correct_count": both_correct,
            "both_wrong_count": both_wrong,
        },
        "qpu_usage_seconds": qpu_usage_seconds,
        "job_metrics": job_metrics,
        "full_triqto_confirmatory_claim_allowed": False,
        "qpu_results_used_for_tuning": False,
        "step11_result_replaced": False,
    }
    atomic_json(output / "step12_summary.json", summary)

    required = [str(v) for v in config["output"]["required_execution_files"] if str(v) != "step12_complete.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Step-12 required output missing before completion: {missing}")
    hashes = {name: step11.sha256_file(output / name) for name in required}
    complete = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "plan_id": plan["plan_id"],
        "ibm_job_id": job_id,
        "backend_name": str(backend.name),
        "physical_chain": list(candidate.physical_chain),
        "qpu_usage_seconds": qpu_usage_seconds,
        "predeclared_gate_passed": bool(gate_result["passed"]),
        "predeclared_interpretation": gate_result["interpretation"],
        "full_triqto_confirmatory_claim": False,
        "qpu_results_used_for_tuning": False,
        "step11_result_replaced": False,
        "file_hashes": hashes,
    }
    atomic_json(output / "step12_complete.json", complete)

    print("\nTRIQTO STEP 12 INDEPENDENT PHASE-GENERALIZATION TEST COMPLETE\n")
    print(f"IBM Runtime job: {job_id}")
    print(f"Backend: {backend.name} | chain={list(candidate.physical_chain)}")
    print(
        "Step10C mechanism correctness: "
        f"{primary_metrics['distorted_mechanism_correct_count']}/{primary_metrics['distorted_case_count']}"
    )
    print(
        "Step9A report-only mechanism correctness: "
        f"{baseline_metrics['distorted_mechanism_correct_count']}/{baseline_metrics['distorted_case_count']}"
    )
    print(f"Predeclared gate passed: {'YES' if gate_result['passed'] else 'NO'}")
    print(f"Interpretation: {gate_result['interpretation']}")
    print(f"Gate criteria: {gate_result['criteria']}")
    print(f"QPU usage seconds: {qpu_usage_seconds}")
    print("Full TriQTO confirmatory claim allowed: NO")
    print("QPU-driven tuning allowed: NO")
    print(f"Output: {output}")
    return output / "step12_complete.json"


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    _assert_frozen_contract(config)
    device = _resolve_device(args.device)
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)

    if args.execute_physical_qpu:
        expected = str(config["execution"]["explicit_confirmation_token"])
        if args.confirmation_token != expected:
            raise RuntimeError(f"physical QPU execution requires --confirmation-token {expected}")
        if args.plan_file is None:
            raise RuntimeError("physical QPU execution requires --plan-file from a prior Step-12 plan")
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
            instance_name=args.instance_name,
            device=device,
        )


if __name__ == "__main__":
    main()
