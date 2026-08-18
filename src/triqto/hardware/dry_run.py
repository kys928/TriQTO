"""Step-9C local hardware-path dry-run helpers.

This module does not submit physical hardware jobs. It exercises the exact
Step-9B compile/Sampler/count-to-DiagnosticTensorBatch path on a local BackendV2,
verifies tensor equivalence with the frozen Step-7 training adapter, and loads
only the already-frozen Step-9A deployment checkpoints for inference.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

from triqto.step7.contracts import DiagnosticTensorBatch, Step7ModelBatch
from triqto.step7.graph_adapter import batch_from_step5_examples
from triqto.step7.model import Step7DiagnosticModel

from .diagnostic_acquisition import (
    BASIS_ORDER,
    PROGRAM_ORDER,
    HardwareDiagnosticAcquisition,
    acquire_paired_diagnostics,
    paired_diagnostic_arrays,
    serialize_intended_graph,
)


@dataclass(frozen=True, slots=True)
class DryRunCircuitCase:
    case_id: str
    family: str
    reference_circuit: QuantumCircuit
    observed_circuit: QuantumCircuit
    initial_layout: tuple[int, ...]
    injected_mechanism_report_only: str
    strength: float


@dataclass(slots=True)
class FrozenDeploymentEnsemble:
    models: dict[int, Step7DiagnosticModel]
    seeds: tuple[int, ...]
    effect_threshold: float
    mechanism_classes: tuple[str, ...]
    checkpoint_hashes: dict[str, str]
    bundle_id: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def verify_step9b_contract(step9b_config: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected = config["source_step9b"]
    if step9b_config.get("schema") != expected["config_schema"]:
        raise RuntimeError("Step-9B config schema drift detected")
    if step9b_config.get("status") != "FROZEN_BEFORE_HARDWARE_DRY_RUN":
        raise RuntimeError("Step-9B contract is not frozen for dry-run consumption")
    deployment = config["deployment_bundle"]
    source_deployment = step9b_config["deployment_bundle"]
    for key in ("bundle_id", "architecture", "seeds", "effect_threshold", "mechanism_classes"):
        if source_deployment[key] != deployment[key]:
            raise RuntimeError(f"Step-9B deployment identity drift: {key}")
    if source_deployment["checkpoint_sha256"] != deployment["checkpoint_sha256"]:
        raise RuntimeError("Step-9B checkpoint identity drift detected")
    if step9b_config["evidence_contract"]["delta_sign"] != "observed_minus_paired_reference":
        raise RuntimeError("Step-9B diagnostic sign drift detected")
    if step9b_config["execution_contract"]["program_order"] != list(PROGRAM_ORDER):
        raise RuntimeError("Step-9B six-program order drift detected")


def verify_deployment_bundle(bundle_dir: Path, deployment: Mapping[str, Any]) -> dict[str, Any]:
    bundle_dir = bundle_dir.expanduser().resolve()
    complete_path = bundle_dir / "bundle_complete.json"
    if not complete_path.is_file():
        raise RuntimeError("deployment bundle is missing bundle_complete.json")
    complete = read_json(complete_path)
    if complete.get("status") != "DEPLOYMENT_REFIT_BUNDLE_FROZEN":
        raise RuntimeError("deployment bundle is not frozen")
    if complete.get("bundle_id") != deployment["bundle_id"]:
        raise RuntimeError("deployment bundle ID mismatch")
    if complete.get("architecture") != deployment["architecture"]:
        raise RuntimeError("deployment architecture mismatch")
    if list(complete.get("seeds", [])) != list(deployment["seeds"]):
        raise RuntimeError("deployment seed identity mismatch")
    if float(complete.get("deployment_effect_threshold")) != float(deployment["effect_threshold"]):
        raise RuntimeError("deployment effect threshold mismatch")
    if list(complete.get("mechanism_classes", [])) != list(deployment["mechanism_classes"]):
        raise RuntimeError("deployment mechanism-class order mismatch")
    if complete.get("weight_provenance") != deployment["weight_provenance"]:
        raise RuntimeError("deployment weight provenance mismatch")
    if bool(complete.get("spent_confirmatory_cohort_accessed", True)):
        raise RuntimeError("deployment bundle unexpectedly reports confirmatory-cohort access")

    recorded = dict(complete.get("file_hashes", {}))
    for filename, expected_hash in deployment["checkpoint_sha256"].items():
        path = bundle_dir / filename
        if not path.is_file():
            raise RuntimeError(f"deployment checkpoint missing: {filename}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"deployment checkpoint hash mismatch: {filename}")
        if recorded.get(filename) != expected_hash:
            raise RuntimeError(f"bundle_complete checkpoint hash mismatch: {filename}")
    return complete


def load_frozen_deployment_ensemble(
    bundle_dir: Path,
    deployment: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> FrozenDeploymentEnsemble:
    verify_deployment_bundle(bundle_dir, deployment)
    target_device = torch.device(device)
    models: dict[int, Step7DiagnosticModel] = {}
    checkpoint_hashes: dict[str, str] = {}
    expected_params = int(deployment["expected_trainable_parameter_count"])

    for seed_value in deployment["seeds"]:
        seed = int(seed_value)
        filename = f"seed{seed}.pt"
        path = bundle_dir.expanduser().resolve() / filename
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("architecture") != "late_concat" or int(payload.get("seed")) != seed:
            raise RuntimeError(f"checkpoint identity mismatch for seed {seed}")
        if payload.get("weight_provenance") != deployment["weight_provenance"]:
            raise RuntimeError(f"checkpoint provenance mismatch for seed {seed}")
        if bool(payload.get("exact_step8_checkpoint_weight", True)):
            raise RuntimeError("Step-9C may not relabel the deployment refit as exact Step-8 weights")
        model = Step7DiagnosticModel(variant="late_concat", initialization_seed=seed)
        model.load_state_dict(payload["state_dict"], strict=True)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if parameter_count != expected_params:
            raise RuntimeError(
                f"checkpoint parameter count mismatch for seed {seed}: {parameter_count} != {expected_params}"
            )
        model.to(target_device)
        model.eval()
        models[seed] = model
        checkpoint_hashes[filename] = sha256_file(path)

    return FrozenDeploymentEnsemble(
        models=models,
        seeds=tuple(int(seed) for seed in deployment["seeds"]),
        effect_threshold=float(deployment["effect_threshold"]),
        mechanism_classes=tuple(str(value) for value in deployment["mechanism_classes"]),
        checkpoint_hashes=checkpoint_hashes,
        bundle_id=str(deployment["bundle_id"]),
    )


def build_local_backend(simulator_config: Mapping[str, Any]) -> GenericBackendV2:
    return GenericBackendV2(
        num_qubits=int(simulator_config["num_qubits"]),
        basis_gates=list(simulator_config["basis_gates"]),
        coupling_map=[list(pair) for pair in simulator_config["coupling_map"]],
        seed=int(simulator_config["backend_seed"]),
        noise_info=bool(simulator_config["noise_info"]),
    )


def build_local_sampler(backend: GenericBackendV2, simulator_config: Mapping[str, Any]) -> AerSamplerV2:
    return AerSamplerV2.from_backend(
        backend,
        seed=int(simulator_config["sampler_seed"]),
    )


def build_dry_run_cases(config: Mapping[str, Any]) -> tuple[DryRunCircuitCase, ...]:
    cases: list[DryRunCircuitCase] = []
    for raw in config["dry_run_cases"]:
        case_id = str(raw["case_id"])
        strength = float(raw["strength"])
        n_qubits = int(raw["n_qubits"])
        layout = tuple(int(value) for value in raw["initial_layout"])
        mechanism = str(raw["injected_mechanism_report_only"])

        reference = QuantumCircuit(n_qubits, name=f"{case_id}__reference")
        observed = QuantumCircuit(n_qubits, name=f"{case_id}__observed")
        if case_id == "bell_rz_q0":
            reference.h(0)
            reference.cx(0, 1)
            observed.h(0)
            observed.rz(strength, 0)
            observed.cx(0, 1)
        elif case_id == "ghz_rx_q1":
            reference.h(0)
            reference.cx(0, 1)
            reference.cx(1, 2)
            observed.h(0)
            observed.cx(0, 1)
            observed.rx(strength, 1)
            observed.cx(1, 2)
        elif case_id == "phase_ry_q0":
            reference.h(0)
            reference.rz(0.7, 0)
            reference.h(0)
            reference.cx(0, 1)
            observed.h(0)
            observed.rz(0.7, 0)
            observed.ry(strength, 0)
            observed.h(0)
            observed.cx(0, 1)
        else:
            raise RuntimeError(f"unimplemented frozen Step-9C case: {case_id}")

        if reference.num_qubits != n_qubits or observed.num_qubits != n_qubits:
            raise RuntimeError(f"dry-run circuit qubit mismatch: {case_id}")
        if len(layout) != n_qubits:
            raise RuntimeError(f"dry-run layout length mismatch: {case_id}")
        cases.append(
            DryRunCircuitCase(
                case_id=case_id,
                family=str(raw["family"]),
                reference_circuit=reference,
                observed_circuit=observed,
                initial_layout=layout,
                injected_mechanism_report_only=mechanism,
                strength=strength,
            )
        )
    return tuple(cases)


def _training_adapter_batch_from_acquisition(
    case: DryRunCircuitCase,
    acquisition: HardwareDiagnosticAcquisition,
    *,
    device: torch.device | str = "cpu",
) -> Step7ModelBatch:
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
    # Targets are required by the legacy adapter API but are deliberately dummy and
    # never used in Step-9C inference or any pass/fail decision.
    example["y__effect_present_target"] = np.asarray([0], dtype=np.int8)
    example["y__mechanism_target"] = np.asarray([0], dtype=np.int8)
    example["y__mechanism_loss_mask"] = np.asarray([False], dtype=np.bool_)
    batch, _targets = batch_from_step5_examples([example], device=device)
    return batch


def _assert_tensor_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise RuntimeError(
            f"Step-9C tensor contract mismatch for {name}: "
            f"actual dtype/shape={actual.dtype}/{tuple(actual.shape)} "
            f"expected={expected.dtype}/{tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        if actual.dtype.is_floating_point:
            max_abs = float((actual - expected).abs().max().detach().cpu()) if actual.numel() else 0.0
            raise RuntimeError(f"Step-9C tensor value mismatch for {name}; max_abs={max_abs:.9g}")
        raise RuntimeError(f"Step-9C tensor value mismatch for {name}")


def assert_training_hardware_batch_equivalent(
    hardware_batch: Step7ModelBatch,
    training_batch: Step7ModelBatch,
) -> None:
    if hardware_batch.graph.graph_count != training_batch.graph.graph_count:
        raise RuntimeError("Step-9C graph_count mismatch between hardware and training adapters")
    for field in fields(hardware_batch.graph):
        name = field.name
        if name == "graph_count":
            continue
        _assert_tensor_equal(
            f"graph.{name}",
            getattr(hardware_batch.graph, name),
            getattr(training_batch.graph, name),
        )
    for field in fields(DiagnosticTensorBatch):
        name = field.name
        _assert_tensor_equal(
            f"diagnostic.{name}",
            getattr(hardware_batch.diagnostic, name),
            getattr(training_batch.diagnostic, name),
        )


def verify_acquisition_structure(
    acquisition: HardwareDiagnosticAcquisition,
    *,
    requested_shots: int,
) -> None:
    if len(acquisition.isa_circuits) != len(PROGRAM_ORDER):
        raise RuntimeError("Step-9C did not receive all six ISA circuits")
    if tuple(acquisition.counts_by_program) != PROGRAM_ORDER:
        raise RuntimeError("Step-9C program/count ordering drift detected")
    for label, circuit in zip(PROGRAM_ORDER, acquisition.isa_circuits, strict=True):
        if "meas" not in {register.name for register in circuit.cregs}:
            raise RuntimeError(f"Step-9C transpilation removed meas register: {label}")
        realized = sum(int(value) for value in acquisition.counts_by_program[label].values())
        if realized != int(requested_shots):
            raise RuntimeError(
                f"Step-9C realized-shot mismatch for {label}: {realized} != {requested_shots}"
            )
    acquisition.model_batch.diagnostic.validate(acquisition.model_batch.graph)


def predict_frozen_ensemble(
    ensemble: FrozenDeploymentEnsemble,
    batch: Step7ModelBatch,
) -> dict[str, Any]:
    effect_logits: list[float] = []
    mechanism_logits: list[np.ndarray] = []
    with torch.no_grad():
        for seed in ensemble.seeds:
            model = ensemble.models[seed]
            model.validate_batch(batch)
            output = model(batch)
            effect_value = float(output.effect_logit.detach().cpu().reshape(-1)[0])
            mechanism_value = output.mechanism_logits.detach().cpu().numpy().reshape(1, 3)[0]
            if not np.isfinite(effect_value) or not np.isfinite(mechanism_value).all():
                raise RuntimeError(f"non-finite frozen-model output for seed {seed}")
            effect_logits.append(effect_value)
            mechanism_logits.append(mechanism_value.astype(np.float64, copy=False))

    mean_effect = float(np.mean(np.asarray(effect_logits, dtype=np.float64)))
    mean_mechanism = np.mean(np.stack(mechanism_logits, axis=0), axis=0)
    mechanism_code = int(np.argmax(mean_mechanism))
    return {
        "seed_effect_logits": {str(seed): effect_logits[index] for index, seed in enumerate(ensemble.seeds)},
        "seed_mechanism_logits": {
            str(seed): mechanism_logits[index].tolist() for index, seed in enumerate(ensemble.seeds)
        },
        "mean_effect_logit": mean_effect,
        "effect_probability": float(1.0 / (1.0 + np.exp(-mean_effect))),
        "effect_threshold": float(ensemble.effect_threshold),
        "effect_present": bool(mean_effect >= ensemble.effect_threshold),
        "mean_mechanism_logits": mean_mechanism.tolist(),
        "mechanism_code": mechanism_code,
        "mechanism_prediction": ensemble.mechanism_classes[mechanism_code],
    }


__all__ = [
    "DryRunCircuitCase",
    "FrozenDeploymentEnsemble",
    "assert_training_hardware_batch_equivalent",
    "build_dry_run_cases",
    "build_local_backend",
    "build_local_sampler",
    "load_frozen_deployment_ensemble",
    "predict_frozen_ensemble",
    "read_json",
    "sha256_file",
    "verify_acquisition_structure",
    "verify_deployment_bundle",
    "verify_step9b_contract",
]
