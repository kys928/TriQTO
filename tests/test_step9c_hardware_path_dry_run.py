from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

from triqto.hardware.diagnostic_acquisition import (
    BASIS_ORDER,
    acquire_paired_diagnostics,
    paired_diagnostic_arrays,
    serialize_intended_graph,
)
from triqto.hardware.dry_run import (
    FrozenDeploymentEnsemble,
    assert_training_hardware_batch_equivalent,
    build_dry_run_cases,
    load_frozen_deployment_ensemble,
    predict_frozen_ensemble,
    sha256_file,
    verify_acquisition_structure,
    verify_step9b_contract,
)
from triqto.step7.graph_adapter import batch_from_step5_examples
from triqto.step7.model import Step7DiagnosticModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v0_2/step9c_hardware_path_dry_run.json"
STEP9B = ROOT / "configs/v0_2/step9b_hardware_acquisition.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def training_batch_from_acquisition(case, acquisition):
    reference_counts = {
        basis: acquisition.counts_by_program[f"reference_{basis}"] for basis in BASIS_ORDER
    }
    observed_counts = {
        basis: acquisition.counts_by_program[f"observed_{basis}"] for basis in BASIS_ORDER
    }
    example = {}
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
    return batch_from_step5_examples([example], device="cpu")[0]


def test_step9c_contract_is_frozen_and_performance_is_not_a_gate() -> None:
    cfg = load_config()
    assert cfg["schema"] == "triqto.v0_2.step9c_hardware_path_dry_run.v1"
    assert cfg["status"] == "FROZEN_BEFORE_DRY_RUN_EXECUTION"
    assert cfg["source_step9b"]["merge_commit"] == "72af2fa0bc6fd4981abbeefb338ffa0e759fd3f5"
    assert cfg["deployment_bundle"]["bundle_id"] == "deploy_ac536a74b2f8dd571d353a12"
    assert cfg["pass_gates"]["prediction_correctness_is_not_a_gate"] is True
    assert cfg["scientific_boundaries"]["physical_qpu_execution"] is False
    assert cfg["scientific_boundaries"]["new_training"] is False
    assert cfg["scientific_boundaries"]["dry_run_predictions_are_confirmatory_evidence"] is False


def test_step9b_contract_matches_step9c_bound_identity() -> None:
    cfg = load_config()
    step9b = json.loads(STEP9B.read_text(encoding="utf-8"))
    verify_step9b_contract(step9b, cfg)


def test_dry_run_cases_cover_three_mechanisms_without_exposing_labels_to_batch() -> None:
    cases = build_dry_run_cases(load_config())
    assert [case.case_id for case in cases] == ["bell_rz_q0", "ghz_rx_q1", "phase_ry_q0"]
    assert {case.injected_mechanism_report_only for case in cases} == {
        "rz_drift",
        "rx_overrotation",
        "ry_overrotation",
    }
    for case in cases:
        assert case.reference_circuit.num_clbits == 0
        assert case.observed_circuit.num_clbits == 0
        assert case.reference_circuit != case.observed_circuit
        assert len(case.initial_layout) == case.reference_circuit.num_qubits


def test_real_aer_sampler_path_matches_step7_training_adapter_tensors() -> None:
    cfg = load_config()
    case = build_dry_run_cases(cfg)[0]
    backend = GenericBackendV2(
        num_qubits=2,
        basis_gates=["id", "rz", "sx", "x", "cx"],
        coupling_map=[[0, 1], [1, 0]],
        seed=17,
        noise_info=False,
    )
    sampler = AerSamplerV2.from_backend(backend, seed=19)
    acquisition = acquire_paired_diagnostics(
        case.reference_circuit,
        case.observed_circuit,
        backend,
        sampler,
        initial_layout=[0, 1],
        shots=64,
        optimization_level=1,
        seed_transpiler=17091,
        device="cpu",
    )
    verify_acquisition_structure(acquisition, requested_shots=64)
    training_batch = training_batch_from_acquisition(case, acquisition)
    assert_training_hardware_batch_equivalent(acquisition.model_batch, training_batch)


def test_tensor_equivalence_fails_closed_on_semantic_drift() -> None:
    cfg = load_config()
    case = build_dry_run_cases(cfg)[0]
    reference = {basis: {"00": 32, "11": 32} for basis in BASIS_ORDER}
    observed = {basis: {"00": 30, "01": 2, "10": 2, "11": 30} for basis in BASIS_ORDER}
    from triqto.hardware.diagnostic_acquisition import build_step7_model_batch_from_counts

    hardware = build_step7_model_batch_from_counts(
        case.reference_circuit, case.initial_layout, reference, observed, device="cpu"
    )
    training = build_step7_model_batch_from_counts(
        case.reference_circuit, case.initial_layout, reference, observed, device="cpu"
    )
    training.diagnostic.local_values[0, 0] += 0.125
    with pytest.raises(RuntimeError, match="diagnostic.local_values"):
        assert_training_hardware_batch_equivalent(hardware, training)


def test_deployment_loader_verifies_hashes_and_loads_cpu_checkpoint(tmp_path: Path) -> None:
    seed = 1701
    model = Step7DiagnosticModel(variant="late_concat", initialization_seed=seed)
    checkpoint = {
        "schema": "triqto.v0_2.step9a_deployment_freeze.v2",
        "architecture": "late_concat",
        "seed": seed,
        "fixed_epoch": 8,
        "epoch_source": "archived_step8_selected_epoch",
        "weight_provenance": "post_confirmation_fixed_epoch_refit_development_only",
        "exact_step8_checkpoint_weight": False,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    path = tmp_path / "seed1701.pt"
    torch.save(checkpoint, path)
    digest = sha256_file(path)
    deployment = {
        "bundle_id": "deploy_test",
        "architecture": "late_concat",
        "seeds": [seed],
        "checkpoint_sha256": {"seed1701.pt": digest},
        "effect_threshold": 0.05939410626888275,
        "mechanism_classes": ["rz_drift", "rx_overrotation", "ry_overrotation"],
        "expected_trainable_parameter_count": 453829,
        "weight_provenance": "post_confirmation_fixed_epoch_refit_development_only",
    }
    complete = {
        "status": "DEPLOYMENT_REFIT_BUNDLE_FROZEN",
        "bundle_id": "deploy_test",
        "architecture": "late_concat",
        "seeds": [seed],
        "deployment_effect_threshold": 0.05939410626888275,
        "mechanism_classes": ["rz_drift", "rx_overrotation", "ry_overrotation"],
        "weight_provenance": "post_confirmation_fixed_epoch_refit_development_only",
        "spent_confirmatory_cohort_accessed": False,
        "file_hashes": {"seed1701.pt": digest},
    }
    (tmp_path / "bundle_complete.json").write_text(json.dumps(complete), encoding="utf-8")
    ensemble = load_frozen_deployment_ensemble(tmp_path, deployment, device="cpu")
    assert ensemble.seeds == (1701,)
    assert ensemble.checkpoint_hashes["seed1701.pt"] == digest
    assert sum(p.numel() for p in ensemble.models[1701].parameters() if p.requires_grad) == 453829


def test_frozen_ensemble_prediction_is_finite_and_uses_frozen_threshold() -> None:
    case = build_dry_run_cases(load_config())[0]
    from triqto.hardware.diagnostic_acquisition import build_step7_model_batch_from_counts

    reference = {basis: {"00": 32, "11": 32} for basis in BASIS_ORDER}
    observed = {basis: {"00": 30, "01": 2, "10": 2, "11": 30} for basis in BASIS_ORDER}
    batch = build_step7_model_batch_from_counts(
        case.reference_circuit, case.initial_layout, reference, observed, device="cpu"
    )
    models = {
        seed: Step7DiagnosticModel(variant="late_concat", initialization_seed=seed).eval()
        for seed in (1701, 1702, 1703)
    }
    ensemble = FrozenDeploymentEnsemble(
        models=models,
        seeds=(1701, 1702, 1703),
        effect_threshold=0.05939410626888275,
        mechanism_classes=("rz_drift", "rx_overrotation", "ry_overrotation"),
        checkpoint_hashes={},
        bundle_id="test",
    )
    prediction = predict_frozen_ensemble(ensemble, batch)
    assert np.isfinite(prediction["mean_effect_logit"])
    assert np.isfinite(prediction["mean_mechanism_logits"]).all()
    assert prediction["effect_threshold"] == pytest.approx(0.05939410626888275, abs=0.0)
    assert prediction["mechanism_prediction"] in ensemble.mechanism_classes
