from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
V02 = SCRIPTS / "v0_2"
for path in (str(SCRIPTS), str(V02)):
    if path not in sys.path:
        sys.path.insert(0, path)

import analyze_step14_latent_frame_inference as latent
import runpod_step14_training_selection as launcher
import runpod_step14_training_worker as worker

FREEZE = "sha256:" + "a" * 64


def small_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.ry(0.31, 0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(-0.22, 0)
    qc.ry(0.27, 1)
    qc.cz(1, 0)
    qc.h(0)
    qc.h(1)
    return qc


def test_candidate_set_depends_only_on_public_circuit_structure() -> None:
    qc = small_circuit()
    candidates = latent.plausible_candidates(qc)
    # Entanglers are at operation indices 2 and 5, so the only candidate
    # boundaries are immediately before/after them. No hidden location is an
    # input to plausible_candidates.
    assert {boundary for _q, boundary in candidates} == {2, 3, 5, 6}
    assert (0, 2) in candidates
    assert (1, 3) in candidates
    assert len(candidates) == len(set(candidates))


def test_primary_candidate_frame_is_finite_shot_and_deterministic() -> None:
    qc = small_circuit()
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    candidates = [(0, 2), (1, 3)]
    cfg = {
        "hardware_facing_frame_calibration": {
            "probe_strength_radians": 0.12,
            "shots_per_basis_per_sign": 128,
            "calibration_seed": 12345,
        }
    }
    first = latent.finite_shot_candidate_jacobians(qc, pairs, candidates, 7, cfg)
    second = latent.finite_shot_candidate_jacobians(qc, pairs, candidates, 7, cfg)
    assert len(first) == len(candidates)
    assert first[0].shape == (12, 3)
    assert np.all(np.isfinite(first[0]))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])

    # Different calibration seed changes sampled Born evidence. This guards
    # against accidentally replacing the primary finite-shot arm with the
    # exact statevector Jacobian.
    cfg2 = {
        "hardware_facing_frame_calibration": {
            "probe_strength_radians": 0.12,
            "shots_per_basis_per_sign": 128,
            "calibration_seed": 54321,
        }
    }
    changed = latent.finite_shot_candidate_jacobians(qc, pairs, candidates, 7, cfg2)
    assert not np.array_equal(first[0], changed[0])


def test_profile_marginalization_has_no_location_labels() -> None:
    delta = np.asarray([0.3, -0.2, 0.1, 0.4], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    j0 = np.asarray(
        [[1.0, 0.0, 0.0], [0.2, 0.8, 0.1], [0.0, 0.1, 1.0], [0.4, -0.1, 0.2]],
        dtype=np.float64,
    )
    j1 = np.asarray(
        [[0.0, 1.0, 0.0], [0.7, 0.1, 0.2], [0.1, 0.0, 0.9], [-0.2, 0.3, 0.4]],
        dtype=np.float64,
    )
    candidate_scores = latent.profile_log_likelihoods(delta, weights, [j0, j1])
    marginal, maximum = latent.mechanism_scores(candidate_scores)
    assert candidate_scores.shape == (2, 3)
    assert marginal.shape == (3,)
    assert maximum.shape == (3,)
    assert np.all(np.isfinite(marginal))
    assert np.all(np.isfinite(maximum))


def test_frozen_protocol_forbids_true_location_in_inference() -> None:
    cfg = json.loads((ROOT / "configs/v0_2/step14_latent_frame_inference.json").read_text())
    boundary = cfg["location_privilege_boundary"]
    assert cfg["status"] == "FROZEN_BEFORE_EXECUTION"
    assert boundary["true_affected_qubit_forbidden_from_candidate_generation"] is True
    assert boundary["true_injection_boundary_forbidden_from_candidate_generation"] is True
    assert boundary["true_location_forbidden_from_scoring"] is True
    assert boundary["true_location_forbidden_from_probe_training"] is True
    assert boundary["true_location_allowed_only_after_prediction_for_localization_audit"] is True
    assert cfg["hardware_facing_frame_calibration"]["exact_statevector_values_may_not_enter_primary_features_or_scores"] is True
    assert cfg["exact_frame_upper_bound"]["may_not_determine_primary_verdict"] is True


def test_typed_latent_frame_operation_routes_to_restricted_worker(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "id": "latent-frame-test",
                "operation": "decompose_latent_frame_inference",
                "gpu_type_ids": ["NVIDIA GeForce RTX 3090"],
                "container_disk_gb": 50,
                "interruptible": False,
                "progress_every": 1000,
                "expected_training_run_id": "training_test",
                "expected_selection_freeze_sha256": FREEZE,
            }
        ),
        encoding="utf-8",
    )
    loaded = launcher.load_request(request_path)
    assert loaded["operation"] == "decompose_latent_frame_inference"

    command = worker.build_command(
        {
            "task": {
                "runner": "step14_training_selection",
                "command": "decompose_latent_frame_inference",
                "workspace": "/workspace/triqto-data/step14_cross_motif_training",
                "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
                "progress_every": 1000,
                "expected_training_run_id": "training_test",
                "expected_selection_freeze_sha256": FREEZE,
            }
        }
    )
    assert command[1].endswith("scripts/v0_2/analyze_step14_latent_frame_inference.py")
    assert "--device" in command
    assert "cuda" in command
