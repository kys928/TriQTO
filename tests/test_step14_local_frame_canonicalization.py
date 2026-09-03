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

import analyze_step14_local_frame_canonicalization as frame
import generate_step14_cross_motif_dataset as generator
import runpod_step14_training_selection as launcher
import runpod_step14_training_worker as worker


FREEZE = "sha256:" + "a" * 64


def test_local_frame_jacobian_matches_finite_difference() -> None:
    circuit = QuantumCircuit(2)
    circuit.ry(0.31, 0)
    circuit.h(1)
    circuit.cx(0, 1)
    circuit.rz(-0.42, 1)
    circuit.ry(0.19, 0)
    boundary = 2
    affected = 0
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    jac = frame.frame_response_jacobian(circuit, boundary, affected, pairs)
    epsilon = 1.0e-6
    for column, mechanism in enumerate(frame.MECHANISMS):
        plus = generator.BASE.inject_hidden_rotation(
            circuit, boundary, affected, mechanism, epsilon
        )
        minus = generator.BASE.inject_hidden_rotation(
            circuit, boundary, affected, mechanism, -epsilon
        )
        finite = (
            generator.ideal_vector(circuit, plus)
            - generator.ideal_vector(circuit, minus)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(jac[:, column], finite, rtol=2.0e-6, atol=2.0e-7)


def test_canonicalizer_recovers_axis_for_exact_linear_response() -> None:
    jac = np.asarray(
        [
            [1.0, 0.1, 0.0],
            [0.0, 1.2, 0.1],
            [0.2, 0.0, 0.9],
            [0.5, -0.2, 0.1],
            [0.1, 0.3, -0.4],
        ],
        dtype=np.float64,
    )
    strength = 0.23
    delta = jac[:, 1] * strength
    feature, coords, audit = frame.canonicalize_evidence(
        delta, jac, np.ones(len(delta), dtype=np.float64)
    )
    assert feature.ndim == 1
    assert int(np.argmax(np.abs(coords))) == 1
    assert audit["residual_fraction"] < 0.01


def test_typed_local_frame_operation_routes_to_restricted_worker(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "id": "local-frame-test",
                "operation": "decompose_local_frame_canonicalization",
                "gpu_type_ids": ["NVIDIA GeForce RTX 3090"],
                "container_disk_gb": 50,
                "interruptible": False,
                "progress_every": 5000,
                "expected_training_run_id": "training_test",
                "expected_selection_freeze_sha256": FREEZE,
            }
        ),
        encoding="utf-8",
    )
    loaded = launcher.load_request(request_path)
    assert loaded["operation"] == "decompose_local_frame_canonicalization"

    command = worker.build_command(
        {
            "task": {
                "runner": "step14_training_selection",
                "command": "decompose_local_frame_canonicalization",
                "workspace": "/workspace/triqto-data/step14_cross_motif_training",
                "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
                "progress_every": 5000,
                "expected_training_run_id": "training_test",
                "expected_selection_freeze_sha256": FREEZE,
            }
        }
    )
    assert command[1].endswith("scripts/v0_2/analyze_step14_local_frame_canonicalization.py")
    assert "--device" in command
    assert "cuda" in command
