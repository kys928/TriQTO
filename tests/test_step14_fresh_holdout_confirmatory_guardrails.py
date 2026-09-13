from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
V02 = SCRIPTS / "v0_2"
for value in (str(SCRIPTS), str(V02)):
    if value not in sys.path:
        sys.path.insert(0, value)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load("step14_launcher_guardrail_test", SCRIPTS / "runpod_step14_training_selection.py")
worker = _load("step14_worker_guardrail_test", SCRIPTS / "runpod_step14_training_worker.py")
predict = _load("step14_predict_guardrail_test", V02 / "predict_step14_equivalence_aware_fresh_holdout_blind.py")

RUN_ID = "training_18e0b4ed6e685af30b6c4a35"
SELECTION_SHA = "sha256:" + "a" * 64
FINAL_SHA = "sha256:" + "b" * 64
DATASET_SHA = "sha256:" + "c" * 64
PREDICTION_SHA = "sha256:" + "d" * 64
PRODUCT_ID = "fresh_holdout_example"


def _base_request(operation: str) -> dict[str, object]:
    return {
        "id": f"test-{operation}",
        "operation": operation,
        "expected_training_run_id": RUN_ID,
        "expected_selection_freeze_sha256": SELECTION_SHA,
        "expected_final_method_freeze_payload_sha256": FINAL_SHA,
    }


def test_launcher_predict_requires_fresh_holdout_identity(tmp_path: Path) -> None:
    request = _base_request("predict_equivalence_aware_fresh_holdout")
    request["expected_fresh_holdout_product_id"] = PRODUCT_ID
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_fresh_holdout_dataset_complete_sha256"):
        launcher.load_request(path)


def test_launcher_score_requires_prediction_freeze_identity(tmp_path: Path) -> None:
    request = _base_request("score_equivalence_aware_fresh_holdout")
    request["expected_fresh_holdout_product_id"] = PRODUCT_ID
    request["expected_fresh_holdout_dataset_complete_sha256"] = DATASET_SHA
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_oracle_free_prediction_complete_sha256"):
        launcher.load_request(path)


def test_phase_a_path_guard_rejects_sealed_truth_and_outside(tmp_path: Path) -> None:
    product = tmp_path / "fresh_holdout"
    blind = product / "blind_artifacts" / "example.npz"
    sealed = product / "sealed_truth" / "raw_artifacts" / "example.npz"
    outside = tmp_path / "elsewhere" / "example.npz"
    assert predict.path_is_blind_artifact(product, blind)
    assert not predict.path_is_blind_artifact(product, sealed)
    assert not predict.path_is_blind_artifact(product, outside)


def test_phase_a_npz_loader_accepts_x_only_and_rejects_non_x(tmp_path: Path) -> None:
    product = tmp_path / "fresh_holdout"
    blind_dir = product / "blind_artifacts"
    blind_dir.mkdir(parents=True)

    good = blind_dir / "good.npz"
    np.savez_compressed(good, x__feature=np.asarray([1.0, 2.0], dtype=np.float32))
    good_row = {
        "artifact_path": "blind_artifacts/good.npz",
        "artifact_sha256": predict.baseline.sha256_file(good),
        "example_id": "good",
    }
    loaded = predict.load_x_only_artifact(product, good_row)
    assert set(loaded) == {"x__feature"}
    np.testing.assert_array_equal(loaded["x__feature"], np.asarray([1.0, 2.0], dtype=np.float32))

    bad = blind_dir / "bad.npz"
    np.savez_compressed(
        bad,
        x__feature=np.asarray([1.0], dtype=np.float32),
        y__mechanism=np.asarray([2], dtype=np.int64),
    )
    bad_row = {
        "artifact_path": "blind_artifacts/bad.npz",
        "artifact_sha256": predict.baseline.sha256_file(bad),
        "example_id": "bad",
    }
    with pytest.raises(RuntimeError, match="non-x array"):
        predict.load_x_only_artifact(product, bad_row)


def test_worker_routes_prediction_without_sealed_truth_path() -> None:
    job = {
        "task": {
            "runner": "step14_training_selection",
            "command": "predict_equivalence_aware_fresh_holdout",
            "workspace": "/workspace/triqto-data/step14_cross_motif_training",
            "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
            "progress_every": 1000,
            "expected_training_run_id": RUN_ID,
            "expected_selection_freeze_sha256": SELECTION_SHA,
            "expected_final_method_freeze_payload_sha256": FINAL_SHA,
            "expected_fresh_holdout_product_id": PRODUCT_ID,
            "expected_fresh_holdout_dataset_complete_sha256": DATASET_SHA,
        }
    }
    command = worker.build_command(job)
    joined = " ".join(command)
    assert "predict_step14_equivalence_aware_fresh_holdout_blind.py" in joined
    assert "sealed_truth" not in joined


def test_worker_score_requires_and_routes_prediction_completion_sha() -> None:
    task = {
        "runner": "step14_training_selection",
        "command": "score_equivalence_aware_fresh_holdout",
        "workspace": "/workspace/triqto-data/step14_cross_motif_training",
        "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
        "progress_every": 1000,
        "expected_training_run_id": RUN_ID,
        "expected_selection_freeze_sha256": SELECTION_SHA,
        "expected_final_method_freeze_payload_sha256": FINAL_SHA,
        "expected_fresh_holdout_product_id": PRODUCT_ID,
        "expected_fresh_holdout_dataset_complete_sha256": DATASET_SHA,
    }
    with pytest.raises(ValueError, match="prediction completion"):
        worker.build_command({"task": task})
    task["expected_oracle_free_prediction_complete_sha256"] = PREDICTION_SHA
    command = worker.build_command({"task": task})
    joined = " ".join(command)
    assert "score_step14_equivalence_aware_fresh_holdout_confirmatory_cached.py" in joined
    assert PREDICTION_SHA in command
