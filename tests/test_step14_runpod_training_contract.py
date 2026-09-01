from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runpod_step14_training_worker as worker  # noqa: E402


def make_job(operation: str) -> dict:
    return {
        "task": {
            "runner": "step14_training_selection",
            "command": operation,
            "workspace": "/workspace/triqto-data/step14_cross_motif_training",
            "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
            "progress_every": 5000,
        }
    }


def test_pretraining_baseline_is_a_dedicated_typed_operation() -> None:
    command = worker.build_command(make_job("evaluate_pretraining_baseline"))
    assert command[1].endswith("scripts/v0_2/evaluate_step14_pretraining_baseline.py")
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cuda"


def test_fit_selection_training_is_a_dedicated_typed_operation() -> None:
    command = worker.build_command(make_job("train_selection"))
    assert command[1].endswith("scripts/v0_2/run_step14_cross_motif_training.py")
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cuda"


def test_outer_evaluation_is_typed_and_hash_pinned() -> None:
    job = make_job("evaluate_outer")
    job["task"]["expected_training_run_id"] = "training_18e0b4ed6e685af30b6c4a35"
    job["task"]["expected_selection_freeze_sha256"] = (
        "sha256:af7ffaece77d16134c33b1c898afb7165c7ace5dcfccfea23f713ae242e4ed6f"
    )
    command = worker.build_command(job)
    assert command[1].endswith("scripts/v0_2/run_step14_frozen_outer_pipeline.py")
    assert command[command.index("--training-run-id") + 1] == "training_18e0b4ed6e685af30b6c4a35"
    assert command[command.index("--selection-freeze-sha256") + 1].startswith("sha256:")
    assert "--device" not in command


def test_outer_evaluation_requires_frozen_identifiers() -> None:
    with pytest.raises(ValueError, match="training run id"):
        worker.build_command(make_job("evaluate_outer"))


@pytest.mark.parametrize("operation", ["outer", "future_hardware", "qpu", "shell", "all"])
def test_worker_rejects_untyped_operations(operation: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Step-14 operation"):
        worker.build_command(make_job(operation))


def test_baseline_training_reject_outer_identifiers() -> None:
    job = make_job("train_selection")
    job["task"]["expected_training_run_id"] = "training_anything"
    with pytest.raises(ValueError, match="may not consume"):
        worker.build_command(job)


def test_worker_is_hard_pinned_to_frozen_protocol_config() -> None:
    job = make_job("train_selection")
    job["task"]["config"] = "configs/v0_2/step12_independent_phase_generalization.json"
    with pytest.raises(ValueError, match="hard-pinned"):
        worker.build_command(job)
