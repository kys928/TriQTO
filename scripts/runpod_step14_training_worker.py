#!/usr/bin/env python3
"""Restricted RunPod worker for frozen Step-14 scientific operations."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
from typing import Any

import runpod_worker as common

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = Path("/workspace/triqto-control/runs")
CONFIG = "configs/v0_2/step14_cross_motif_generalization_training.json"
ALLOWED_OPERATIONS = {
    "evaluate_pretraining_baseline",
    "train_selection",
    "evaluate_outer",
    "decompose_representation",
    "decompose_oracle_raw_evidence",
    "decompose_local_frame_canonicalization",
    "decompose_latent_frame_inference",
    "decompose_candidate_frame_ambiguity",
    "fit_equivalence_aware_latent_frame",
    "generate_equivalence_aware_fresh_holdout",
    "predict_equivalence_aware_fresh_holdout",
    "score_equivalence_aware_fresh_holdout",
}
POST_SELECTION_OPERATIONS = {
    "evaluate_outer",
    "decompose_representation",
    "decompose_oracle_raw_evidence",
    "decompose_local_frame_canonicalization",
    "decompose_latent_frame_inference",
    "decompose_candidate_frame_ambiguity",
    "fit_equivalence_aware_latent_frame",
    "generate_equivalence_aware_fresh_holdout",
    "predict_equivalence_aware_fresh_holdout",
    "score_equivalence_aware_fresh_holdout",
}
FINAL_METHOD_FREEZE_OPERATIONS = {
    "generate_equivalence_aware_fresh_holdout",
    "predict_equivalence_aware_fresh_holdout",
    "score_equivalence_aware_fresh_holdout",
}
FRESH_HOLDOUT_IDENTITY_OPERATIONS = {
    "predict_equivalence_aware_fresh_holdout",
    "score_equivalence_aware_fresh_holdout",
}
PREDICTION_FREEZE_OPERATIONS = {"score_equivalence_aware_fresh_holdout"}
TERMINAL_STATES = {"completed", "failed"}


def _require_sha(task: dict[str, Any], key: str, label: str) -> str:
    value = str(task.get(key, ""))
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(label)
    return value


def build_command(job: dict[str, Any]) -> list[str]:
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    if str(task.get("runner", "")) != "step14_training_selection":
        raise ValueError("Step-14 worker requires runner='step14_training_selection'")

    operation = str(task.get("command", ""))
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Unsupported Step-14 operation: {operation!r}")

    workspace = common.safe_workspace(task.get("workspace"))
    config = common.safe_config(task.get("config"))
    if config != CONFIG:
        raise ValueError("Step-14 worker is hard-pinned to the frozen Step-14 config")

    progress_every = int(task.get("progress_every", 5000))
    if progress_every < 1 or progress_every > 100000:
        raise ValueError("task.progress_every must be between 1 and 100000")

    if operation in POST_SELECTION_OPERATIONS:
        run_id = str(task.get("expected_training_run_id", ""))
        freeze_sha = str(task.get("expected_selection_freeze_sha256", ""))
        if not run_id.startswith("training_") or len(run_id) > 96:
            raise ValueError("post-selection Step-14 operation requires a frozen training run id")
        if not freeze_sha.startswith("sha256:") or len(freeze_sha) != 71:
            raise ValueError("post-selection Step-14 operation requires the frozen selection-freeze SHA-256")

        final_sha = None
        if operation in FINAL_METHOD_FREEZE_OPERATIONS:
            final_sha = _require_sha(
                task,
                "expected_final_method_freeze_payload_sha256",
                "final-freeze-gated Step-14 operation requires the final method freeze payload SHA-256",
            )
        elif task.get("expected_final_method_freeze_payload_sha256") is not None:
            raise ValueError("final-method-freeze identity is reserved for final-freeze-gated operations")

        product_id = None
        dataset_sha = None
        if operation in FRESH_HOLDOUT_IDENTITY_OPERATIONS:
            product_id = str(task.get("expected_fresh_holdout_product_id", ""))
            if not product_id.startswith("fresh_holdout_") or len(product_id) > 96:
                raise ValueError("fresh-holdout prediction/scoring requires the frozen fresh holdout product id")
            dataset_sha = _require_sha(
                task,
                "expected_fresh_holdout_dataset_complete_sha256",
                "fresh-holdout prediction/scoring requires the dataset-complete SHA-256",
            )
        elif task.get("expected_fresh_holdout_product_id") is not None or task.get("expected_fresh_holdout_dataset_complete_sha256") is not None:
            raise ValueError("fresh-holdout identity is reserved for fresh-holdout prediction/scoring")

        prediction_sha = None
        if operation in PREDICTION_FREEZE_OPERATIONS:
            prediction_sha = _require_sha(
                task,
                "expected_oracle_free_prediction_complete_sha256",
                "confirmatory scoring requires the oracle-free prediction completion SHA-256",
            )
        elif task.get("expected_oracle_free_prediction_complete_sha256") is not None:
            raise ValueError("oracle-free prediction completion identity is reserved for confirmatory scoring")

        if operation == "generate_equivalence_aware_fresh_holdout":
            script = REPO_ROOT / "scripts" / "v0_2" / "generate_step14_equivalence_aware_fresh_holdout_redacted.py"
            return [
                sys.executable, str(script),
                "--training-run-id", run_id,
                "--selection-freeze-sha256", freeze_sha,
                "--final-method-freeze-payload-sha256", str(final_sha),
                "--progress-every", str(progress_every),
            ]

        if operation == "predict_equivalence_aware_fresh_holdout":
            script = REPO_ROOT / "scripts" / "v0_2" / "predict_step14_equivalence_aware_fresh_holdout_blind.py"
            return [
                sys.executable, str(script),
                "--training-run-id", run_id,
                "--selection-freeze-sha256", freeze_sha,
                "--final-method-freeze-payload-sha256", str(final_sha),
                "--fresh-holdout-product-id", str(product_id),
                "--fresh-holdout-dataset-complete-sha256", str(dataset_sha),
                "--device", "cuda",
                "--progress-every", str(progress_every),
            ]

        if operation == "score_equivalence_aware_fresh_holdout":
            script = REPO_ROOT / "scripts" / "v0_2" / "score_step14_equivalence_aware_fresh_holdout_confirmatory_cached.py"
            return [
                sys.executable, str(script),
                "--training-run-id", run_id,
                "--selection-freeze-sha256", freeze_sha,
                "--final-method-freeze-payload-sha256", str(final_sha),
                "--fresh-holdout-product-id", str(product_id),
                "--fresh-holdout-dataset-complete-sha256", str(dataset_sha),
                "--oracle-free-prediction-complete-sha256", str(prediction_sha),
                "--device", "cuda",
                "--progress-every", str(progress_every),
            ]

        if operation == "evaluate_outer":
            script = REPO_ROOT / "scripts" / "v0_2" / "run_step14_frozen_outer_pipeline.py"
            return [
                sys.executable,
                str(script),
                "--training-run-id", run_id,
                "--selection-freeze-sha256", freeze_sha,
                "--progress-every", str(progress_every),
            ]
        if operation == "decompose_representation":
            script = REPO_ROOT / "scripts" / "v0_2" / "analyze_step14_representation_fusion_head.py"
        elif operation == "decompose_oracle_raw_evidence":
            script = REPO_ROOT / "scripts" / "v0_2" / "run_step14_oracle_raw_evidence_ceiling.py"
        elif operation == "decompose_local_frame_canonicalization":
            script = REPO_ROOT / "scripts" / "v0_2" / "analyze_step14_local_frame_canonicalization.py"
        elif operation == "decompose_latent_frame_inference":
            script = REPO_ROOT / "scripts" / "v0_2" / "analyze_step14_latent_frame_inference.py"
        elif operation == "decompose_candidate_frame_ambiguity":
            script = REPO_ROOT / "scripts" / "v0_2" / "analyze_step14_candidate_frame_ambiguity.py"
        else:
            script = REPO_ROOT / "scripts" / "v0_2" / "fit_step14_equivalence_aware_latent_frame_strict.py"
        return [
            sys.executable,
            str(script),
            "--training-run-id", run_id,
            "--selection-freeze-sha256", freeze_sha,
            "--device", "cuda",
            "--progress-every", str(progress_every),
        ]

    if task.get("expected_training_run_id") is not None or task.get("expected_selection_freeze_sha256") is not None:
        raise ValueError("baseline/training stage may not consume a Step-14 selection freeze")
    for key in (
        "expected_final_method_freeze_payload_sha256",
        "expected_fresh_holdout_product_id",
        "expected_fresh_holdout_dataset_complete_sha256",
        "expected_oracle_free_prediction_complete_sha256",
    ):
        if task.get(key) is not None:
            raise ValueError(f"baseline/training stage may not consume {key}")

    if operation == "evaluate_pretraining_baseline":
        script = REPO_ROOT / "scripts" / "v0_2" / "evaluate_step14_pretraining_baseline.py"
    else:
        script = REPO_ROOT / "scripts" / "v0_2" / "run_step14_cross_motif_training.py"

    return [
        sys.executable,
        str(script),
        "--config", config,
        "--output-parent", str(workspace),
        "--device", "cuda",
        "--progress-every", str(progress_every),
    ]


def _existing_terminal_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(value, dict) and str(value.get("state", "")) in TERMINAL_STATES:
        return value
    return None


def _terminal_returncode(status: dict[str, Any]) -> int:
    if str(status.get("state")) == "completed":
        return 0
    value = status.get("returncode")
    if isinstance(value, int) and value != 0:
        return value
    return 1


def run() -> int:
    job = common.load_job()
    job_id = common.safe_job_id(job.get("id"))
    control_run_id = common.safe_job_id(job.get("_control_run_id") or job_id)
    command = build_command(job)
    task = job["task"]
    operation = str(task["command"])

    run_dir = CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "worker.log"

    previous = _existing_terminal_status(status_path)
    if previous is not None:
        print(json.dumps({
            "job_id": job_id,
            "control_run_id": control_run_id,
            "state": "restart-refused-terminal-status-preserved",
            "terminal_state": previous.get("state"),
            "log_path": previous.get("log_path", str(log_path)),
        }, sort_keys=True), flush=True)
        return _terminal_returncode(previous)

    common.atomic_json(run_dir / "job.json", job)
    started = {
        "schema_version": 2,
        "job_id": job_id,
        "control_run_id": control_run_id,
        "runner": "step14_training_selection",
        "scientific_operation": operation,
        "state": "running",
        "started_at": common.utc_now(),
        "repo_root": str(REPO_ROOT),
        "command": command,
        "gpu": common.gpu_snapshot(),
    }
    common.atomic_json(status_path, started)

    try:
        returncode = common.run_subprocess(command, log_path)
        completed = {
            **started,
            "state": "completed" if returncode == 0 else "failed",
            "completed_at": common.utc_now(),
            "returncode": returncode,
            "log_path": str(log_path),
        }
        common.atomic_json(status_path, completed)
        return returncode
    except BaseException as exc:
        failure = {
            **started,
            "state": "failed",
            "completed_at": common.utc_now(),
            "returncode": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "log_path": str(log_path),
        }
        common.atomic_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
