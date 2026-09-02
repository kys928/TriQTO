#!/usr/bin/env python3
"""Restricted RunPod worker for frozen Step-14 scientific operations.

Allowed stages are deliberately typed: pretraining baseline, fit/selection
training, one-shot post-selection outer evaluation, or frozen post-selection
diagnostics that never update the main model.
"""
from __future__ import annotations

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
}
POST_SELECTION_OPERATIONS = {
    "evaluate_outer",
    "decompose_representation",
    "decompose_oracle_raw_evidence",
}


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
        if operation == "evaluate_outer":
            script = REPO_ROOT / "scripts" / "v0_2" / "run_step14_frozen_outer_pipeline.py"
            return [
                sys.executable,
                str(script),
                "--training-run-id",
                run_id,
                "--selection-freeze-sha256",
                freeze_sha,
                "--progress-every",
                str(progress_every),
            ]
        if operation == "decompose_representation":
            script = REPO_ROOT / "scripts" / "v0_2" / "analyze_step14_representation_fusion_head.py"
        else:
            script = REPO_ROOT / "scripts" / "v0_2" / "analyze_step14_oracle_raw_evidence_ceiling.py"
        return [
            sys.executable,
            str(script),
            "--training-run-id",
            run_id,
            "--selection-freeze-sha256",
            freeze_sha,
            "--device",
            "cuda",
            "--progress-every",
            str(progress_every),
        ]

    if task.get("expected_training_run_id") is not None or task.get("expected_selection_freeze_sha256") is not None:
        raise ValueError("baseline/training stage may not consume a Step-14 selection freeze")

    if operation == "evaluate_pretraining_baseline":
        script = REPO_ROOT / "scripts" / "v0_2" / "evaluate_step14_pretraining_baseline.py"
    else:
        script = REPO_ROOT / "scripts" / "v0_2" / "run_step14_cross_motif_training.py"

    return [
        sys.executable,
        str(script),
        "--config",
        config,
        "--output-parent",
        str(workspace),
        "--device",
        "cuda",
        "--progress-every",
        str(progress_every),
    ]


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
