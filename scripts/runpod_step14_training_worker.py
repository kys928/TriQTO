#!/usr/bin/env python3
"""Restricted RunPod worker for Step-14 baseline and fit/selection training only."""
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
ALLOWED_OPERATIONS = {"evaluate_pretraining_baseline", "train_selection"}


def build_command(job: dict[str, Any]) -> list[str]:
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    if str(task.get("runner", "")) != "step14_training_selection":
        raise ValueError("Step-14 training worker requires runner='step14_training_selection'")

    operation = str(task.get("command", ""))
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Unsupported Step-14 training operation: {operation!r}")
    if task.get("selection_freeze") is not None:
        raise ValueError("Baseline/training stage may not consume a Step-14 selection freeze")

    workspace = common.safe_workspace(task.get("workspace"))
    config = common.safe_config(task.get("config"))
    if config != CONFIG:
        raise ValueError("Step-14 training worker is hard-pinned to the frozen Step-14 config")

    progress_every = int(task.get("progress_every", 5000))
    if progress_every < 1 or progress_every > 100000:
        raise ValueError("task.progress_every must be between 1 and 100000")

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
