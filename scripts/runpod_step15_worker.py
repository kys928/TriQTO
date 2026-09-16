#!/usr/bin/env python3
"""Restricted detached RunPod worker for Step-15 frame ranking only."""
from __future__ import annotations

import json
from pathlib import Path
import traceback

import runpod_worker as common

ALLOWED_OPERATIONS = {"fit_and_verify", "evaluate_spent_holdout"}


def build_commands(job: dict) -> list[tuple[str, list[str]]]:
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    if str(task.get("runner", "")) != "step15_frame_ranking":
        raise ValueError("Step-15 worker requires task.runner='step15_frame_ranking'")
    operation = str(task.get("operation", ""))
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Unsupported Step-15 operation: {operation!r}")

    python = common.sys.executable
    v02 = common.REPO_ROOT / "scripts" / "v0_2"
    verifier = [python, str(v02 / "verify_step15_frame_ranker_freeze.py")]
    if operation == "fit_and_verify":
        return [
            (
                "fit",
                [
                    python,
                    str(v02 / "fit_step15_frame_ranker.py"),
                    "--device",
                    "cuda",
                    "--progress-every",
                    "1000",
                ],
            ),
            ("verify", verifier),
        ]
    return [
        ("pre_evaluation_verify", verifier),
        (
            "spent_holdout_evaluation",
            [
                python,
                str(v02 / "evaluate_step15_spent_holdout.py"),
                "--device",
                "cuda",
                "--progress-every",
                "1000",
            ],
        ),
    ]


def run() -> int:
    job = common.load_job()
    job_id = common.safe_job_id(job.get("id"))
    control_run_id = common.safe_job_id(job.get("_control_run_id") or job_id)
    commands = build_commands(job)
    task = job["task"]
    operation = str(task["operation"])

    run_dir = common.CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    common.atomic_json(run_dir / "job.json", job)

    started = {
        "schema_version": 2,
        "job_id": job_id,
        "control_run_id": control_run_id,
        "runner": "step15_frame_ranking",
        "scientific_operation": operation,
        "state": "running",
        "started_at": common.utc_now(),
        "repo_root": str(common.REPO_ROOT),
        "commands": [command for _name, command in commands],
        "gpu": common.gpu_snapshot(),
        "provenance": dict(job.get("provenance") or {}),
    }
    common.atomic_json(status_path, started)

    completed_steps: list[dict] = []
    try:
        for index, (name, command) in enumerate(commands, start=1):
            log_path = run_dir / f"step15_{index:02d}_{name}.log"
            returncode = common.run_subprocess(command, log_path)
            completed_steps.append(
                {
                    "name": name,
                    "returncode": int(returncode),
                    "log_path": str(log_path),
                }
            )
            if returncode != 0:
                completed = {
                    **started,
                    "state": "failed",
                    "completed_at": common.utc_now(),
                    "returncode": int(returncode),
                    "completed_steps": completed_steps,
                }
                common.atomic_json(status_path, completed)
                return int(returncode)

        result_pointer = (
            Path("/workspace/triqto-data/step15_frame_ranking/current_step15_frame_ranker.json")
            if operation == "fit_and_verify"
            else Path("/workspace/triqto-data/step15_spent_holdout_audit/current_step15_spent_holdout_audit.json")
        )
        completed = {
            **started,
            "state": "completed",
            "completed_at": common.utc_now(),
            "returncode": 0,
            "completed_steps": completed_steps,
            "result_pointer": str(result_pointer),
            "result_pointer_exists": result_pointer.is_file(),
        }
        if not result_pointer.is_file():
            raise RuntimeError(f"Step-15 operation completed without result pointer: {result_pointer}")
        common.atomic_json(status_path, completed)
        return 0
    except BaseException as exc:
        failure = {
            **started,
            "state": "failed",
            "completed_at": common.utc_now(),
            "returncode": None,
            "completed_steps": completed_steps,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        common.atomic_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
