#!/usr/bin/env python3
"""Restricted detached RunPod worker for Step-15 frame ranking only."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
import traceback

import runpod_worker as common

ALLOWED_OPERATIONS = {"fit_and_verify", "evaluate_spent_holdout", "diagnose_step14_replay"}


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
                    str(v02 / "fit_step15_frame_ranker_true_positive.py"),
                    "--device",
                    "cuda",
                    "--progress-every",
                    "1000",
                ],
            ),
            ("verify", verifier),
        ]
    if operation == "diagnose_step14_replay":
        return [
            ("pre_diagnostic_verify", verifier),
            ("step14_replay_diagnostic", [
                python, str(v02 / "diagnose_step14_phasea_replay.py"),
                "--device", "cuda", "--max-controls", "20",
            ]),
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
    # Network volume survives container restarts. An exclusive sentinel ensures a
    # failed/restarted scientific subprocess is never silently rerun under the
    # same control-run ID, which previously overwrote a terminal failure status.
    start_guard = run_dir / "step15_start_guard.json"
    guard = {
        "job_id": job_id,
        "control_run_id": control_run_id,
        "operation": operation,
        "image_name": str((job.get("provenance") or {}).get("image_name")),
        "first_started_at": common.utc_now(),
    }
    try:
        fd = os.open(str(start_guard), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not start_guard.is_file():
            raise
        prior_guard = json.loads(start_guard.read_text(encoding="utf-8"))
        for key in ("job_id", "control_run_id", "operation", "image_name"):
            if prior_guard.get(key) != guard[key]:
                raise RuntimeError("Step-15 restart guard identity drift: " + key)
        prior_status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file() else None
        )
        if not prior_status or prior_status.get("state") not in {"completed", "failed"}:
            common.atomic_json(status_path, {
                "schema_version": 2, "job_id": job_id,
                "control_run_id": control_run_id, "runner": "step15_frame_ranking",
                "scientific_operation": operation, "state": "failed",
                "reason": "restarted_before_terminal_scientific_status",
                "prior_status": prior_status,
                "failed_at": common.utc_now(),
            })
        print(json.dumps({
            "state": "RESTART_GUARD_BLOCKED", "job_id": job_id,
            "control_run_id": control_run_id,
            "prior_terminal_preserved": bool(prior_status and prior_status.get("state") in {"completed", "failed"}),
        }), flush=True)
        # The supervisor must observe the terminal status and delete this pod.
        # Do not re-execute scientific code, and do not overwrite the original log.
        while True:
            time.sleep(60)
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(guard, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
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
            else (
                Path("/workspace/triqto-data/step15_spent_holdout_audit/current_step15_spent_holdout_audit.json")
                if operation == "evaluate_spent_holdout"
                else Path("/workspace/triqto-data/step15_replay_diagnostics/current_step15_replay_diagnostic.json")
            )
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
