#!/usr/bin/env python3
"""Restricted TriQTO worker executed inside an ephemeral RunPod Pod."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = Path("/workspace/triqto-control/runs")
ALLOWED_COMMANDS = {"prepare", "preflight", "data", "train", "evaluate", "aggregate", "all"}
ALLOWED_WORKSPACE_ROOT = Path("/workspace/triqto-data")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_job() -> dict[str, Any]:
    encoded = os.environ.get("TRIQTO_JOB_B64", "")
    if not encoded:
        raise RuntimeError("TRIQTO_JOB_B64 is not set")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        job = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("TRIQTO_JOB_B64 is not valid base64-encoded JSON") from exc
    if not isinstance(job, dict):
        raise RuntimeError("Job payload must be a JSON object")
    return job


def safe_job_id(value: object) -> str:
    job_id = str(value or "")
    if not job_id or len(job_id) > 96:
        raise ValueError("job.id must be 1-96 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in job_id):
        raise ValueError("job.id contains unsafe characters")
    return job_id


def safe_workspace(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError("task.workspace must be absolute")
    resolved = path.resolve(strict=False)
    root = ALLOWED_WORKSPACE_ROOT.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"task.workspace must be {root} or a child of it")
    return resolved


def safe_config(value: object | None) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("task.config must be a safe repository-relative path")
    if not path.parts or path.parts[0] != "configs":
        raise ValueError("task.config must live under configs/")
    return path.as_posix()


def build_command(job: dict[str, Any]) -> list[str]:
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    if task.get("runner") != "phase15_6":
        raise ValueError("Only task.runner='phase15_6' is enabled in v1")

    command = str(task.get("command", ""))
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Unsupported Phase 15.6 command: {command!r}")

    workspace = safe_workspace(task.get("workspace"))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_phase15_6_campaign.py"),
        command,
        "--workspace",
        str(workspace),
    ]

    config = safe_config(task.get("config"))
    if config is not None:
        if command not in {"prepare", "all"}:
            raise ValueError("task.config is only valid for prepare/all")
        cmd.extend(["--config", config])

    if "seed" in task and task["seed"] is not None:
        if command not in {"train", "evaluate"}:
            raise ValueError("task.seed is only valid for train/evaluate")
        seed = int(task["seed"])
        if seed < 0 or seed > 2**31 - 1:
            raise ValueError("task.seed is outside the accepted range")
        cmd.extend(["--seed", str(seed)])

    return cmd


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return {"returncode": result.returncode, "output": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"error": repr(exc)}


def run() -> int:
    job = load_job()
    job_id = safe_job_id(job.get("id"))
    control_run_id = safe_job_id(job.get("_control_run_id") or job_id)
    run_dir = CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "worker.log"
    atomic_json(run_dir / "job.json", job)

    command = build_command(job)
    started = {
        "schema_version": 1,
        "job_id": job_id,
        "control_run_id": control_run_id,
        "state": "running",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "repo_root": str(REPO_ROOT),
        "command": command,
        "gpu": gpu_snapshot(),
    }
    atomic_json(status_path, started)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()

        completed = {
            **started,
            "state": "completed" if returncode == 0 else "failed",
            "completed_at": utc_now(),
            "returncode": returncode,
            "log_path": str(log_path),
        }
        atomic_json(status_path, completed)
        return returncode
    except BaseException as exc:
        failure = {
            **started,
            "state": "failed",
            "completed_at": utc_now(),
            "returncode": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "log_path": str(log_path),
        }
        atomic_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
