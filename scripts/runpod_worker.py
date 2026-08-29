#!/usr/bin/env python3
"""Restricted TriQTO worker executed inside an ephemeral RunPod Pod."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
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


def build_phase15_6_command(job: dict[str, Any]) -> list[str]:
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
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


def build_step14_cross_motif_command(job: dict[str, Any]) -> list[str]:
    """Build the one enabled Step-14 operation: fit+selection generation only."""
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    if str(task.get("command", "")) != "generate_development":
        raise ValueError("Step-14 RunPod runner enables only command='generate_development'")
    if task.get("selection_freeze") is not None:
        raise ValueError("Step-14 development generation may not consume a selection freeze")
    requested_mode = task.get("mode")
    if requested_mode not in {None, "development"}:
        raise ValueError("Step-14 RunPod runner is hard-pinned to development mode")

    output_parent = safe_workspace(task.get("workspace"))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "v0_2" / "generate_step14_cross_motif_dataset.py"),
        "--mode",
        "development",
        "--output-parent",
        str(output_parent),
    ]

    config = safe_config(task.get("config"))
    if config is not None:
        cmd.extend(["--config", config])

    progress_every = int(task.get("progress_every", 25))
    if progress_every < 1 or progress_every > 10000:
        raise ValueError("task.progress_every must be between 1 and 10000")
    cmd.extend(["--progress-every", str(progress_every)])
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
        return {
            "returncode": result.returncode,
            "output": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def memory_total_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return round(kib / 1024**2, 3)
    except Exception:
        return None
    return None


def infra_smoke(run_dir: Path) -> dict[str, Any]:
    import torch
    import qiskit
    import qiskit_aer
    import triqto  # noqa: F401

    disk = shutil.disk_usage("/workspace")
    control_probe = run_dir / "write_probe.txt"
    control_probe.write_text("TriQTO RunPod control-plane write probe\n", encoding="utf-8")
    probe_ok = control_probe.read_text(encoding="utf-8").startswith("TriQTO RunPod")
    control_probe.unlink()

    result = {
        "checked_at": utc_now(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_total_gib": memory_total_gib(),
        "workspace": {
            "path": "/workspace",
            "total_gib": round(disk.total / 1024**3, 3),
            "used_gib": round(disk.used / 1024**3, 3),
            "free_gib": round(disk.free / 1024**3, 3),
            "control_write_probe": probe_ok,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "gpu": gpu_snapshot(),
        "repo_root": str(REPO_ROOT),
        "triqto_import": True,
    }
    atomic_json(run_dir / "infra_smoke.json", result)

    if not probe_ok:
        raise RuntimeError("Network Volume control write/read probe failed")
    if not result["torch"]["cuda_available"]:
        raise RuntimeError("CUDA is not available inside the RunPod GPU worker")
    if int(result["torch"]["device_count"]) < 1:
        raise RuntimeError("No CUDA GPU devices are visible to PyTorch")
    return result


def run_subprocess(command: list[str], log_path: Path) -> int:
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
        return process.wait()


def run() -> int:
    job = load_job()
    job_id = safe_job_id(job.get("id"))
    control_run_id = safe_job_id(job.get("_control_run_id") or job_id)
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job.task must be an object")
    runner = str(task.get("runner", ""))
    if runner not in {"phase15_6", "infra_smoke", "step14_cross_motif"}:
        raise ValueError(
            "Only task.runner='phase15_6', 'infra_smoke', or 'step14_cross_motif' is enabled"
        )

    run_dir = CONTROL_ROOT / job_id / control_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "worker.log"
    atomic_json(run_dir / "job.json", job)

    if runner == "infra_smoke":
        command = ["internal:infra_smoke"]
    elif runner == "step14_cross_motif":
        command = build_step14_cross_motif_command(job)
    else:
        command = build_phase15_6_command(job)

    started = {
        "schema_version": 2,
        "job_id": job_id,
        "control_run_id": control_run_id,
        "runner": runner,
        "state": "running",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "repo_root": str(REPO_ROOT),
        "command": command,
        "gpu": gpu_snapshot(),
    }
    atomic_json(status_path, started)

    try:
        if runner == "infra_smoke":
            result = infra_smoke(run_dir)
            log_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            returncode = 0
        else:
            returncode = run_subprocess(command, log_path)

        completed = {
            **started,
            "state": "completed" if returncode == 0 else "failed",
            "completed_at": utc_now(),
            "returncode": returncode,
            "log_path": str(log_path),
            "result_path": str(run_dir / "infra_smoke.json") if runner == "infra_smoke" else None,
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
