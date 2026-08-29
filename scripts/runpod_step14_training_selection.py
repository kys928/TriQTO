#!/usr/bin/env python3
"""Launch frozen Step-14 baseline evaluation or fit/selection training on RunPod.

Only two scientific operations are accepted:
- evaluate_pretraining_baseline
- train_selection

Neither operation may materialize Step-14 outer data, inspect the future-hardware
reserve, or access a QPU.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import time

import runpod_control_v2 as control

WORKSPACE = "/workspace/triqto-data/step14_cross_motif_training"
CONFIG = "configs/v0_2/step14_cross_motif_generalization_training.json"
ALLOWED_OPERATIONS = {"evaluate_pretraining_baseline", "train_selection"}
ALLOWED_REQUEST_KEYS = {
    "id",
    "operation",
    "gpu_type_ids",
    "container_disk_gb",
    "interruptible",
    "progress_every",
}


def load_request(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Step-14 training request must be a JSON object")
    unexpected = sorted(set(value) - ALLOWED_REQUEST_KEYS)
    if unexpected:
        raise ValueError(f"Unsupported Step-14 training request field(s): {unexpected}")
    operation = str(value.get("operation", ""))
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Unsupported Step-14 training operation: {operation!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = load_request(args.request)
    operation = str(request["operation"])

    allowed = [
        value.strip()
        for value in control.required_env("RUNPOD_ALLOWED_GPU_TYPES").split(",")
        if value.strip()
    ]
    requested = request.get("gpu_type_ids") or allowed
    if not isinstance(requested, list) or not requested:
        raise ValueError("No GPU types requested or allowed")
    requested = [str(value) for value in requested]
    disallowed = [value for value in requested if value not in allowed]
    if disallowed:
        raise ValueError(f"Step-14 request contains disallowed GPU types: {disallowed}")

    default_id = (
        "step14-pretraining-baseline"
        if operation == "evaluate_pretraining_baseline"
        else "step14-fit-selection-training"
    )
    job_id = str(request.get("id") or f"{default_id}-{int(time.time())}")
    allowed_id_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not job_id or len(job_id) > 96 or any(ch not in allowed_id_chars for ch in job_id):
        raise ValueError("Step-14 request id contains unsafe characters")

    progress_every = int(request.get("progress_every", 5000))
    if progress_every < 1 or progress_every > 100000:
        raise ValueError("progress_every must be between 1 and 100000")
    container_disk_gb = int(request.get("container_disk_gb", 50))
    if container_disk_gb < 20 or container_disk_gb > 200:
        raise ValueError("container_disk_gb must be between 20 and 200")

    control_run_id = f"run-{int(time.time())}-{secrets.token_hex(4)}"
    worker_job = {
        "version": 1,
        "id": job_id,
        "kind": "compute",
        "task": {
            "runner": "step14_training_selection",
            "command": operation,
            "workspace": WORKSPACE,
            "config": CONFIG,
            "progress_every": progress_every,
        },
        "gpu": {"type_ids": requested, "count": 1},
        "lifecycle": {"mode": "detached"},
        "_control_run_id": control_run_id,
    }
    encoded = base64.b64encode(
        json.dumps(worker_job, sort_keys=True).encode("utf-8")
    ).decode("ascii")

    payload = {
        "name": f"triqto-{job_id}"[:191],
        "computeType": "GPU",
        "cloudType": "SECURE",
        "gpuTypeIds": requested,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "dataCenterIds": [control.required_env("RUNPOD_DATACENTER_ID")],
        "dataCenterPriority": "availability",
        "networkVolumeId": control.required_env("RUNPOD_NETWORK_VOLUME_ID"),
        "volumeMountPath": "/workspace",
        "containerDiskInGb": container_disk_gb,
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [
            "cd /opt/TriQTO && python scripts/runpod_step14_training_worker.py"
        ],
        "ports": [],
        "env": {"TRIQTO_JOB_B64": encoded, "PYTHONUNBUFFERED": "1"},
        "interruptible": bool(request.get("interruptible", False)),
        "imageName": control.required_env("RUNPOD_IMAGE_NAME"),
    }
    registry_auth = os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "").strip()
    if registry_auth:
        payload["containerRegistryAuthId"] = registry_auth

    pod_id = None
    registered = False
    try:
        pod = control.runpod_request("POST", "/pods", payload)
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RuntimeError(f"RunPod create response did not contain Pod id: {pod!r}")
        pod_id = str(pod["id"])
        raw_cost = pod.get("adjustedCostPerHr")
        if raw_cost is None:
            raw_cost = pod.get("costPerHr")
        if raw_cost is None:
            raise RuntimeError("RunPod did not report the Pod hourly price")
        cost = float(raw_cost)
        record = {
            "schema_version": 2,
            "job_id": job_id,
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "lifecycle_mode": "detached",
            "runner": "step14_training_selection",
            "scientific_operation": operation,
            "launched_at": control.utc_now(),
            "cost_per_hour": cost,
            "network_volume_id": control.required_env("RUNPOD_NETWORK_VOLUME_ID"),
            "datacenter": control.required_env("RUNPOD_DATACENTER_ID"),
            "status_key": control.status_key(job_id, control_run_id),
            "requested_gpu_types": requested,
            "workspace": WORKSPACE,
            "protocol_config": CONFIG,
        }
        control.internal_put_json(control.active_key(control_run_id), record)
        registered = True
        print(json.dumps(record, indent=2))
    finally:
        if pod_id and not registered:
            control.delete_pod(pod_id, best_effort=True)


if __name__ == "__main__":
    main()
