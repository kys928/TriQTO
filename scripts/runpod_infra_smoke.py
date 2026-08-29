#!/usr/bin/env python3
"""Launch the typed TriQTO infrastructure smoke worker on one RunPod GPU.

This is intentionally separate from the scientific Phase 15.6 runner so the
first compute validation does not inherit historical campaign disk thresholds.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import secrets
import time

import runpod_control_v2 as control


def load_request(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Smoke request must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = load_request(args.request)

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
        raise ValueError(f"Smoke request contains disallowed GPU types: {disallowed}")

    job_id = str(request.get("id") or f"runpod-infra-smoke-{int(time.time())}")
    if not job_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("Smoke request id contains unsafe characters")
    control_run_id = f"run-{int(time.time())}-{secrets.token_hex(4)}"
    worker_job = {
        "version": 1,
        "id": job_id,
        "kind": "compute",
        "task": {"runner": "infra_smoke", "command": "check"},
        "gpu": {"type_ids": requested, "count": 1},
        "lifecycle": {"mode": "detached"},
        "_control_run_id": control_run_id,
    }
    encoded = base64.b64encode(
        json.dumps(worker_job, sort_keys=True).encode("utf-8")
    ).decode("ascii")

    image = control.required_env("RUNPOD_IMAGE_NAME")
    payload = {
        "name": f"triqto-{job_id}"[:191],
        "computeType": "GPU",
        "cloudType": str(request.get("cloud_type", "SECURE")).upper(),
        "gpuTypeIds": requested,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "dataCenterIds": [control.required_env("RUNPOD_DATACENTER_ID")],
        "dataCenterPriority": "availability",
        "networkVolumeId": control.required_env("RUNPOD_NETWORK_VOLUME_ID"),
        "volumeMountPath": "/workspace",
        "containerDiskInGb": int(request.get("container_disk_gb", 50)),
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": ["cd /opt/TriQTO && python scripts/runpod_worker.py"],
        "ports": [],
        "env": {"TRIQTO_JOB_B64": encoded, "PYTHONUNBUFFERED": "1"},
        "interruptible": bool(request.get("interruptible", False)),
        "imageName": image,
    }

    registry_auth = __import__("os").environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "").strip()
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
            "runner": "infra_smoke",
            "launched_at": control.utc_now(),
            "cost_per_hour": cost,
            "network_volume_id": control.required_env("RUNPOD_NETWORK_VOLUME_ID"),
            "datacenter": control.required_env("RUNPOD_DATACENTER_ID"),
            "status_key": control.status_key(job_id, control_run_id),
            "requested_gpu_types": requested,
        }
        control.internal_put_json(control.active_key(control_run_id), record)
        registered = True
        print(json.dumps(record, indent=2))
    finally:
        if pod_id and not registered:
            control.delete_pod(pod_id, best_effort=True)


if __name__ == "__main__":
    main()
