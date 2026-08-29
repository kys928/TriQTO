#!/usr/bin/env python3
"""TriQTO RunPod control plane v2.

Design goals:
- GitHub Actions holds all RunPod/S3 credentials.
- Persistent research data lives on a RunPod Network Volume.
- Compute defaults to a detached lifecycle, so experiment duration is not tied
  to a GitHub-hosted runner timeout.
- Unset cost/runtime variables mean no controller-imposed scientific cap.
- A lightweight reconciler deletes Pods when workers report terminal status.
- User-facing S3 operations remain read-only.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError as exc:
    raise SystemExit("boto3 is required: python -m pip install boto3") from exc

RUNPOD_REST = "https://rest.runpod.io/v1"
DEFAULT_POLL_SECONDS = 20
ACTIVE_PREFIX = "triqto-control/active/"
HISTORY_PREFIX = "triqto-control/history/"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def optional_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def optional_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def load_job(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Job manifest must be a JSON object")
    if payload.get("version") != 1:
        raise ValueError("Only job manifest version 1 is supported")
    job_id = str(payload.get("id", ""))
    if not job_id or len(job_id) > 96:
        raise ValueError("job.id must be 1-96 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in job_id):
        raise ValueError("job.id contains unsafe characters")
    if payload.get("kind") not in {"compute", "storage"}:
        raise ValueError("job.kind must be 'compute' or 'storage'")
    return payload


def runpod_request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {
        "Authorization": f"Bearer {required_env('RUNPOD_API_KEY')}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{RUNPOD_REST}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"RunPod API {method} {path} failed: HTTP {exc.code}: {detail}"
        ) from exc


def s3_client():
    datacenter = required_env("RUNPOD_DATACENTER_ID")
    endpoint = os.environ.get("RUNPOD_S3_ENDPOINT", "").strip()
    if not endpoint:
        endpoint = f"https://s3api-{datacenter.lower()}.runpod.io/"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=datacenter.lower(),
        aws_access_key_id=required_env("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required_env("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


def bucket() -> str:
    return required_env("RUNPOD_NETWORK_VOLUME_ID")


def internal_put_json(key: str, payload: dict[str, Any]) -> None:
    if not (key.startswith(ACTIVE_PREFIX) or key.startswith(HISTORY_PREFIX)):
        raise ValueError("Internal writes are restricted to TriQTO control metadata")
    s3_client().put_object(
        Bucket=bucket(),
        Key=key,
        Body=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def internal_delete(key: str) -> None:
    if not key.startswith(ACTIVE_PREFIX):
        raise ValueError("Internal deletes are restricted to active control metadata")
    s3_client().delete_object(Bucket=bucket(), Key=key)


def read_json_object(key: str) -> dict[str, Any] | None:
    try:
        obj = s3_client().get_object(Bucket=bucket(), Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    raw = obj["Body"].read(2_000_000)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object at s3://{bucket()}/{key}")
    return value


def s3_key(value: object, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if text.startswith("s3://"):
        raise ValueError("Use a volume-relative path, not an s3:// URL")
    if text.startswith("/workspace/"):
        text = text[len("/workspace/") :]
    text = text.lstrip("/")
    if not text and allow_empty:
        return ""
    path = PurePosixPath(text)
    if not text or ".." in path.parts or "\x00" in text:
        raise ValueError("Unsafe or empty storage path")
    return path.as_posix()


def storage_job(job: dict[str, Any]) -> None:
    spec = job.get("storage")
    if not isinstance(spec, dict):
        raise ValueError("storage job requires a storage object")
    operation = str(spec.get("operation", ""))
    if operation not in {"list", "head", "read_text", "download"}:
        raise ValueError("Storage operations are read-only: list, head, read_text, download")

    client = s3_client()
    key = s3_key(spec.get("path", ""), allow_empty=operation == "list")

    if operation == "list":
        max_items = min(max(int(spec.get("max_items", 200)), 1), 1000)
        response = client.list_objects_v2(Bucket=bucket(), Prefix=key, MaxKeys=max_items)
        entries = [
            {
                "key": item.get("Key"),
                "size": item.get("Size"),
                "last_modified": item.get("LastModified").isoformat()
                if item.get("LastModified")
                else None,
                "etag": item.get("ETag"),
            }
            for item in response.get("Contents", [])
        ]
        print(json.dumps({"job_id": job["id"], "operation": operation, "entries": entries}, indent=2))
        return

    metadata = client.head_object(Bucket=bucket(), Key=key)
    head = {
        "key": key,
        "size": int(metadata.get("ContentLength", 0)),
        "etag": metadata.get("ETag"),
        "last_modified": metadata.get("LastModified").isoformat()
        if metadata.get("LastModified")
        else None,
        "content_type": metadata.get("ContentType"),
    }
    if operation == "head":
        print(json.dumps({"job_id": job["id"], "operation": operation, "object": head}, indent=2))
        return

    if operation == "read_text":
        hard_cap = optional_int("RUNPOD_MAX_READ_TEXT_BYTES") or 1_048_576
        requested = int(spec.get("max_bytes", min(hard_cap, 262_144)))
        max_bytes = min(max(requested, 1), hard_cap)
        if head["size"] > max_bytes:
            raise RuntimeError(
                f"Object is {head['size']} bytes, above read_text limit {max_bytes}; use download instead"
            )
        raw = client.get_object(Bucket=bucket(), Key=key)["Body"].read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise RuntimeError("Object exceeded read_text limit while downloading")
        print(
            json.dumps(
                {
                    "job_id": job["id"],
                    "operation": operation,
                    "object": head,
                    "text": raw.decode(str(spec.get("encoding", "utf-8"))),
                },
                indent=2,
            )
        )
        return

    hard_cap = optional_int("RUNPOD_MAX_ARTIFACT_BYTES") or 536_870_912
    requested = int(spec.get("max_bytes", hard_cap))
    max_bytes = min(max(requested, 1), hard_cap)
    if head["size"] > max_bytes:
        raise RuntimeError(f"Object is {head['size']} bytes, above download limit {max_bytes}")
    destination = Path("runpod_artifacts") / str(job["id"]) / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket(), key, str(destination))
    print(
        json.dumps(
            {
                "job_id": job["id"],
                "operation": operation,
                "object": head,
                "downloaded_to": str(destination),
            },
            indent=2,
        )
    )


def effective_cap(limits: dict[str, Any]) -> float | None:
    values: list[float] = []
    global_cap = optional_float("RUNPOD_MAX_HOURLY_USD")
    if global_cap is not None:
        if global_cap <= 0:
            raise ValueError("RUNPOD_MAX_HOURLY_USD must be positive when set")
        values.append(global_cap)
    if limits.get("max_hourly_usd") is not None:
        value = float(limits["max_hourly_usd"])
        if value <= 0:
            raise ValueError("limits.max_hourly_usd must be positive when set")
        values.append(value)
    return min(values) if values else None


def effective_timeout(limits: dict[str, Any]) -> int | None:
    values: list[int] = []
    global_timeout = optional_int("RUNPOD_MAX_RUNTIME_MINUTES")
    if global_timeout is not None:
        if global_timeout < 1:
            raise ValueError("RUNPOD_MAX_RUNTIME_MINUTES must be positive when set")
        values.append(global_timeout)
    if limits.get("timeout_minutes") is not None:
        value = int(limits["timeout_minutes"])
        if value < 1:
            raise ValueError("limits.timeout_minutes must be positive when set")
        values.append(value)
    return min(values) if values else None


def validate_compute(job: dict[str, Any]) -> tuple[dict[str, Any], float | None, int | None, str, str]:
    task = job.get("task")
    gpu = job.get("gpu")
    limits = job.get("limits") or {}
    lifecycle = job.get("lifecycle") or {}
    if not isinstance(task, dict) or not isinstance(gpu, dict):
        raise ValueError("compute jobs require task and gpu objects")
    if not isinstance(limits, dict) or not isinstance(lifecycle, dict):
        raise ValueError("limits and lifecycle must be objects when present")

    if task.get("runner") != "phase15_6":
        raise ValueError("v2 enables only task.runner='phase15_6'")
    if str(task.get("command", "")) not in {
        "prepare", "preflight", "data", "train", "evaluate", "aggregate", "all"
    }:
        raise ValueError("Unsupported Phase 15.6 command")

    requested_types = gpu.get("type_ids")
    if not isinstance(requested_types, list) or not requested_types:
        raise ValueError("gpu.type_ids must be a non-empty list")
    requested_types = [str(value) for value in requested_types]
    allowed_types = {
        value.strip()
        for value in required_env("RUNPOD_ALLOWED_GPU_TYPES").split(",")
        if value.strip()
    }
    disallowed = [value for value in requested_types if value not in allowed_types]
    if disallowed:
        raise ValueError(f"GPU type(s) not in RUNPOD_ALLOWED_GPU_TYPES: {disallowed}")
    if int(gpu.get("count", 1)) != 1:
        raise ValueError("v2 currently permits exactly one GPU per Pod")

    cloud_type = str(gpu.get("cloud_type", "SECURE")).upper()
    if cloud_type not in {"SECURE", "COMMUNITY"}:
        raise ValueError("gpu.cloud_type must be SECURE or COMMUNITY")

    mode = str(lifecycle.get("mode", "detached")).lower()
    if mode not in {"detached", "attached"}:
        raise ValueError("lifecycle.mode must be detached or attached")
    timeout_minutes = effective_timeout(limits)
    if mode == "attached" and timeout_minutes is None:
        raise ValueError("attached mode requires an explicit timeout; use detached for unbounded runs")

    control_run_id = f"run-{int(time.time())}-{secrets.token_hex(4)}"
    worker_job = dict(job)
    worker_job["_control_run_id"] = control_run_id
    encoded_job = base64.b64encode(
        json.dumps(worker_job, sort_keys=True).encode("utf-8")
    ).decode("ascii")

    payload: dict[str, Any] = {
        "name": f"triqto-{job['id']}"[:191],
        "computeType": "GPU",
        "cloudType": cloud_type,
        "gpuTypeIds": requested_types,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "dataCenterIds": [required_env("RUNPOD_DATACENTER_ID")],
        "dataCenterPriority": "availability",
        "networkVolumeId": required_env("RUNPOD_NETWORK_VOLUME_ID"),
        "volumeMountPath": "/workspace",
        "containerDiskInGb": int(gpu.get("container_disk_gb", 50)),
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": ["cd /opt/TriQTO && python scripts/runpod_worker.py"],
        "ports": [],
        "env": {"TRIQTO_JOB_B64": encoded_job, "PYTHONUNBUFFERED": "1"},
        "interruptible": bool(gpu.get("interruptible", False)),
    }

    template_id = os.environ.get("RUNPOD_TEMPLATE_ID", "").strip()
    image = os.environ.get("RUNPOD_IMAGE_NAME", "").strip()
    if template_id:
        payload["templateId"] = template_id
    elif image:
        payload["imageName"] = image
    else:
        raise RuntimeError("Set either RUNPOD_TEMPLATE_ID or RUNPOD_IMAGE_NAME")
    registry_auth = os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "").strip()
    if registry_auth:
        payload["containerRegistryAuthId"] = registry_auth

    return payload, effective_cap(limits), timeout_minutes, control_run_id, mode


def status_key(job_id: str, control_run_id: str) -> str:
    return f"triqto-control/runs/{job_id}/{control_run_id}/status.json"


def active_key(control_run_id: str) -> str:
    return f"{ACTIVE_PREFIX}{control_run_id}.json"


def history_key(control_run_id: str) -> str:
    return f"{HISTORY_PREFIX}{control_run_id}.json"


def delete_pod(pod_id: str, *, best_effort: bool = False) -> bool:
    try:
        runpod_request("DELETE", f"/pods/{pod_id}")
        print(json.dumps({"pod_id": pod_id, "cleanup": "deleted"}))
        return True
    except Exception as exc:
        if not best_effort:
            raise
        print(json.dumps({"pod_id": pod_id, "cleanup": "failed", "error": repr(exc)}))
        return False


def compute_job(job: dict[str, Any]) -> None:
    payload, hourly_cap, timeout_minutes, control_run_id, mode = validate_compute(job)
    pod_id: str | None = None
    detached_registered = False
    try:
        pod = runpod_request("POST", "/pods", payload)
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RuntimeError(f"RunPod create response did not contain a Pod id: {pod!r}")
        pod_id = str(pod["id"])
        raw_cost = pod.get("adjustedCostPerHr")
        if raw_cost is None:
            raw_cost = pod.get("costPerHr")
        cost = float(raw_cost) if raw_cost is not None else None
        if cost is None:
            raise RuntimeError("RunPod did not report Pod hourly cost; refusing an unpriced run")
        if hourly_cap is not None and cost > hourly_cap:
            raise RuntimeError(
                f"Allocated Pod costs ${cost:.4f}/h, above effective cap ${hourly_cap:.4f}/h"
            )

        launch = {
            "schema_version": 2,
            "job_id": str(job["id"]),
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "lifecycle_mode": mode,
            "launched_at": utc_now(),
            "cost_per_hour": cost,
            "hourly_cap": hourly_cap,
            "timeout_minutes": timeout_minutes,
            "network_volume_id": required_env("RUNPOD_NETWORK_VOLUME_ID"),
            "datacenter": required_env("RUNPOD_DATACENTER_ID"),
            "status_key": status_key(str(job["id"]), control_run_id),
            "requested_gpu_types": payload["gpuTypeIds"],
        }
        print(json.dumps(launch, indent=2))

        if mode == "detached":
            internal_put_json(active_key(control_run_id), launch)
            detached_registered = True
            print(json.dumps({
                "job_id": job["id"],
                "control_run_id": control_run_id,
                "state": "detached",
                "message": "Pod continues independently; reconciler will delete it after terminal worker status.",
            }, indent=2))
            return

        assert timeout_minutes is not None
        deadline = time.monotonic() + timeout_minutes * 60
        poll_seconds = max(optional_int("RUNPOD_STATUS_POLL_SECONDS") or DEFAULT_POLL_SECONDS, 5)
        last_state: str | None = None
        while time.monotonic() < deadline:
            status = read_json_object(status_key(str(job["id"]), control_run_id))
            if status is not None:
                state = str(status.get("state", ""))
                if state != last_state:
                    print(json.dumps({"job_id": job["id"], "worker_status": status}, indent=2))
                    last_state = state
                if state == "completed":
                    return
                if state == "failed":
                    raise RuntimeError(f"TriQTO worker failed: {json.dumps(status, sort_keys=True)}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Attached job exceeded timeout of {timeout_minutes} minutes")
    finally:
        if pod_id and not detached_registered:
            delete_pod(pod_id, best_effort=True)


def list_active_records() -> list[tuple[str, dict[str, Any]]]:
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    records: list[tuple[str, dict[str, Any]]] = []
    for page in paginator.paginate(Bucket=bucket(), Prefix=ACTIVE_PREFIX):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if not key.endswith(".json"):
                continue
            value = read_json_object(key)
            if value is not None:
                records.append((key, value))
    return records


def archive_and_remove(key: str, record: dict[str, Any], terminal: dict[str, Any]) -> None:
    control_run_id = str(record["control_run_id"])
    archive = {
        **record,
        "reconciled_at": utc_now(),
        "terminal_status": terminal,
    }
    internal_put_json(history_key(control_run_id), archive)
    internal_delete(key)


def reconcile_detached() -> None:
    records = list_active_records()
    print(json.dumps({"active_run_count": len(records), "checked_at": utc_now()}))
    for key, record in records:
        job_id = str(record.get("job_id", ""))
        control_run_id = str(record.get("control_run_id", ""))
        pod_id = str(record.get("pod_id", ""))
        if not job_id or not control_run_id or not pod_id:
            print(json.dumps({"active_key": key, "state": "invalid-record"}))
            continue
        status = read_json_object(str(record.get("status_key") or status_key(job_id, control_run_id)))
        state = str(status.get("state", "")) if status else "pending"
        print(json.dumps({
            "job_id": job_id,
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "worker_state": state,
        }))
        if state not in {"completed", "failed"}:
            continue
        deleted = delete_pod(pod_id, best_effort=True)
        if deleted:
            archive_and_remove(key, record, status or {"state": state})


def terminate_run(control_run_id: str) -> None:
    key = active_key(control_run_id)
    record = read_json_object(key)
    if record is None:
        raise RuntimeError(f"No active detached run {control_run_id}")
    pod_id = str(record.get("pod_id", ""))
    if not pod_id:
        raise RuntimeError("Active record is missing pod_id")
    if delete_pod(pod_id, best_effort=False):
        archive_and_remove(
            key,
            record,
            {"state": "terminated", "reason": "manual", "terminated_at": utc_now()},
        )


def execute(path: Path) -> None:
    job = load_job(path)
    print(json.dumps({"job": str(path), "job_id": job["id"], "kind": job["kind"]}))
    if job["kind"] == "storage":
        storage_job(job)
    else:
        compute_job(job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    execute_parser = subparsers.add_parser("execute", help="Validate and execute one manifest")
    execute_parser.add_argument("--job", type=Path, required=True)

    subparsers.add_parser("reconcile-detached", help="Delete detached Pods with terminal worker status")

    terminate_parser = subparsers.add_parser("terminate-run", help="Explicitly terminate one detached run")
    terminate_parser.add_argument("--control-run-id", required=True)

    args = parser.parse_args()
    if args.command == "execute":
        execute(args.job)
    elif args.command == "reconcile-detached":
        reconcile_detached()
    else:
        terminate_run(args.control_run_id)


if __name__ == "__main__":
    main()
