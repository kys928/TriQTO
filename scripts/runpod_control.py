#!/usr/bin/env python3
"""GitHub-side RunPod + S3 control plane for TriQTO.

This process keeps infrastructure credentials in GitHub Actions. GPU workers
receive only a validated job payload and a mounted network volume.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
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
ACTIVE_PODS_FILE = Path(os.environ.get("RUNPOD_ACTIVE_PODS_FILE", ".runpod-active-pods.json"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def optional_float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def optional_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


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
    api_key = required_env("RUNPOD_API_KEY")
    data = None
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{RUNPOD_REST}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod API {method} {path} failed: HTTP {exc.code}: {detail}") from exc


def s3_client():
    datacenter = required_env("RUNPOD_DATACENTER_ID")
    endpoint = os.environ.get("RUNPOD_S3_ENDPOINT", "").strip()
    if not endpoint:
        endpoint = f"https://s3api-{datacenter.lower()}.runpod.io/"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=datacenter,
        aws_access_key_id=required_env("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required_env("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


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
        raise ValueError("v1 storage operations are read-only: list, head, read_text, download")

    bucket = required_env("RUNPOD_NETWORK_VOLUME_ID")
    client = s3_client()
    key = s3_key(spec.get("path", ""), allow_empty=operation == "list")

    if operation == "list":
        max_items = min(max(int(spec.get("max_items", 200)), 1), 1000)
        response = client.list_objects(Bucket=bucket, Prefix=key, MaxKeys=max_items)
        entries = []
        for item in response.get("Contents", []):
            entries.append(
                {
                    "key": item.get("Key"),
                    "size": item.get("Size"),
                    "last_modified": item.get("LastModified").isoformat()
                    if item.get("LastModified")
                    else None,
                    "etag": item.get("ETag"),
                }
            )
        print(json.dumps({"job_id": job["id"], "operation": operation, "entries": entries}, indent=2))
        return

    metadata = client.head_object(Bucket=bucket, Key=key)
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
        hard_cap = optional_int_env("RUNPOD_MAX_READ_TEXT_BYTES", 1_048_576)
        requested = int(spec.get("max_bytes", min(hard_cap, 262_144)))
        max_bytes = min(max(requested, 1), hard_cap)
        if head["size"] > max_bytes:
            raise RuntimeError(
                f"Object is {head['size']} bytes, above read_text limit {max_bytes}; use download instead"
            )
        obj = client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise RuntimeError("Object exceeded read_text limit while downloading")
        text = raw.decode(str(spec.get("encoding", "utf-8")))
        print(
            json.dumps(
                {"job_id": job["id"], "operation": operation, "object": head, "text": text},
                indent=2,
            )
        )
        return

    hard_cap = optional_int_env("RUNPOD_MAX_ARTIFACT_BYTES", 536_870_912)
    requested = int(spec.get("max_bytes", hard_cap))
    max_bytes = min(max(requested, 1), hard_cap)
    if head["size"] > max_bytes:
        raise RuntimeError(f"Object is {head['size']} bytes, above download limit {max_bytes}")
    destination = Path("runpod_artifacts") / str(job["id"]) / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
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


def active_pods() -> list[str]:
    if not ACTIVE_PODS_FILE.exists():
        return []
    try:
        value = json.loads(ACTIVE_PODS_FILE.read_text(encoding="utf-8"))
        return [str(item) for item in value if item]
    except Exception:
        return []


def write_active_pods(pods: list[str]) -> None:
    if pods:
        ACTIVE_PODS_FILE.write_text(json.dumps(sorted(set(pods)), indent=2) + "\n", encoding="utf-8")
    elif ACTIVE_PODS_FILE.exists():
        ACTIVE_PODS_FILE.unlink()


def remember_pod(pod_id: str) -> None:
    pods = active_pods()
    if pod_id not in pods:
        pods.append(pod_id)
    write_active_pods(pods)


def forget_pod(pod_id: str) -> None:
    write_active_pods([value for value in active_pods() if value != pod_id])


def delete_pod(pod_id: str, *, best_effort: bool = False) -> None:
    try:
        runpod_request("DELETE", f"/pods/{pod_id}")
        print(json.dumps({"pod_id": pod_id, "cleanup": "deleted"}))
    except Exception as exc:
        if not best_effort:
            raise
        print(json.dumps({"pod_id": pod_id, "cleanup": "failed", "error": repr(exc)}))
    else:
        forget_pod(pod_id)


def validate_compute(job: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
    task = job.get("task")
    gpu = job.get("gpu")
    limits = job.get("limits")
    if not isinstance(task, dict) or not isinstance(gpu, dict) or not isinstance(limits, dict):
        raise ValueError("compute jobs require task, gpu, and limits objects")

    if task.get("runner") != "phase15_6":
        raise ValueError("v1 enables only task.runner='phase15_6'")
    if str(task.get("command", "")) not in {
        "prepare",
        "preflight",
        "data",
        "train",
        "evaluate",
        "aggregate",
        "all",
    }:
        raise ValueError("Unsupported Phase 15.6 command")

    requested_types = gpu.get("type_ids")
    if not isinstance(requested_types, list) or not requested_types:
        raise ValueError("gpu.type_ids must be a non-empty list")
    requested_types = [str(value) for value in requested_types]

    allowed_raw = required_env("RUNPOD_ALLOWED_GPU_TYPES")
    allowed_types = {value.strip() for value in allowed_raw.split(",") if value.strip()}
    disallowed = [value for value in requested_types if value not in allowed_types]
    if disallowed:
        raise ValueError(f"GPU type(s) not in RUNPOD_ALLOWED_GPU_TYPES: {disallowed}")

    count = int(gpu.get("count", 1))
    if count != 1:
        raise ValueError("v1 control plane permits exactly one GPU per Pod")

    cloud_type = str(gpu.get("cloud_type", "SECURE")).upper()
    if cloud_type not in {"SECURE", "COMMUNITY"}:
        raise ValueError("gpu.cloud_type must be SECURE or COMMUNITY")

    global_cap = optional_float_env("RUNPOD_MAX_HOURLY_USD", 5.0)
    manifest_cap = float(limits.get("max_hourly_usd", global_cap))
    if manifest_cap <= 0:
        raise ValueError("limits.max_hourly_usd must be positive")
    effective_cap = min(global_cap, manifest_cap)

    global_timeout = optional_int_env("RUNPOD_MAX_RUNTIME_MINUTES", 360)
    timeout_minutes = int(limits.get("timeout_minutes", global_timeout))
    if timeout_minutes < 1 or timeout_minutes > global_timeout:
        raise ValueError(f"limits.timeout_minutes must be 1..{global_timeout}")

    encoded_job = base64.b64encode(json.dumps(job, sort_keys=True).encode("utf-8")).decode("ascii")
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
        "env": {
            "TRIQTO_JOB_B64": encoded_job,
            "PYTHONUNBUFFERED": "1",
        },
        "interruptible": bool(gpu.get("interruptible", False)),
    }

    image = os.environ.get("RUNPOD_IMAGE_NAME", "").strip()
    template_id = os.environ.get("RUNPOD_TEMPLATE_ID", "").strip()
    if template_id:
        payload["templateId"] = template_id
    elif image:
        payload["imageName"] = image
    else:
        raise RuntimeError("Set either RUNPOD_TEMPLATE_ID or RUNPOD_IMAGE_NAME")

    registry_auth = os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "").strip()
    if registry_auth:
        payload["containerRegistryAuthId"] = registry_auth

    return payload, timeout_minutes, effective_cap


def status_key(job_id: str) -> str:
    return f"triqto-control/runs/{job_id}/status.json"


def read_worker_status(job_id: str) -> dict[str, Any] | None:
    client = s3_client()
    bucket = required_env("RUNPOD_NETWORK_VOLUME_ID")
    key = status_key(job_id)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    raw = obj["Body"].read(2_000_000)
    return json.loads(raw.decode("utf-8"))


def compute_job(job: dict[str, Any]) -> None:
    payload, timeout_minutes, hourly_cap = validate_compute(job)
    pod_id: str | None = None
    deadline = time.monotonic() + timeout_minutes * 60

    try:
        pod = runpod_request("POST", "/pods", payload)
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RuntimeError(f"RunPod create response did not contain a Pod id: {pod!r}")
        pod_id = str(pod["id"])
        remember_pod(pod_id)

        raw_cost = pod.get("adjustedCostPerHr")
        if raw_cost is None:
            raw_cost = pod.get("costPerHr")
        cost = float(raw_cost) if raw_cost is not None else None
        print(
            json.dumps(
                {
                    "job_id": job["id"],
                    "pod_id": pod_id,
                    "created_at": utc_now(),
                    "cost_per_hour": cost,
                    "hourly_cap": hourly_cap,
                    "network_volume_id": required_env("RUNPOD_NETWORK_VOLUME_ID"),
                    "datacenter": required_env("RUNPOD_DATACENTER_ID"),
                },
                indent=2,
            )
        )
        if cost is None:
            raise RuntimeError("RunPod did not report Pod hourly cost; refusing an unpriced run")
        if cost > hourly_cap:
            raise RuntimeError(f"Allocated Pod costs ${cost:.4f}/h, above cap ${hourly_cap:.4f}/h")

        last_state: str | None = None
        poll_seconds = max(optional_int_env("RUNPOD_STATUS_POLL_SECONDS", DEFAULT_POLL_SECONDS), 5)
        while time.monotonic() < deadline:
            status = read_worker_status(str(job["id"]))
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

        raise TimeoutError(f"Job exceeded timeout of {timeout_minutes} minutes")
    finally:
        if pod_id:
            delete_pod(pod_id, best_effort=True)


def cleanup_active() -> None:
    for pod_id in list(active_pods()):
        delete_pod(pod_id, best_effort=True)


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

    execute_parser = subparsers.add_parser("execute", help="Validate and execute one job manifest.")
    execute_parser.add_argument("--job", type=Path, required=True)

    subparsers.add_parser("cleanup-active", help="Best-effort delete Pods recorded by this workflow.")

    args = parser.parse_args()
    if args.command == "cleanup-active":
        cleanup_active()
        return
    execute(args.job)


if __name__ == "__main__":
    main()
