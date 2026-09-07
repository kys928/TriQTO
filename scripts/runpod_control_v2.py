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
RUNPOD_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36 "
    "TriQTO-RunPod-Control/2"
)
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
        "User-Agent": RUNPOD_USER_AGENT,
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
        response = s3_client().get_object(Bucket=bucket(), Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404"}:
            return None
        raise
    raw = response["Body"].read()
    return json.loads(raw.decode("utf-8"))


def list_json_objects(prefix: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            value = read_json_object(key)
            if value is not None:
                out.append((key, value))
    return out


def active_key(control_run_id: str) -> str:
    return f"{ACTIVE_PREFIX}{control_run_id}.json"


def history_key(control_run_id: str) -> str:
    return f"{HISTORY_PREFIX}{control_run_id}.json"


def status_key(job_id: str, control_run_id: str) -> str:
    return f"triqto-control/runs/{job_id}/{control_run_id}/status.json"


def delete_pod(pod_id: str, *, best_effort: bool = False) -> None:
    try:
        runpod_request("DELETE", f"/pods/{pod_id}")
    except Exception:
        if not best_effort:
            raise


def archive_active(key: str, value: dict[str, Any], *, outcome: str, detail: dict[str, Any] | None = None) -> None:
    control_run_id = str(value["control_run_id"])
    payload = {
        **value,
        "outcome": outcome,
        "archived_at": utc_now(),
    }
    if detail:
        payload["detail"] = detail
    internal_put_json(history_key(control_run_id), payload)
    internal_delete(key)


def reconcile_once() -> int:
    active = list_json_objects(ACTIVE_PREFIX)
    for key, record in active:
        control_run_id = str(record.get("control_run_id", ""))
        pod_id = str(record.get("pod_id", ""))
        job_id = str(record.get("job_id", ""))
        if not control_run_id or not pod_id or not job_id:
            archive_active(key, record, outcome="invalid-active-record")
            continue
        status = read_json_object(status_key(job_id, control_run_id))
        if status and status.get("state") in {"completed", "failed"}:
            delete_pod(pod_id, best_effort=True)
            archive_active(key, record, outcome=str(status["state"]), detail={"status": status})
            continue

        # If the Pod disappeared before the worker wrote terminal status, archive
        # it explicitly rather than silently losing the control-plane record.
        try:
            pod = runpod_request("GET", f"/pods/{pod_id}")
        except RuntimeError as exc:
            message = str(exc)
            if "HTTP 404" in message:
                archive_active(key, record, outcome="pod-missing-before-terminal-status")
                continue
            raise
        desired = str(pod.get("desiredStatus", "")) if isinstance(pod, dict) else ""
        actual = str(pod.get("runtime", {}).get("uptimeInSeconds", "")) if isinstance(pod, dict) else ""
        if desired and desired not in {"RUNNING", "STARTING"}:
            delete_pod(pod_id, best_effort=True)
            archive_active(
                key,
                record,
                outcome="pod-not-running-before-terminal-status",
                detail={"desiredStatus": desired, "uptimeInSeconds": actual},
            )
    return len(active)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    reconcile.add_argument("--iterations", type=int)

    args = parser.parse_args()
    if args.command == "reconcile":
        iterations = args.iterations
        completed = 0
        while iterations is None or completed < iterations:
            active_count = reconcile_once()
            print(json.dumps({"time": utc_now(), "active_jobs": active_count}), flush=True)
            completed += 1
            if iterations is None or completed < iterations:
                time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
