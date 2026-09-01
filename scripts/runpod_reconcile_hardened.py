#!/usr/bin/env python3
"""Hardened detached RunPod reconciliation.

Worker status on the Network Volume remains the primary completion signal. If
that status is stale/pending, this reconciler additionally checks RunPod's Pod
lifecycle. It only cleans up a nonterminal worker record when RunPod itself says
the Pod is already EXITED or TERMINATED. A RUNNING Pod is never killed because
of age, missing output, or a stale worker status, so this adds no scientific
runtime cap.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from botocore.exceptions import ClientError

import runpod_control_v2 as control

TERMINAL_WORKER_STATES = {"completed", "failed"}
TERMINAL_POD_STATES = {"EXITED", "TERMINATED"}


def is_pending_status_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return True
    text = str(exc).lower()
    return (
        "invalidargument" in text
        and "getobject" in text
        and "object not found" in text
    )


def read_status(key: str) -> dict[str, Any] | None:
    try:
        return control.read_json_object(key)
    except Exception as exc:
        if is_pending_status_error(exc):
            return None
        raise


def pod_desired_status(pod_id: str) -> str | None:
    try:
        pod = control.runpod_request("GET", f"/pods/{pod_id}")
    except Exception as exc:
        # A transient/not-yet-visible Pod lookup must never cause us to drop the
        # active record: doing so could orphan a subsequently visible RUNNING Pod.
        text = str(exc).lower()
        if "http 404" in text or "not found" in text:
            print(json.dumps({"pod_id": pod_id, "pod_lifecycle": "not-visible", "action": "keep-active"}))
            return None
        raise
    if not isinstance(pod, dict):
        raise RuntimeError(f"RunPod GET /pods/{pod_id} returned non-object payload")
    raw = pod.get("desiredStatus")
    return str(raw).upper() if raw is not None else None


def reconcile_detached() -> None:
    records = control.list_active_records()
    print(json.dumps({"active_run_count": len(records), "checked_at": control.utc_now()}))
    for key, record in records:
        job_id = str(record.get("job_id", ""))
        control_run_id = str(record.get("control_run_id", ""))
        pod_id = str(record.get("pod_id", ""))
        if not job_id or not control_run_id or not pod_id:
            print(json.dumps({"active_key": key, "state": "invalid-record"}))
            continue

        status_key = str(record.get("status_key") or control.status_key(job_id, control_run_id))
        status = read_status(status_key)
        worker_state = str(status.get("state", "")) if status else "pending"
        print(json.dumps({
            "job_id": job_id,
            "control_run_id": control_run_id,
            "pod_id": pod_id,
            "worker_state": worker_state,
        }))

        if worker_state in TERMINAL_WORKER_STATES:
            if control.delete_pod(pod_id, best_effort=True):
                control.archive_and_remove(key, record, status or {"state": worker_state})
            continue

        desired = pod_desired_status(pod_id)
        if desired not in TERMINAL_POD_STATES:
            # Crucial safety boundary: RUNNING/unknown Pods are left untouched,
            # irrespective of elapsed time or how stale status.json is.
            print(json.dumps({
                "job_id": job_id,
                "control_run_id": control_run_id,
                "pod_desired_status": desired or "UNKNOWN",
                "action": "keep-active",
            }))
            continue

        terminal = {
            "state": "pod_terminal_with_stale_worker_status",
            "worker_state": worker_state,
            "pod_desired_status": desired,
            "observed_at": control.utc_now(),
            "worker_status": status,
        }
        if desired == "TERMINATED":
            # Nothing remains to delete; retire the stale control record.
            control.archive_and_remove(key, record, terminal)
            print(json.dumps({"pod_id": pod_id, "cleanup": "already-terminated", "stale_worker_state": worker_state}))
        elif control.delete_pod(pod_id, best_effort=True):
            # EXITED means the container has already stopped, so deletion cannot
            # interrupt a live scientific process.
            control.archive_and_remove(key, record, terminal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    reconcile_detached()


if __name__ == "__main__":
    main()
