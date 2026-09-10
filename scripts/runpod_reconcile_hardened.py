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

import runpod_control_v2 as control

TERMINAL_WORKER_STATES = {"completed", "failed"}
TERMINAL_POD_STATES = {"EXITED", "TERMINATED"}


# Preserve the hardened reconciler's tested compatibility surface while using
# the current control-plane primitives. The aliases are deliberately created at
# module import so tests and older supervisors can monkeypatch them safely.
if not hasattr(control, "list_active_records"):
    def _list_active_records() -> list[tuple[str, dict[str, Any]]]:
        return control.list_json_objects(control.ACTIVE_PREFIX)

    control.list_active_records = _list_active_records  # type: ignore[attr-defined]

if not hasattr(control, "archive_and_remove"):
    def _archive_and_remove(
        key: str,
        record: dict[str, Any],
        terminal: dict[str, Any],
    ) -> None:
        outcome = str(terminal.get("state", "terminal"))
        control.archive_active(key, record, outcome=outcome, detail={"status": terminal})

    control.archive_and_remove = _archive_and_remove  # type: ignore[attr-defined]


def is_pending_status_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        any(token in text for token in ("nosuchkey", "notfound", "http 404", "404"))
        or (
            "invalidargument" in text
            and "getobject" in text
            and "object not found" in text
        )
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
    records = control.list_active_records()  # type: ignore[attr-defined]
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
            control.delete_pod(pod_id, best_effort=True)
            control.archive_and_remove(  # type: ignore[attr-defined]
                key,
                record,
                status or {"state": worker_state},
            )
            print(json.dumps({"pod_id": pod_id, "cleanup": "terminal-worker-cleaned"}))
            continue

        desired = pod_desired_status(pod_id)
        if desired not in TERMINAL_POD_STATES:
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
            control.archive_and_remove(key, record, terminal)  # type: ignore[attr-defined]
            print(json.dumps({
                "pod_id": pod_id,
                "cleanup": "already-terminated",
                "stale_worker_state": worker_state,
            }))
        else:
            control.delete_pod(pod_id, best_effort=True)
            control.archive_and_remove(key, record, terminal)  # type: ignore[attr-defined]
            print(json.dumps({
                "pod_id": pod_id,
                "cleanup": "exited-pod-deleted",
                "stale_worker_state": worker_state,
            }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    reconcile_detached()


if __name__ == "__main__":
    main()
