#!/usr/bin/env python3
"""Continuously supervise detached TriQTO RunPod jobs.

The RunPod worker never receives the RunPod API key.  Instead this process runs
inside GitHub Actions, where the existing credential boundary already lives.
It repeatedly invokes the idempotent detached reconciler and exits only when
there are no active control records or when its GitHub-runner lease is nearly
exhausted.  Exit code 75 means the workflow should dispatch a successor
supervisor; it is not a scientific timeout.
"""
from __future__ import annotations

import argparse
import json
import time

import runpod_control_v2 as control

HANDOFF_EXIT_CODE = 75


def is_pending_status_error(exc: Exception) -> bool:
    """Recognize RunPod S3's nonstandard missing-object response.

    Before a worker writes status.json, RunPod's S3-compatible endpoint may
    answer GetObject with InvalidArgument + "object not found" rather than a
    normal 404/NoSuchKey.  That is a normal startup state, not a supervisor
    transport failure.
    """
    text = str(exc).lower()
    return (
        "invalidargument" in text
        and "getobject" in text
        and "object not found" in text
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--lease-minutes", type=int, default=300)
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    args = parser.parse_args()

    if args.poll_seconds < 10:
        raise SystemExit("--poll-seconds must be at least 10")
    if not 1 <= args.lease_minutes <= 330:
        raise SystemExit("--lease-minutes must be between 1 and 330")
    if args.max_consecutive_errors < 1:
        raise SystemExit("--max-consecutive-errors must be positive")

    started = time.monotonic()
    deadline = started + args.lease_minutes * 60
    consecutive_errors = 0

    print(json.dumps({
        "supervisor": "started",
        "poll_seconds": args.poll_seconds,
        "lease_minutes": args.lease_minutes,
        "checked_at": control.utc_now(),
    }))

    while True:
        try:
            control.reconcile_detached()
            remaining = control.list_active_records()
            consecutive_errors = 0
        except Exception as exc:
            if is_pending_status_error(exc):
                remaining = control.list_active_records()
                consecutive_errors = 0
                print(json.dumps({
                    "supervisor": "status-pending",
                    "active_run_count": len(remaining),
                    "checked_at": control.utc_now(),
                    "message": "Worker status object has not been written yet; continuing normal polling.",
                }))
            else:
                consecutive_errors += 1
                print(json.dumps({
                    "supervisor": "reconcile-error",
                    "consecutive_errors": consecutive_errors,
                    "max_consecutive_errors": args.max_consecutive_errors,
                    "error": repr(exc),
                    "checked_at": control.utc_now(),
                }))
                if consecutive_errors >= args.max_consecutive_errors:
                    raise
                # Short bounded retry for transient RunPod/S3 failures.
                time.sleep(min(args.poll_seconds * consecutive_errors, 120))
                continue

        if not remaining:
            print(json.dumps({
                "supervisor": "complete",
                "active_run_count": 0,
                "checked_at": control.utc_now(),
            }))
            return

        now = time.monotonic()
        if now >= deadline:
            print(json.dumps({
                "supervisor": "handoff-required",
                "active_run_count": len(remaining),
                "checked_at": control.utc_now(),
                "message": "GitHub-runner lease ending; RunPod jobs remain active and must continue under a successor supervisor.",
            }))
            raise SystemExit(HANDOFF_EXIT_CODE)

        print(json.dumps({
            "supervisor": "watching",
            "active_run_count": len(remaining),
            "seconds_until_handoff": int(deadline - now),
            "checked_at": control.utc_now(),
        }))
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
