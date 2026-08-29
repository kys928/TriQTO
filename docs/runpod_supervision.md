# RunPod detached supervision

TriQTO detached RunPod jobs are intentionally not limited by a scientific runtime cap. Cleanup is therefore driven by worker terminal state rather than elapsed experiment time.

## Primary cleanup path

Every successful detached launch dispatches `TriQTO RunPod Supervisor`. The supervisor runs in GitHub Actions, where the RunPod API key and S3 credentials remain available without exposing them to the GPU worker.

The supervisor polls the S3-backed active-run registry every 30 seconds. For each active run it invokes the same idempotent reconciler used by the manual cleanup workflow. A worker state of `completed` or `failed` causes the reconciler to delete the RunPod Pod and then archive/remove the active control record.

The supervisor uses a 300-minute GitHub-runner lease. This is not a RunPod or scientific runtime limit. If jobs are still active near the end of that lease, the supervisor exits with a handoff signal and the workflow dispatches a successor supervisor on a fresh GitHub-hosted runner. The RunPod Pod is not interrupted during this handoff.

Transient S3 or RunPod reconciliation errors are retried with bounded backoff. Repeated errors fail the supervisor visibly rather than silently claiming cleanup succeeded.

## Independent backstop

`TriQTO RunPod Reconciler` remains scheduled and manually dispatchable. It is intentionally independent of the continuous supervisor. GitHub scheduled workflows can be delayed, so the cron workflow is a backstop rather than the primary low-latency cleanup path.

Manual termination remains available through the reconciler workflow's `terminate_control_run_id` input.

## Credential boundary

The RunPod worker receives only the validated TriQTO job payload. It does not receive `RUNPOD_API_KEY`, S3 credentials, or a GitHub token. Pod deletion and supervisor handoff are performed from GitHub Actions.

## Failure model

- Launch fails before a Pod ID exists: no active record is registered.
- Launch creates a Pod but registration fails: launch cleanup deletes the Pod best-effort.
- Worker completes or fails: supervisor detects terminal status and deletes the Pod.
- Supervisor reaches its GitHub-runner lease: supervision is handed to a successor without stopping the Pod.
- GitHub cron is delayed: primary supervisor continues independently.
- Supervisor encounters repeated infrastructure errors: workflow fails visibly; scheduled/manual reconciler remains available as a backstop.
