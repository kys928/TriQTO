# TriQTO RunPod Control Plane v2

This document supersedes the v1 compute-lifecycle details in `docs/runpod_control_plane.md`.

## Core policy

The control plane separates scientific runtime from GitHub Actions runtime.

- Compute jobs default to `lifecycle.mode = "detached"`.
- GitHub Actions creates and validates the RunPod Pod, records the active run on the Network Volume, and exits.
- The GPU Pod continues independently for as long as the TriQTO task needs.
- A lightweight GitHub Actions reconciler runs every five minutes and reads worker status through the RunPod S3-compatible API.
- When the worker reports `completed` or `failed`, the reconciler deletes the Pod and archives the control record.
- A detached run can also be explicitly terminated by control-run ID through the reconciler's manual dispatch input.

This means the GitHub-hosted runner timeout does not impose a scientific experiment timeout.

## Optional cost and runtime limits

`RUNPOD_MAX_HOURLY_USD` and `RUNPOD_MAX_RUNTIME_MINUTES` are optional.

If `RUNPOD_MAX_HOURLY_USD` is unset and a compute manifest does not provide `limits.max_hourly_usd`, the controller imposes no hourly-price cap. It still requires RunPod to report the Pod's actual hourly price before accepting the launch.

If `RUNPOD_MAX_RUNTIME_MINUTES` is unset and a detached compute manifest does not provide `limits.timeout_minutes`, the controller imposes no scientific runtime limit.

For short synchronous diagnostics, `lifecycle.mode = "attached"` is still available. Attached mode requires an explicit timeout because it intentionally waits inside one GitHub Actions job.

## Detached registry

Active runs are registered only inside the control namespace of the Network Volume:

```text
triqto-control/active/<control-run-id>.json
```

Terminal runs are archived at:

```text
triqto-control/history/<control-run-id>.json
```

Worker status and logs remain at:

```text
triqto-control/runs/<job-id>/<control-run-id>/status.json
triqto-control/runs/<job-id>/<control-run-id>/worker.log
```

The controller's internal S3 write/delete capability is hard-restricted to these control-metadata prefixes. User-facing storage jobs remain read-only (`list`, `head`, `read_text`, `download`).

## Credentials

GitHub Actions holds:

- `RUNPOD_API_KEY`
- `RUNPOD_S3_ACCESS_KEY_ID`
- `RUNPOD_S3_SECRET_ACCESS_KEY`

The GPU Pod does not receive any of those credentials.

## Current TriQTO volume

Validated live on 2026-08-29:

```text
Network Volume ID: aaxciry320
Name:              TriQTO
Datacenter:        EU-CZ-1
Capacity:          150 GB
S3 endpoint:       https://s3api-eu-cz-1.runpod.io/
S3 signing region: eu-cz-1
```

## Current EU-CZ-1 GPU discovery

The live discovery run returned the following GPU types, all with low stock at the time of the query:

```text
NVIDIA GeForce RTX 5090
NVIDIA GeForce RTX 4090
NVIDIA GeForce RTX 3090
NVIDIA RTX PRO 6000 Blackwell Workstation Edition
NVIDIA RTX PRO 6000 Blackwell Server Edition
```

The repository allowlist can be narrower than datacenter availability.

## Example detached compute manifest

```json
{
  "version": 1,
  "id": "phase15-6-train-seed-42",
  "kind": "compute",
  "task": {
    "runner": "phase15_6",
    "command": "train",
    "workspace": "/workspace/triqto-data/phase15_6",
    "seed": 42
  },
  "gpu": {
    "type_ids": [
      "NVIDIA GeForce RTX 5090",
      "NVIDIA GeForce RTX 4090",
      "NVIDIA GeForce RTX 3090"
    ],
    "count": 1,
    "cloud_type": "SECURE",
    "container_disk_gb": 50,
    "interruptible": false
  },
  "lifecycle": {
    "mode": "detached"
  }
}
```

No `limits` object is required for detached work.

## Storage policy

Do not resize the Network Volume merely because an old campaign config contains a conservative free-disk preflight threshold. First inventory actual storage usage and determine whether the requirement is scientifically necessary for the intended run. If additional free space is genuinely needed, prefer evidence-based cleanup of obsolete regenerable datasets/caches/checkpoints before paying for additional capacity, unless preserving those artifacts has research value.
