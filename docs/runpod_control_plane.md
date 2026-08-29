# TriQTO RunPod Control Plane

> **Current lifecycle:** compute orchestration has moved to the detached v2 controller. See `docs/runpod_control_plane_v2.md` for the authoritative compute lifecycle, optional-limit semantics, and reconciler design.

This control plane connects GitHub Actions to RunPod compute and RunPod Network Volume storage while keeping infrastructure credentials out of GPU workers.

## Architecture

```text
ChatGPT / researcher
        |
        v
GitHub job manifest
        |
        v
GitHub Actions  -----------------------> RunPod S3-compatible API
  |                                        |
  | RunPod REST API                        | inspect/download persistent files
  v                                        |
Ephemeral GPU Pod                          |
  |                                        |
  +------ /workspace ----------------------+
         RunPod Network Volume
```

GitHub Actions is the credential boundary. The Pod receives only a validated base64-encoded job manifest. It does **not** receive the RunPod API key or S3 secret.

## Current implementation

The current controller is `scripts/runpod_control_v2.py`.

Key properties:

- one GPU per Pod
- Phase 15.6 campaign runner only
- persistent workspace must live under `/workspace/triqto-data`
- user-facing S3 operations are read-only (`list`, `head`, `read_text`, `download`)
- GPU types are restricted by a repository-level allowlist
- detached compute is the default
- unset `RUNPOD_MAX_HOURLY_USD` means no controller-imposed hourly-price cap
- unset `RUNPOD_MAX_RUNTIME_MINUTES` means no controller-imposed scientific runtime cap for detached runs
- RunPod must still report the actual Pod price before a launch is accepted
- active detached runs are recorded under `triqto-control/active/`
- the scheduled reconciler deletes Pods only after terminal worker status (`completed` or `failed`)
- manual termination is available by detached control-run ID

## Why the code image uses `/opt/TriQTO`

RunPod Network Volumes mount at `/workspace`. Application code must therefore not be baked into `/workspace`, because attaching the volume would hide that image content.

The control-plane image stores the repository at:

```text
/opt/TriQTO
```

and reserves `/workspace` for persistent RunPod data.

## Repository configuration

### Secrets

```text
RUNPOD_API_KEY
RUNPOD_S3_ACCESS_KEY_ID
RUNPOD_S3_SECRET_ACCESS_KEY
```

### Required variables

```text
RUNPOD_NETWORK_VOLUME_ID
RUNPOD_DATACENTER_ID
RUNPOD_ALLOWED_GPU_TYPES
RUNPOD_IMAGE_NAME
```

### Useful optional variables

```text
RUNPOD_S3_ENDPOINT
RUNPOD_MAX_HOURLY_USD
RUNPOD_MAX_RUNTIME_MINUTES
RUNPOD_MAX_READ_TEXT_BYTES
RUNPOD_MAX_ARTIFACT_BYTES
RUNPOD_STATUS_POLL_SECONDS
RUNPOD_TEMPLATE_ID
RUNPOD_CONTAINER_REGISTRY_AUTH_ID
```

Cost and runtime caps are optional. They should be set only when a particular operational policy requires them.

## Validated RunPod environment

Live discovery validated:

```text
Network Volume ID: aaxciry320
Name: TriQTO
Capacity: 150 GB
Datacenter: EU-CZ-1
S3 endpoint: https://s3api-eu-cz-1.runpod.io/
S3 signing region: eu-cz-1
```

## Worker image

`.github/workflows/build-runpod-agent.yml` builds:

```text
ghcr.io/kys928/triqto-runpod:latest
```

The image stores the repository at `/opt/TriQTO` and mounts the Network Volume at `/workspace`.

## Compute jobs

Compute manifests live under `runpod/jobs/` and may omit `limits` entirely when using detached mode. Example:

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

Supported Phase 15.6 commands:

```text
prepare
preflight
data
train
evaluate
aggregate
all
```

## S3 storage jobs

Storage jobs do not create a GPU Pod. Supported operations are:

```text
list
head
read_text
download
```

The public job interface cannot delete, overwrite, or move Network Volume objects.

## Storage-capacity policy

Do not enlarge the Network Volume solely to satisfy a conservative historical preflight threshold. Inventory actual usage first. If a future experiment genuinely needs additional free space, determine whether regenerable caches, obsolete datasets, superseded checkpoints, or other large artifacts can be removed safely while preserving reproducibility and research value.

## Safety boundary

The system intentionally does **not** expose a generic remote shell. New scientific operations should be added as typed, validated runners rather than arbitrary commands from job manifests.
