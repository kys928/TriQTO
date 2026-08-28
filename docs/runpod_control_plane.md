# TriQTO RunPod Control Plane

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

The first version is intentionally narrow:

- one GPU per Pod
- Phase 15.6 campaign runner only
- persistent workspace must live under `/workspace/triqto-data`
- S3 control operations are read-only (`list`, `head`, `read_text`, `download`)
- GPU types are restricted by a repository-level allowlist
- every compute job has both a repository-level runtime cap and a job-level runtime cap
- every compute job has both a repository-level hourly-price cap and a job-level hourly-price cap
- the controller deletes the ephemeral Pod after completion, failure, price rejection, or timeout
- an Actions `always()` cleanup step retries deletion for any Pod recorded as active

## Why the code image uses `/opt/TriQTO`

RunPod Network Volumes mount at `/workspace`. Application code must therefore not be baked into `/workspace`, because attaching the volume would hide that image content.

The control-plane image stores the repository at:

```text
/opt/TriQTO
```

and reserves:

```text
/workspace
```

for persistent RunPod data.

## One-time GitHub setup

Open the TriQTO repository, then go to:

```text
Settings -> Secrets and variables -> Actions
```

### Secrets

Create these repository secrets:

```text
RUNPOD_API_KEY
RUNPOD_S3_ACCESS_KEY_ID
RUNPOD_S3_SECRET_ACCESS_KEY
```

`RUNPOD_API_KEY` is the normal RunPod API key used for Pod creation/deletion.

The S3 credentials are a **separate** RunPod credential pair. For RunPod's S3-compatible API, the access-key ID is the RunPod S3 user/access identifier shown when the S3 API key is created, and the secret is the generated S3 secret.

Never put any of these values in a job JSON file or source code.

### Variables

Create these repository variables:

```text
RUNPOD_NETWORK_VOLUME_ID
RUNPOD_DATACENTER_ID
RUNPOD_ALLOWED_GPU_TYPES
RUNPOD_MAX_HOURLY_USD
RUNPOD_MAX_RUNTIME_MINUTES
RUNPOD_IMAGE_NAME
```

Recommended initial values:

```text
RUNPOD_MAX_HOURLY_USD=4.00
RUNPOD_MAX_RUNTIME_MINUTES=360
RUNPOD_IMAGE_NAME=ghcr.io/kys928/triqto-runpod:latest
```

`RUNPOD_ALLOWED_GPU_TYPES` is a comma-separated allowlist using exact RunPod GPU type IDs/names, for example:

```text
NVIDIA H100 80GB HBM3,NVIDIA A100 80GB PCIe
```

Use the exact names returned by RunPod for the datacenter; do not guess them.

Optional variables:

```text
RUNPOD_S3_ENDPOINT
RUNPOD_TEMPLATE_ID
RUNPOD_CONTAINER_REGISTRY_AUTH_ID
RUNPOD_MAX_READ_TEXT_BYTES
RUNPOD_MAX_ARTIFACT_BYTES
RUNPOD_STATUS_POLL_SECONDS
```

If `RUNPOD_S3_ENDPOINT` is empty, the controller derives it as:

```text
https://s3api-<lowercase-datacenter-id>.runpod.io/
```

If `RUNPOD_TEMPLATE_ID` is set, it takes precedence over `RUNPOD_IMAGE_NAME`.

If the GHCR image is private, either make the package public or configure a RunPod container-registry credential and put its ID in `RUNPOD_CONTAINER_REGISTRY_AUTH_ID`.

## Network Volume requirement

The workspace must be a RunPod **Network Volume**, not merely a Pod-local volume, because the control plane terminates compute Pods after jobs finish.

The same volume is addressed in three forms:

```text
Pod:        /workspace/triqto-data/...
Serverless: /runpod-volume/triqto-data/...
S3:        s3://<NETWORK_VOLUME_ID>/triqto-data/...
```

The Network Volume's datacenter must support RunPod's S3-compatible API.

## Building the worker image

The workflow:

```text
.github/workflows/build-runpod-agent.yml
```

builds:

```text
ghcr.io/kys928/triqto-runpod:latest
```

and also tags the image with the Git commit SHA.

It runs when relevant TriQTO source/config/dependency files change on `main`, or it can be started manually with GitHub's `workflow_dispatch` UI.

## Compute jobs

A compute job is a JSON file committed under:

```text
runpod/jobs/
```

The push to `main` triggers the control-plane workflow.

Example:

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
    "type_ids": ["NVIDIA H100 80GB HBM3"],
    "count": 1,
    "cloud_type": "SECURE",
    "container_disk_gb": 50,
    "interruptible": false
  },
  "limits": {
    "max_hourly_usd": 4.0,
    "timeout_minutes": 360
  }
}
```

Supported Phase 15.6 commands in v1:

```text
prepare
preflight
data
train
evaluate
aggregate
all
```

For `prepare` and `all`, a repository-relative config can be supplied:

```json
"config": "configs/experiments/phase15_6_research_pilot.json"
```

The config path must remain under `configs/`.

## Compute lifecycle

For each compute manifest, GitHub Actions:

1. validates the manifest
2. validates the requested GPU against `RUNPOD_ALLOWED_GPU_TYPES`
3. attaches `RUNPOD_NETWORK_VOLUME_ID` at `/workspace`
4. starts an ephemeral RunPod Pod
5. receives the actual allocated hourly price from RunPod
6. immediately terminates the Pod if that price exceeds the effective cap
7. polls the worker status file through the S3-compatible API
8. waits for `completed` or `failed`
9. terminates the Pod
10. runs an additional best-effort emergency cleanup step

The effective price cap is the **lower** of:

```text
repository RUNPOD_MAX_HOURLY_USD
job limits.max_hourly_usd
```

The effective runtime is likewise constrained by the repository maximum.

## Persistent run metadata

The worker writes control metadata to:

```text
/workspace/triqto-control/runs/<job-id>/
```

including:

```text
job.json
status.json
worker.log
```

The actual Phase 15.6 dataset/checkpoints/results remain under the workspace supplied to the campaign runner, for example:

```text
/workspace/triqto-data/phase15_6
```

## S3 storage jobs

Storage jobs also live under `runpod/jobs/`, but they do **not** create a GPU Pod.

### List

```json
{
  "version": 1,
  "id": "inspect-phase15-6",
  "kind": "storage",
  "storage": {
    "operation": "list",
    "path": "triqto-data/phase15_6",
    "max_items": 200
  }
}
```

### Read a small text/JSON file

```json
{
  "version": 1,
  "id": "read-campaign-plan",
  "kind": "storage",
  "storage": {
    "operation": "read_text",
    "path": "triqto-data/phase15_6/campaign_plan.json",
    "max_bytes": 262144
  }
}
```

### Inspect metadata without downloading

Use:

```json
"operation": "head"
```

This reports object size, ETag, modification time, and content type.

### Download an object as a GitHub Actions artifact

Use:

```json
{
  "version": 1,
  "id": "download-report",
  "kind": "storage",
  "storage": {
    "operation": "download",
    "path": "triqto-data/phase15_6/report.json",
    "max_bytes": 10485760
  }
}
```

The workflow uploads downloaded files as a short-lived GitHub Actions artifact.

## What v1 intentionally cannot do

The first version does not support:

- arbitrary shell commands from job JSON
- arbitrary Python entrypoints
- more than one GPU
- S3 delete
- S3 overwrite
- S3 move
- account/billing administration
- passing the RunPod API key into the GPU Pod

Those restrictions are deliberate. Additional experiment runners should be added as explicit typed/validated runners rather than by opening a generic remote shell primitive.

## First live test sequence

Use this order after the repository settings are configured:

1. verify that the Network Volume is in an S3-supported RunPod datacenter
2. build/publish the `triqto-runpod` image
3. run a storage `list` job against the volume; this costs no GPU time
4. run a storage `read_text` job on an existing small campaign metadata file
5. run a very short Phase 15.6 `preflight` compute job with a strict price/runtime cap
6. verify that `status.json` and `worker.log` appear through S3
7. verify that the Pod is automatically terminated
8. only then allow training/evaluation jobs

Do not use a valuable training run as the first end-to-end test.
