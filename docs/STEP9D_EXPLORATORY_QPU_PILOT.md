# Step 9D — exploratory IBM-QPU transfer pilot

Status: **FROZEN BEFORE PHYSICAL QPU EXECUTION**

Step 9D is the first TriQTO run on a physical QPU. It is deliberately exploratory. It does not reopen model development and it does not create confirmatory evidence.

## Frozen deployment identity

The pilot uses deployment bundle `deploy_ac536a74b2f8dd571d353a12` unchanged:

- architecture: `late_concat`;
- seeds: `1701`, `1702`, `1703`;
- effect threshold: `0.05939410626888275`;
- mechanism classes: `rz_drift`, `rx_overrotation`, `ry_overrotation`;
- no training, checkpoint selection, architecture change, or threshold change is permitted.

Step 9C dry run `dryrun_e2af1f01746a772332ae2255` is the required predecessor and passed exact training-path / hardware-path tensor equivalence.

## Pilot matrix

The pilot uses three intentionally small circuit families:

1. Bell-like, two logical qubits;
2. GHZ, three logical qubits;
3. phase-interference, two logical qubits.

Each family is executed under four conditions:

- clean paired control;
- controlled `RZ(0.15)` insertion;
- controlled `RX(0.15)` insertion;
- controlled `RY(0.15)` insertion.

This produces 12 paired acquisitions. Each acquisition uses the frozen six-program Z/X/Y reference/observed design from Step 9B, for 72 ISA circuits total.

Each program uses 4096 shots, so the complete Sampler workload contains 294,912 circuit executions. All programs use the same shot count and are submitted in one SamplerV2 job.

The known clean/distortion labels are report-only and are never model inputs.

## Backend and layout selection

The planner queries physical, operational IBM backends accessible to the configured account and searches each backend for calibrated connected three-qubit chains.

The preferred native two-qubit calibration is CZ, then ECR. Calibration entries with missing error values or error `>= 1` are rejected.

Candidates are ranked lexicographically by:

1. lowest worst edge two-qubit error;
2. lowest mean two-qubit error;
3. lowest mean readout error across the three qubits;
4. lowest pending-job count;
5. backend name;
6. physical chain.

This is quality-first, not merely least-busy selection.

The chosen three-qubit chain is used for the entire pilot. Bell and phase-interference cases use the first two physical qubits of the chain; GHZ uses all three.

All circuits are transpiled at optimization level 1 with seed `17091`. A routing permutation is forbidden: if the selected chain cannot support the small circuit without routing-induced state permutation, planning/execution fails closed.

## Two-stage physical execution gate

### 1. Plan only

The normal command performs no QPU submission. It:

- authenticates through `QiskitRuntimeService`;
- selects the backend and physical chain;
- snapshots calibration/status metadata;
- compiles all 72 programs;
- verifies the `meas` register and no-routing condition;
- writes `pilot_plan.json` and `backend_snapshot.json`.

### 2. Explicit execution

Physical submission requires all of the following:

- the saved `pilot_plan.json`;
- `--execute-physical-qpu`;
- `--confirmation-token STEP9D_EXPLORATORY_QPU`.

Immediately before submission, the runner rechecks the backend version, calibration timestamp, best physical chain, and compiled-program metadata. If any changed, it refuses to execute and requires a fresh plan.

The execution path writes `physical_access_started.json` before calling IBM Runtime. A plan cannot be physically submitted twice. A retry is a new exploratory attempt with a new plan and remains separately visible in the audit trail.

## Physical workload

The runner constructs `SamplerV2(mode=backend)` and submits all 72 ISA circuits in one `run(...)` call with 4096 shots per program. `max_execution_time` is capped at 900 QPU-seconds as a cost/safety boundary.

No measurement mitigation, dynamical decoupling, or twirling is enabled. This preserves the acquisition semantics already tested in Step 9B/9C.

## Result interpretation

Step 9D reports only descriptive pilot metrics:

- clean effect false positives;
- effect detection on nine controlled distortions;
- mechanism accuracy on the nine controlled distortions;
- the three-class confusion matrix;
- per-family predictions;
- diagnostic RMS;
- backend calibration snapshot;
- IBM Runtime job metrics/QPU usage when available.

There is **no support/not-supported gate** and no confidence claim. Twelve cases on one QPU calibration are not a hardware confirmation dataset.

A poor result means the simulator-confirmed model does not transfer cleanly under this pilot configuration and should start a new hardware-development cycle. A promising result justifies designing a larger, predeclared multi-calibration/multi-backend hardware study.

## Commands after merge

First generate a plan only:

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step9d_exploratory_qpu_pilot.py \
  --deployment-bundle-dir /workspace/triqto-data/step9a_deployment_bundle/deploy_ac536a74b2f8dd571d353a12
```

Do not execute the QPU until the generated plan has been reviewed.

After review, physical execution uses the printed plan file:

```bash
PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step9d_exploratory_qpu_pilot.py \
  --deployment-bundle-dir /workspace/triqto-data/step9a_deployment_bundle/deploy_ac536a74b2f8dd571d353a12 \
  --plan-file /workspace/triqto-data/step9d_exploratory_qpu_pilot/<PLAN_ID>/pilot_plan.json \
  --execute-physical-qpu \
  --confirmation-token STEP9D_EXPLORATORY_QPU
```
