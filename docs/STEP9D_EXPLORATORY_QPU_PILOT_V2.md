# Step 9D v2 — exploratory IBM-QPU pilot

Status: **FROZEN BEFORE PHYSICAL QPU EXECUTION**

Step 9D v2 supersedes v1 before any physical QPU job was submitted. The scientific pilot remains unchanged: 3 circuit families × clean/RZ/RX/RY = 12 paired cases, six Z/X/Y reference/observed programs per case, 72 programs total, 4096 shots/program, frozen `late_concat` deployment ensemble, and exploratory-only interpretation.

## Why v2 exists

The first v1 plan was valid as a circuit/backend plan, but two execution-safety boundaries were missing:

- package-version drift was not fail-closed;
- the IBM Quantum instance was not frozen into the plan identity.

v2 fixes both before hardware access.

## Frozen software

The guarded runner requires exact versions before it authenticates or queries hardware:

- Qiskit `2.1.2`
- Qiskit Aer `0.17.1`
- Qiskit IBM Runtime `0.40.1`

Any mismatch terminates planning/execution.

## Open Plan only

The guarded runner calls `QiskitRuntimeService.instances()`, filters to instance rows whose plan is `open`, and then constructs a new `QiskitRuntimeService(instance=<exact CRN>)`.

Physical execution on paid plans is not permitted by this protocol.

If exactly one Open Plan instance exists, it is selected automatically. If multiple Open instances exist, planning requires `--instance-name <name>` so the selection is explicit.

The resulting plan identity freezes:

- instance CRN;
- instance name;
- instance plan;
- exact software versions;
- backend name/version;
- calibration timestamp;
- physical chain;
- deployment checkpoint hashes;
- compiled-program metadata hash.

Execution reconstructs the service with that same instance CRN and refuses to proceed if any frozen identity differs.

## QPU usage boundary

The pilot still requests 294,912 Sampler executions, comfortably below the 10-million-execution Sampler job limit. The v2 maximum execution time is reduced to 300 QPU seconds. This is a safety ceiling, not an expected usage estimate.

## Planning

Run only from the frozen `.venv-step9d` environment:

```bash
PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step9d_exploratory_qpu_pilot_v2.py \
  --deployment-bundle-dir /workspace/triqto-data/step9a_deployment_bundle/deploy_ac536a74b2f8dd571d353a12
```

Planning does not submit a QPU job.

## Physical execution

Physical execution is a separate command and requires both the v2 plan file and exact confirmation token:

```bash
PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step9d_exploratory_qpu_pilot_v2.py \
  --deployment-bundle-dir /workspace/triqto-data/step9a_deployment_bundle/deploy_ac536a74b2f8dd571d353a12 \
  --plan-file /path/to/v2/pilot_plan.json \
  --execute-physical-qpu \
  --confirmation-token STEP9D_EXPLORATORY_QPU
```

Do not execute a v1 plan. The v1 config is explicitly marked superseded.

## Interpretation

Step 9D remains exploratory. Results may describe clean false positives, distorted effect detections, mechanism confusion, diagnostics, calibration, and QPU usage. They may not be described as confirmatory evidence or used to retune the frozen model during the recorded pilot.
