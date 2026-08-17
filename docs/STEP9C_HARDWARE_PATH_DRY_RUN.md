# Step 9C — hardware-path dry run

Status: **FROZEN BEFORE DRY-RUN EXECUTION**

Step 9C is the final local engineering gate before any physical IBM QPU pilot. It does not train, tune, select a model, or produce confirmatory evidence. It exercises the exact Step-9B hardware-facing acquisition path on a local BackendV2/Aer SamplerV2 and then runs the already-frozen Step-9A deployment ensemble.

## Frozen inputs

Step 9C is bound to:

- Step-9B merge commit `72af2fa0bc6fd4981abbeefb338ffa0e759fd3f5`;
- deployment bundle `deploy_ac536a74b2f8dd571d353a12`;
- `late_concat` seeds `1701`, `1702`, `1703`;
- the three frozen checkpoint SHA-256 identities;
- deployment effect threshold `0.05939410626888275`;
- mechanism class order `rz_drift`, `rx_overrotation`, `ry_overrotation`.

The runner verifies the checkpoint hashes before loading any model.

## Local hardware-like execution path

The frozen dry run uses `GenericBackendV2` with five physical qubits and a fixed line coupling map. `qiskit_aer.primitives.SamplerV2.from_backend(...)` supplies a local SamplerV2-compatible execution engine using a fixed backend seed and sampler seed.

For each case, the runner calls the exact Step-9B function `acquire_paired_diagnostics(...)`. Therefore the path under test is:

1. reference/observed circuit pair;
2. six Z/X/Y measurement programs;
3. explicit logical-to-physical layout;
4. backend-ISA transpilation;
5. SamplerV2 execution;
6. `meas.get_counts()` extraction;
7. Qiskit bit-order correction;
8. observed-minus-reference local/pair/parity diagnostics;
9. exact `DiagnosticTensorBatch` construction;
10. frozen-model inference.

No physical Runtime service or QPU credential is used.

## Frozen dry-run cases

The three cases are plumbing probes, not a performance benchmark:

- `bell_rz_q0`: 2-qubit Bell-like circuit with report-only RZ injection metadata;
- `ghz_rx_q1`: 3-qubit GHZ circuit with report-only RX injection metadata;
- `phase_ry_q0`: 2-qubit phase-interference circuit with report-only RY injection metadata.

The injected labels are never model inputs. Prediction correctness is explicitly **not** a Step-9C pass/fail gate.

## Strong semantic-equivalence gate

After Step 9B produces a hardware-path model batch from real sampled counts, Step 9C reconstructs the same graph and diagnostic arrays through the original Step-7 training adapter. Every graph and diagnostic tensor must match exactly in dtype, shape, index semantics, and value.

This detects hardware-path drift that ordinary shape checks can miss, including:

- qubit-order reversal;
- Z/X/Y ordering mistakes;
- pair-index changes;
- sign reversal;
- graph-feature drift;
- shot/reference metadata drift.

## Pass gates

Step 9C passes only if:

- the Step-9B identity is unchanged;
- all three deployment checkpoint hashes verify;
- each case executes all six programs;
- the `meas` register survives transpilation;
- realized counts equal requested shots for every program;
- hardware-path tensors exactly match Step-7 training-adapter tensors;
- all three frozen checkpoints load at the expected parameter count;
- every frozen-model output is finite.

Prediction correctness against the report-only injected mechanism is not a gate and cannot select or modify anything.

## Run after merge

```bash
cd /workspace/triqto
git fetch origin
git switch main
git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_step9c_hardware_path_dry_run.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step9c_hardware_path_dry_run.py \
  --deployment-bundle-dir /workspace/triqto-data/step9a_deployment_bundle/deploy_ac536a74b2f8dd571d353a12 \
  --device cpu
```

A successful run ends with `TRIQTO STEP 9C HARDWARE-PATH DRY RUN PASS` and `Step 9D exploratory QPU pilot unlocked: YES`.

Upload the resulting dry-run directory or ZIP for independent audit before Step 9D is designed or executed.
