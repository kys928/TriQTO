# Step 9B — hardware diagnostic acquisition

Status: **FROZEN BEFORE HARDWARE DRY RUN**

Step 9B implements the hardware-facing acquisition path required by the frozen Step-9A deployment ensemble. It does not retrain the model and does not execute a QPU during this stage.

## Bound deployment bundle

The adapter is bound to deployment bundle `deploy_ac536a74b2f8dd571d353a12`:

- architecture: `late_concat`;
- seeds: `1701`, `1702`, `1703`;
- deployment effect threshold: `0.05939410626888275`;
- mechanism classes: `rz_drift`, `rx_overrotation`, `ry_overrotation`;
- weight provenance: post-confirmation fixed-epoch development-only refit.

## Six-program paired acquisition

For one intended/reference circuit and one observed circuit, Step 9B constructs exactly six programs in this order:

1. reference Z;
2. reference X (`H` then Z measurement);
3. reference Y (`Sdg`, `H`, then Z measurement);
4. observed Z;
5. observed X;
6. observed Y.

Every program writes logical qubit `q[i]` to classical bit `meas[i]`.

## Qiskit bit order

Sampler count keys are interpreted as `c[n-1] ... c[0]`. The adapter reverses each count key before computing logical-qubit eigenvalues so the first model row always corresponds to logical `q0`.

This is tested explicitly with the count key `01`, which must be interpreted as logical `(q0=1, q1=0)`.

## Diagnostic semantics

For each basis, the adapter derives:

- each logical-qubit expectation;
- every same-basis logical-qubit pair correlation;
- global parity.

The frozen sign convention is then applied:

`diagnostic = observed - paired_reference`

The result is converted directly to the existing `DiagnosticTensorBatch` contract with basis codes `[0, 1, 2]` for Z/X/Y and reference-kind code `0`.

## Graph semantics

The model graph is built only from the intended/reference circuit, never from the observed circuit containing the controlled or hidden distortion. Step 9B reuses the exact Step-7 graph feature builder rather than introducing a second graph representation.

Measurement operations and basis rotations are not added to the model graph.

## Backend execution

`compile_paired_measurement_circuits(...)` requires an explicit logical-to-physical initial layout and transpiles all six programs to the selected backend ISA with the same layout and transpiler seed.

`make_ibm_runtime_sampler(...)` constructs `qiskit_ibm_runtime.SamplerV2(mode=backend)` using the repository-pinned Runtime package. `acquire_paired_diagnostics(...)` accepts a SamplerV2-compatible object and converts its six PUB results into the frozen model input contract.

Physical-hardware submission remains subject to the repository's existing credential and explicit-confirmation boundary. Step 9B tests do not submit hardware jobs.

## Next gate — Step 9C

Before any QPU job, Step 9C must run this same compile / Sampler / count-extraction / model-batch path against a local simulator or fake BackendV2. The dry run must verify:

- all six circuits transpile;
- one fixed layout is retained as the acquisition layout contract;
- the `meas` register survives transpilation;
- SamplerV2 result extraction works;
- bit order and diagnostic signs remain correct through execution;
- the frozen Step-9A checkpoints can load and perform inference on the resulting batch.

Only after Step 9C passes should the exploratory Step-9D QPU pilot be allowed.
