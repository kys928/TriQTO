# Step 11 — exploratory IBM hardware transfer pilot

Status: **FROZEN BEFORE PHYSICAL-QPU EXECUTION**

This stage is the first hardware step after the simulator-development hard stop frozen in Step 10D. The full simulator gate remains unmet. Therefore Step 11 is explicitly exploratory transfer evidence, not confirmatory evidence for the full TriQTO hypothesis.

## Why hardware now

Step 10D was the final allowed simulator-development intervention before IBM hardware. Its predeclared late-LR refinement did not improve the primary bridge-selection mechanism metric, so the frozen primary hardware candidate remains the Step-10C warm-start ensemble. No further simulator hyperparameter tuning is allowed before this pilot.

The hardware opportunity is time-limited, while simulator experiments can be resumed later. Step 11 therefore prioritizes a tightly controlled physical-QPU transfer test without relaxing any prior simulator outcome.

## Fixed primary model

Primary hardware candidate: `step10c_warm_start` from benchmark `benchmark_f9478da45d68795655259054`.

Selected checkpoints are frozen before QPU access:

- seed 1701: `sha256:40f6c86981f038b3a44cd21148ebc67e4bac9db825bac9f4482a7fd098830769`
- seed 1702: `sha256:65b4535f89ec4d51f4126e2cb90ce9dc606d67ef2673b33740193e5aea10f39a`
- seed 1703: `sha256:d97601bcad8a6256d4433f12ca229ac0387acc927f859fe9e30ba4c2d3f3eb1f`

The ensemble effect threshold is frozen at `-0.125638447701931`, selected only from Step-10C development selection data before its fresh outer evaluation.

Architecture remains `late_concat`, 453,829 trainable parameters. Mechanism class order remains `rz_drift`, `rx_overrotation`, `ry_overrotation`.

## Paired no-extra-QPU baseline

The frozen Step-9A deployment ensemble is also evaluated on the **same acquired QPU counts**. This costs no extra QPU executions and provides a direct paired model comparison on identical hardware evidence.

The Step-9A baseline is not eligible to become the primary model after seeing QPU results. It is report-only. The Step-10C primary model identity is immutable once this protocol is merged.

## Hardware matrix

Step 11 deliberately reuses the fixed 12-case Step-9D anchor matrix:

- families: `bell_like`, `ghz`, `phase_interference`;
- per family: clean, RZ drift, RX overrotation, RY overrotation;
- injected distortion strength: 0.15;
- affected logical qubit: q0 for bell/phase, q1 for GHZ;
- clean observed circuit equals the paired reference circuit.

This is **not** claimed to be a fresh confirmatory circuit family. The earlier hardware context is known and motivated subsequent simulator development. Reusing it is intentional: Step 11 asks whether the targeted coverage repair that led to Step 10C transfers back to the physical context that exposed the original weakness.

The exact Step-9D pilot graph was excluded from the Step-10 bridge training data, but that fact does not convert this into an independent confirmatory hardware test because the earlier QPU outcome was already known during development.

## Acquisition contract

One physical backend and one connected three-qubit chain are used for the entire pilot. Backend/chain selection is quality-first from calibration metadata only, never from model predictions.

Frozen acquisition settings:

- IBM Runtime `SamplerV2`;
- one job containing 72 programs;
- 12 cases × 6 paired basis-measurement programs/case;
- 4096 shots/program;
- 294,912 total circuit shots;
- optimization level 1;
- transpiler seed 17091;
- maximum QPU execution time 300 seconds;
- no measurement mitigation;
- no dynamical decoupling;
- no twirling;
- no routing permutation allowed;
- measurement register required.

The plan must bind an explicit IBM Quantum **Open Plan** instance. Paid-plan execution is forbidden. Planning does not submit a QPU job.

## Software fail-closed boundary

The pilot uses the already-proven Step-9D hardware environment and acquisition path:

- Qiskit 2.1.2;
- Qiskit Aer 0.17.1;
- qiskit-ibm-runtime 0.40.1;
- intended environment: `/workspace/triqto/.venv-step9d`.

Planning or execution fails closed on version drift. If IBM service/API compatibility makes these exact versions unusable, only an execution-enabling software amendment with unchanged scientific design may be proposed and frozen before QPU submission.

## Plan-before-execute boundary

The planner records:

- config hash;
- primary and baseline checkpoint hashes;
- primary effect threshold;
- explicit Open Plan instance CRN/name/plan;
- backend identity/version;
- calibration update timestamp;
- selected physical chain and error metrics;
- compiled-circuit metadata hash;
- software versions;
- program count and shot count.

Physical execution requires a separately saved plan plus the exact confirmation token `STEP11_EXPLORATORY_IBM_QPU`.

If backend version, calibration timestamp, best calibrated chain, config hash, model hashes, or compiled-circuit metadata changes between planning and execution, execution fails and a new plan must be generated. Creating a new plan does not itself consume QPU time.

## Predeclared analysis

Primary targeted metric:

**Step-10C mechanism correctness on the three distorted `phase_interference` anchor cases.**

Exploratory interpretation is frozen as:

- 3/3 correct: strong targeted-repair transfer signal;
- 2/3 correct: partial targeted-repair transfer signal;
- 0/3 or 1/3 correct: weak or absent targeted-repair transfer signal.

This is descriptive exploratory language, not a confirmatory pass/fail gate.

Secondary report-only outputs include:

- mechanism accuracy across all nine distorted cases;
- effect detection count across all nine distorted cases;
- clean false-positive count across three clean cases;
- mechanism confusion matrices;
- per-family predictions;
- diagnostic RMS;
- count of cases where Step 10C is correct and Step 9A is wrong;
- count of cases where Step 9A is correct and Step 10C is wrong;
- backend/calibration snapshot;
- actual QPU usage seconds from Runtime metrics when available.

Known injected labels are never model inputs.

## Scientific hard boundaries

The following are forbidden during Step 11:

- training or fine-tuning;
- architecture changes;
- checkpoint replacement;
- threshold changes;
- QPU-driven shot-count tuning;
- backend choice based on predictions;
- adaptive measurement mitigation;
- adaptive dynamical decoupling/twirling;
- switching primary candidate after seeing QPU results;
- treating the pilot as confirmation that the unmet simulator gate passed;
- using QPU results to retroactively rewrite Step 10C or Step 10D.

Posthoc diagnosis is allowed only after the frozen pilot artifact has been recorded. Any rerun is a new exploratory attempt and cannot replace or erase the first attempt.

## Required execution sequence

1. Merge this frozen protocol and runner to `main`.
2. Verify tests and software versions in `.venv-step9d`.
3. Generate a **plan only**; no QPU submission.
4. Inspect/freeze the generated backend/instance/compiled-circuit plan.
5. Execute exactly that plan with explicit confirmation.
6. Freeze the raw counts, predictions, Runtime job ID/metrics, and exploratory interpretation before any posthoc diagnosis.

No QPU job is authorized merely by merging this protocol.
