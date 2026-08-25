# Step 12 — independent cross-backend phase-mechanism generalization

Status: **FROZEN BEFORE PHYSICAL-QPU EXECUTION**

Step 12 follows the completed and frozen Step-11 exploratory IBM-QPU result. Step 11 observed 3/3 mechanism correctness for the Step-10C primary model on the three targeted `phase_interference` anchor distortions, versus 0/3 for the paired Step-9A report-only baseline. That result was explicitly exploratory and localized. Step 12 does not reuse those three cases. It asks the next narrower scientific question: **does the frozen Step-10C phase-mechanism capability generalize to new phase-sensitive circuits and a different physical backend without retraining or adaptation?**

This protocol is a predeclared component-generalization test. It is not a confirmatory test of the full TriQTO hypothesis.

## 1. Source result and non-reuse boundary

The source Step-11 result is frozen at merge commit `5a5d6c517864322e803481678b5f744ef5b3c2f0`, plan `qpuplan_eaef7207095be0da5e12212b`, IBM Runtime job `da6n7es6l22c73dmiaag`, and backend `ibm_kingston`.

Step 11 is used only to motivate the question. Its QPU counts are not inputs to Step 12, are not used to select the Step-12 threshold, and are not part of the Step-12 support-gate calculation.

Step 12 deliberately does **not** reuse:

- the Step-9D/Step-11 12-case anchor matrix;
- the Step-11 `phase_interference` reference circuit;
- any Step-10 bridge motif name;
- any Step-10 bridge distortion strength;
- the Step-11 backend `ibm_kingston`.

The Step-10 bridge used motifs `pilot_core_variant`, `spectator_pre`, `spectator_mid`, and `spectator_tail`, with strengths 0.08, 0.15, 0.22, and 0.30. Step 12 instead freezes three new circuit motifs and strengths 0.13 and 0.27.

## 2. Frozen models

Primary model: the unchanged Step-10C warm-start ensemble from `benchmark_f9478da45d68795655259054`.

- architecture: `late_concat`;
- trainable parameters: 453,829;
- seeds: 1701/1702/1703;
- effect threshold: `-0.125638447701931`;
- mechanism order: `rz_drift`, `rx_overrotation`, `ry_overrotation`;
- checkpoint SHA256 values are frozen in `configs/v0_2/step12_independent_phase_generalization.json`.

Paired baseline: the unchanged Step-9A deployment ensemble `deploy_ac536a74b2f8dd571d353a12`, evaluated on the **same Step-12 QPU counts**. It is report-only and cannot replace the primary candidate.

No training, fine-tuning, architecture change, checkpoint replacement, threshold adjustment, or model selection is allowed in Step 12.

## 3. Independent phase-sensitive circuit matrix

Three new motifs are frozen before QPU access:

1. `cz_echo_ramsey` — two logical qubits, affected logical qubit q0, mapped to the left-center physical edge;
2. `dual_arm_recombination` — two logical qubits, affected logical qubit q1, mapped to the right-center physical edge;
3. `three_qubit_phase_fanout` — three logical qubits, affected logical qubit q1, mapped in reverse order over the selected three-qubit chain.

The exact operation signatures are frozen in the config and enforced by the runner. The three-qubit motif uses only logical nearest-neighbor interactions q0-q1 and q1-q2 so the reverse-chain mapping remains nearest-neighbor and does not require routing permutations.

Each motif contributes:

- one clean paired-control case;
- RZ drift at strength 0.13;
- RX overrotation at strength 0.13;
- RY overrotation at strength 0.13;
- RZ drift at strength 0.27;
- RX overrotation at strength 0.27;
- RY overrotation at strength 0.27.

Totals:

- 3 clean controls;
- 18 distorted cases;
- 21 total cases;
- each mechanism appears 6 times;
- each strength appears 9 times.

Known injected labels are report-only and are never provided to the model.

## 4. Model-blind identifiability audit

Before a QPU plan can be created, the runner performs a frozen statevector-only identifiability audit of the new motifs. This audit uses **no Step-10C or Step-9A predictions**.

For each motif and distortion mechanism, it computes the change in the concatenated single-qubit Bloch-vector evidence relative to the clean reference. At the lower strength 0.13, the plan is allowed only if:

- every mechanism has diagnostic-delta norm at least `0.04`;
- every pair of mechanism-delta vectors is separated by at least `0.10`.

This is a feasibility screen, not evidence that the learned model will succeed. Its purpose is to avoid spending scarce QPU time on a circuit matrix whose frozen observable bundle is intrinsically too weak to distinguish the injected mechanisms.

The motifs and audit thresholds are frozen before QPU access. No model prediction is permitted during plan generation.

## 5. Cross-backend hardware boundary

Step 12 must execute on an IBM physical backend different from Step 11. `ibm_kingston` is excluded in the frozen config.

Within the user's explicit IBM Open Plan instance, the runner selects the best allowed connected three-qubit chain using calibration metadata only, with the same quality-first ordering used in Step 11:

1. minimum worst two-qubit error;
2. minimum mean two-qubit error;
3. minimum mean readout error;
4. minimum pending jobs;
5. backend name;
6. physical-chain identity.

Model predictions cannot influence backend or chain selection. One backend and one connected three-qubit chain are used for the entire Step-12 test.

## 6. Acquisition contract

Frozen acquisition settings:

- IBM Runtime `SamplerV2`;
- one physical-QPU job;
- 21 cases × 6 paired X/Y/Z measurement programs per case;
- 126 total programs;
- 4096 shots per program;
- 516,096 total circuit shots;
- optimization level 1;
- transpiler seed 17121;
- maximum QPU execution time 300 seconds;
- no measurement mitigation;
- no dynamical decoupling;
- no twirling;
- measurement register required;
- no routing permutation allowed.

The software environment remains frozen at Qiskit 2.1.2, Qiskit Aer 0.17.1, and qiskit-ibm-runtime 0.40.1 in `/workspace/triqto/.venv-step9d`.

## 7. Plan-before-execute and fail-closed rules

Planning does not submit a QPU job. The plan records the source Step-11 manifest hash, config hash, model checkpoint hashes, software versions, Open Plan instance identity, selected backend/chain and calibration timestamp, model-blind identifiability audit, case-design hash, and compiled-circuit metadata hash.

Physical execution requires both:

- a saved `generalization_plan.json` from a prior plan-only invocation;
- confirmation token `STEP12_INDEPENDENT_PHASE_GENERALIZATION_QPU`.

Execution fails before QPU submission if the source freeze, config, model hashes, software, identifiability audit, selected backend, selected chain, backend version, calibration timestamp, case design, or compiled circuits have changed.

Once `physical_access_started.json` exists, that plan is spent. Any rerun requires a new plan and is a new Step-12 attempt rather than a replacement.

## 8. Predeclared support gate

The primary metric is Step-10C mechanism correctness across all 18 distorted Step-12 cases.

A narrow cross-backend phase-generalization claim is supported **only if every one** of the following frozen criteria is satisfied:

- total Step-10C mechanism correctness ≥ 14/18;
- each mechanism class ≥ 4/6 correct;
- each strength level ≥ 6/9 correct;
- distorted effect detection ≥ 14/18;
- clean effect false positives ≤ 1/3;
- Step-10C mechanism-correct count exceeds Step-9A by at least 3 cases on the same QPU counts.

If all criteria pass, the frozen interpretation is:

`NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_SUPPORTED`

Otherwise:

`NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_NOT_SUPPORTED`

The gate is deliberately conjunctive so a high aggregate score cannot hide collapse on one mechanism or one strength. The Step-9A comparison protects against declaring support when the new Step-10C model does not materially outperform the old deployment baseline on the same hardware evidence.

This is a predeclared engineering/scientific support gate, not a formal independent statistical significance test.

## 9. Claim boundary

Even if the gate passes, Step 12 supports only the following narrow statement:

> The frozen Step-10C diagnostic model generalized its phase-mechanism discrimination to the frozen set of new phase-sensitive motifs, two unseen bridge strengths, changed layouts, and a physical IBM backend different from the Step-11 backend under the predeclared Step-12 acquisition conditions.

It does **not** establish:

- general superiority over all quantum circuits or hardware;
- full TriQTO hardware validation;
- correction-policy effectiveness;
- optimization advantage;
- causal proof that the Step-10 bridge alone produced the improvement;
- retroactive passage of the unmet Step-10 full simulator gate.

If the gate fails, the result must be frozen as a failure of this predeclared narrow generalization test before any diagnosis or retraining.

## 10. Required sequence

1. Merge this protocol, config, runner, and contract tests to `main`.
2. Pull the merged commit and verify the frozen software/tests.
3. Generate a Step-12 **plan only**.
4. Inspect the new backend, chain, calibration, identifiability audit, and compiled plan before physical submission.
5. Execute that exact plan with the explicit confirmation token.
6. Freeze the first completed Step-12 outcome before any posthoc diagnosis or model change.
