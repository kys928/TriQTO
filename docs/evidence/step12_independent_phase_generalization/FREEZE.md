# Step 12 pre-QPU freeze

Status: **FROZEN BEFORE PHYSICAL-QPU EXECUTION**

This file freezes the implementation and scientific identity of the first Step-12 independent cross-backend phase-mechanism generalization attempt.

## Frozen source state

- Step-11 result-freeze merge commit: `5a5d6c517864322e803481678b5f744ef5b3c2f0`
- Step-11 plan: `qpuplan_eaef7207095be0da5e12212b`
- Step-11 IBM Runtime job: `da6n7es6l22c73dmiaag`
- Step-11 backend: `ibm_kingston`
- Step-11 targeted Step-10C result: 3/3, frozen as a strong exploratory targeted-repair transfer signal
- Step-11 remains exploratory and is not rewritten by Step 12.

## Frozen model identity

Primary: unchanged Step-10C warm-start ensemble from `benchmark_f9478da45d68795655259054`.

- architecture: `late_concat`
- trainable parameters: 453,829
- seeds: 1701/1702/1703
- effect threshold: `-0.125638447701931`
- exact checkpoint SHA256 values are frozen in the Step-12 config

Paired report-only baseline: unchanged Step-9A deployment ensemble `deploy_ac536a74b2f8dd571d353a12` on the same QPU counts.

No training, architecture change, checkpoint replacement, threshold change, or QPU-result-driven model selection is permitted.

## Frozen independent test matrix

Step 12 uses three new phase-sensitive motifs:

- `cz_echo_ramsey`
- `dual_arm_recombination`
- `three_qubit_phase_fanout`

The Step-9D/Step-11 anchor matrix is not reused. Step-10 bridge motif names are not reused. Distortion strengths are frozen at 0.13 and 0.27, which are distinct from the Step-10 bridge strengths 0.08, 0.15, 0.22, and 0.30.

The matrix contains 3 clean controls and 18 distorted cases: three mechanisms × two strengths × three motifs.

A statevector-only model-blind identifiability audit must pass before planning. It may not use Step-10C or Step-9A predictions.

## Frozen cross-backend rule

`ibm_kingston`, the Step-11 backend, is excluded. Step 12 must use a different operational physical backend selected from the explicit Open Plan instance using calibration quality only.

One backend and one connected three-qubit chain are used for the full attempt. Prediction-driven backend selection is forbidden.

## Frozen acquisition

- 21 cases
- six paired X/Y/Z measurement programs per case
- 126 total programs
- 4096 shots per program
- 516,096 total circuit shots
- SamplerV2 single-job execution
- optimization level 1
- transpiler seed 17121
- maximum QPU execution time 300 s
- no mitigation
- no dynamical decoupling
- no twirling
- no routing permutation
- explicit Open Plan only

## Frozen support gate

A narrow cross-backend phase-generalization statement is supported only if **all** criteria pass:

- Step-10C total mechanism correctness ≥ 14/18
- each mechanism ≥ 4/6
- each strength ≥ 6/9
- distorted effect detection ≥ 14/18
- clean false positives ≤ 1/3
- Step-10C minus Step-9A mechanism-correct advantage ≥ 3

Pass interpretation: `NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_SUPPORTED`.

Fail interpretation: `NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_NOT_SUPPORTED`.

No post-QPU threshold or interpretation adjustment is allowed.

## Frozen implementation blob identities

The following Git blob SHA-1 values were computed from the exact pre-merge file bytes:

- config `configs/v0_2/step12_independent_phase_generalization.json`: `62c53d5dd127d13f2d1358610c9015912f6e2cb4`
- runner `scripts/v0_2/run_step12_independent_phase_generalization.py`: `36522aada1c30f2d5fbdf02320c8af5962208c03`
- contract tests `tests/test_step12_independent_phase_generalization_contract.py`: `fa2d455425d4c8ebcd85dad5a5416b65a7601f57`
- protocol `docs/evidence/step12_independent_phase_generalization/PROTOCOL.md`: `4bbd58ef8cdf9a523789ed9fd2c2c73b039e5351`

Any change to these files after this freeze requires a documented pre-execution amendment and a fresh plan. Changes that alter the model, motifs, strengths, backend-exclusion rule, shot count, acquisition contract, or support gate are not compatibility amendments; they define a different experiment.

## Execution boundary

Merging this freeze does not authorize or submit a QPU job. Planning must happen first and must produce `generalization_plan.json`, `backend_snapshot.json`, and `identifiability_audit.json` for inspection.

Physical execution requires the exact confirmation token `STEP12_INDEPENDENT_PHASE_GENERALIZATION_QPU` and a saved plan. The runner fails closed on source, software, model, calibration, chain, circuit, or audit drift.

The first plan that reaches `physical_access_started.json` is spent. A later attempt cannot replace or erase it.

## Claim boundary

Step 12 can at most support a narrow generalization claim for this frozen diagnostic component. It cannot establish full TriQTO hardware validation, correction-policy success, optimization advantage, or retroactive passage of the unmet simulator gate.
