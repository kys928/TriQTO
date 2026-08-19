# Step 9D post-hoc phase-mechanism counterfactual

Status: **POST-HOC DIAGNOSIS ONLY — NOT CONFIRMATORY**

Source physical-QPU evidence: `docs/evidence/step9d_exploratory_qpu_pilot/`

Counterfactual JSON SHA-256: `be19632f60a93dbd4e2a2cb038aed5d0b826ff908933752ef9d37a7e881fe8f4`

## Question

Why did the frozen Step-9A deployment ensemble detect all three Step-9D phase-interference distortions but identify none of their mechanisms correctly on physical hardware?

The analysis tests whether the failure is primarily attributable to:

1. physical-QPU noise/distortion of the six-program diagnostic evidence;
2. the intended-circuit graph contribution; or
3. the learned diagnostic-representation-to-mechanism mapping outside the simulator training distribution.

No QPU work, retraining, weight changes, architecture changes, or threshold changes are performed.

## Counterfactuals

For each distorted phase-interference case the frozen ensemble is evaluated under four conditions:

- **hardware + original graph**: the recorded Step-9D QPU diagnostics and intended Step-9D graph;
- **ideal + original graph**: exact statevector-derived diagnostics for the same Step-9D reference/observed circuits and the same graph;
- **hardware + zero graph**: recorded hardware diagnostics with the graph embedding zeroed before late concatenation;
- **ideal + zero graph**: exact diagnostics with the graph embedding zeroed.

The zero-graph rows are attribution probes only and are not deployable-model performance claims.

## Results

| true mechanism | hardware | ideal / same graph | hardware / zero graph | ideal / zero graph | hardware↔ideal diagnostic cosine |
|---|---|---|---|---|---:|
| RZ | RY (59.65%) | RY (54.25%) | RY (94.80%) | RY (93.14%) | 0.988917 |
| RX | RZ (87.06%) | RZ (93.93%) | RZ (95.66%) | RZ (97.81%) | 0.941246 |
| RY | RZ (79.55%) | RZ (88.35%) | RZ (97.52%) | RZ (98.15%) | 0.942910 |

All percentages are ensemble softmax probabilities for the predicted mechanism class.

## Findings

### 1. Physical hardware noise is not necessary for the phase mechanism failure

Replacing the recorded QPU diagnostics with exact ideal diagnostics does not recover any mechanism label. All three cases retain the same wrong mapping:

- true RZ -> predicted RY;
- true RX -> predicted RZ;
- true RY -> predicted RZ.

The hardware/ideal diagnostic cosine similarities are `0.988917`, `0.941246`, and `0.942910`, so the real QPU diagnostic vectors remain strongly aligned with the corresponding ideal Step-9D diagnostic directions.

Therefore the observed physical hardware noise cannot be the primary causal explanation for the 0/3 phase mechanism result.

### 2. The graph embedding is not necessary for the failure

Zeroing the graph embedding does not recover any mechanism label. Instead, the same wrong predictions generally become substantially more confident.

For example:

- true RZ: wrong RY confidence increases from 59.65% to 94.80% on hardware diagnostics;
- true RX: wrong RZ confidence increases from 87.06% to 95.66%;
- true RY: wrong RZ confidence increases from 79.55% to 97.52%.

The graph therefore appears partially corrective for these cases rather than the source of the label inversion.

This invalidates the stronger version of the earlier working hypothesis that graph conditioning itself caused the phase-family mechanism collapse.

### 3. The failure localizes downstream of diagnostic acquisition

The six-program acquisition path preserves mechanism-specific signed structure, and ideal replacement does not fix the classifier. The zero-graph probe also leaves the failure intact.

The remaining failure region is therefore the learned mapping from the deployable diagnostic evidence through:

`DiagnosticEncoder -> late_concat_fusion -> mechanism_head`

for this Step-9D phase circuit/distribution.

This analysis does **not** yet distinguish which of those learned components is principally responsible.

### 4. A simulator-to-Step-9D distribution shift remains the leading upstream explanation

The Step-5 training family named `phase_interference` is structurally different from the Step-9D hardware pilot's phase circuit.

The Step-5 family generator applies H gates across the register, parameterized RZ+RX layers, and H gates again. The Step-9D pilot uses the fixed two-qubit circuit:

`H(q0) -> RZ(0.7,q0) -> H(q0) -> CX(q0,q1)`

with the hidden RZ/RX/RY intervention inserted before the second H.

Bell and GHZ Step-9D circuits much more closely match their corresponding training generators, while phase-interference does not. This is consistent with the family-specific mechanism failure, but the present counterfactual does not by itself prove that generator mismatch is the causal source.

The more precise working hypothesis is therefore:

> the Step-9D phase diagnostic signatures lie outside, or are mapped differently from, the mechanism-labelled simulator distribution learned by the frozen diagnostic representation/head; the graph embedding partly mitigates but does not correct that mismatch.

## Effect-vs-mechanism interpretation

The result also clarifies why effect detection transferred better than mechanism classification.

The deployed Step-7 architecture gives effect prediction an explicit magnitude-feature pathway in addition to the learned representation. Mechanism classification uses the learned representation only. In the three phase cases, effect probabilities remain high under both hardware and ideal diagnostics, while the mechanism labels are systematically wrong.

This is consistent with magnitude transferring more robustly than mechanism-label geometry.

## What this does not establish

This post-hoc analysis does not establish:

- confirmatory QPU robustness;
- a corrected mechanism classifier;
- that graph information is generally beneficial or harmful;
- that the Step-5/Step-9D generator mismatch is definitively causal;
- hardware generalization to other backends, calibrations, chains, or circuit families;
- any quantum advantage claim.

## Next diagnostic test

Before any retraining or new QPU run, compare the three ideal Step-9D phase diagnostic vectors against the frozen Step-5 training distribution.

The key questions are:

1. Are the Step-9D ideal phase vectors out-of-distribution relative to Step-5 phase-interference examples?
2. Which training mechanism class is nearest in deployable diagnostic space to each Step-9D vector?
3. Does the learned representation preserve or invert those nearest-class relationships?
4. Is the mismatch specific to the phase family or visible in Bell/GHZ controls?

That analysis can be performed entirely offline with the frozen Step-5 dataset and Step-9A weights.
