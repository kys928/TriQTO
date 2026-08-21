# Step 9F zero-QPU frame-failure decomposition

Status: **FROZEN BEFORE STEP-9F OUTCOME**

Scientific boundary: post-hoc exploratory audit only. No QPU submission, no TriQTO checkpoint retraining, no TriQTO weight change, no deployment-threshold change, and no confirmatory interpretation.

## Motivation

Step 9D established that the exact same-context RZ/RX/RY response frame is sufficient in the targeted phase audit. Step 9E then showed that three hardware-valid approximations did not reproduce that frame well enough:

- Tier 1 circuit-only estimator: partial Step-5 signal but poor held-out/QPU transfer;
- Tier 2 circuit + absolute Born context: no improvement;
- Tier 3 known-axis probes at strength 0.05 and 512 shots: low frame alignment and failed decoding.

Step 9F does **not** try a larger TriQTO model. It decomposes the failure into three narrower causes before any redesign.

## A. Probe linearity / finite-strength deformation

For each frozen Step-5 validation context, construct exact simulator response frames at probe strengths:

`[0.025, 0.05, 0.10, 0.15]`

The exact 0.15 frame remains the target. For each lower-strength frame:

1. scale the vector magnitude by `0.15 / probe_strength`;
2. compare each mechanism direction to the exact 0.15 mechanism vector;
3. substitute the scaled exact probe frame into the frozen finite-shot Step-5 decoder.

This isolates finite-strength nonlinearity from measurement noise.

Interpretation:

- strong exact-frame cosine but poor finite-shot probe performance later -> SNR/acquisition problem;
- poor exact-frame cosine already at low strength -> response direction rotates with strength, so naive low-strength probing is structurally invalid.

## B. Probe SNR ladder

For each probe strength above, simulate hardware-compatible Z/X/Y known-axis probe measurements at shots per basis:

`[512, 1024, 2048, 4096, 8192]`

Use 24 frozen Monte-Carlo repeats per validation context/cell. Each repeat uses one shared reference sample plus independent RZ/RX/RY probe samples, matching the Step-9E Tier-3 protocol structure.

For every `(probe_strength, shots)` cell report:

- pooled frame cosine distribution against exact 0.15 target frame;
- balanced accuracy for cosine and Euclidean local decoding;
- mechanism recalls;
- cluster-bootstrap 95% interval over original validation contexts, keeping all shot-noise repeats within the sampled context cluster.

This is a diagnostic surface, not a new deployment gate. The inherited Step-9D thresholds are reported only as reference markers.

## C. Circuit-estimator coverage vs estimator failure

Using Tier-1 deployable circuit/context features only:

1. fit the same frozen RBF kernel-ridge estimator on Step-5 train contexts;
2. compute each held-out validation context's nearest standardized train-feature distance and maximum RBF similarity;
3. compute the frozen Step-9D phase pilot context's corresponding distance/similarity and its percentile relative to validation;
4. compare the RBF estimator with fixed 1-nearest-neighbor and 5-nearest-neighbor exact-frame transfer baselines using the same deployable feature metric;
5. evaluate all three estimators on Step-5 validation and the frozen Step-9D phase QPU counts.

The nearest-neighbor targets remain simulator-privileged audit targets. They are not proposed as deployment machinery; they diagnose whether the failure comes from the RBF estimator or from poor neighborhood coverage/smoothness in the chosen feature space.

Interpretation:

- NN succeeds while RBF fails -> estimator choice/capacity is implicated;
- RBF and NN both fail with normal validation distances -> chosen circuit feature metric is not smoothly aligned with response-frame geometry;
- QPU context is an extreme-distance outlier -> Step-5 context coverage/domain shift is implicated;
- QPU context has normal distance but all estimators fail -> feature semantics/response mapping, not simple OOD distance, is implicated.

## Frozen data discipline

- Step-5 train contexts remain estimator-fit data.
- Step-5 validation contexts remain held-out evaluation data.
- Frozen Step-9D QPU counts are read-only external transfer evidence.
- Exact simulator frames are audit targets only.
- No historical test or confirmatory cohort is touched.
- No new IBM credentials are required or requested.

## Hard decision boundary

Step 9F cannot authorize TriQTO retraining by itself. Its role is to choose the **next representation experiment**:

- probe-budget redesign if exact low-strength frames align but finite-shot SNR is inadequate;
- target-strength probing if low-strength nonlinearity is the main failure;
- circuit-feature redesign if feature-space smoothness/coverage fails;
- estimator redesign only if feature neighborhoods are informative but RBF regression is specifically deficient.
