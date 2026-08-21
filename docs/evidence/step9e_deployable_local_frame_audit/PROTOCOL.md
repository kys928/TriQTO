# Step 9E post-hoc deployable local-response-frame audit

Status: **FROZEN BEFORE STEP-9E OUTCOME**

Scientific boundary: post-hoc exploratory audit only. No QPU submission, no TriQTO checkpoint retraining, no TriQTO weight change, no deployment-threshold change, and no confirmatory interpretation.

## Upstream frozen result

Step 9E is motivated by the frozen Step-9D context-conditioned identifiability result:

`LOCAL_RESPONSE_FRAME_SUFFICIENT_IN_TARGETED_AUDIT__SIMULATOR_PRIVILEGED`

The exact returned Step-9D artifact is `context_identifiability_v1.json` with SHA-256:

`2e1f314eb14217c736785aede3c418a8303109299ab5b1178e9f5c1b5e4d245d`

That audit established that the existing six-program finite-shot Born diagnostic is strongly mechanism-identifiable when interpreted in the exact same-context RZ/RX/RY response frame, including 3/3 frozen phase QPU cases for both cosine and Euclidean local-oracle decoding. It did **not** establish that the exact frame is available at hardware inference time.

## Question

Can a useful approximation to the same-context local RZ/RX/RY response frame be obtained from information that can be available in hardware mode, before changing or retraining TriQTO?

The target remains the same targeted phase contract used by Step 9D:

- family: `phase_interference`;
- 2 logical qubits;
- candidate qubit 0;
- target intervention strength 0.15;
- complete RZ/RX/RY mechanism triplets;
- exact Step-5 response frame used only as simulator-privileged supervision/evaluation truth.

The candidate qubit and insertion boundary identify the **local context being scored**. This audit does not claim that a future deployment already knows the hidden distortion location. Localization is a separate problem; a deployable system may enumerate candidate contexts or obtain them from a separate localization head.

## Tier 1 — circuit-only local frame

Fit a fixed audit estimator from intended/reference circuit structure plus candidate local context to three 12D response vectors:

`graph + candidate(q,boundary) -> [v_RZ, v_RX, v_RY]`

Allowed inputs include:

- gate identities;
- qubit incidence/connectivity;
- gate order;
- sin/cos parameter encoding;
- candidate qubit;
- candidate insertion boundary/depth position.

Forbidden estimator inputs include:

- hidden mechanism label;
- finite-shot mechanism query delta;
- exact local frame;
- statevector/Hilbert metrics;
- target-derived phenomenology.

The estimator is a fixed RBF kernel-ridge regressor with no outcome-driven hyperparameter search. Exact Step-5 local response frames are simulator-privileged training targets only.

## Tier 2 — circuit + existing Born context

Use all Tier-1 inputs and add an absolute baseline Born-context snapshot:

- Z/X/Y local expectations;
- same-basis pairwise correlation;
- global parity;
- shot metadata.

The baseline is independent of the hidden mechanism query. The finite-shot query delta remains forbidden as an estimator input.

For Step 5, baseline Born features are finite-shot simulator samples of a hardware-observable measurement protocol. For the frozen Step-9D QPU check, the baseline comes from already-recorded phase clean reference counts. No new QPU access occurs.

## Tier 3 — cheap known-axis diagnostic probes

Tier 3 is conditional by default. It runs only if Tiers 1–2 fail to establish a full deployable-frame gate, unless explicitly forced.

At the candidate context, emulate a bounded hardware-valid calibration protocol:

1. measure one shared reference circuit in Z/X/Y;
2. insert a known low-strength RZ probe and measure Z/X/Y;
3. insert a known low-strength RX probe and measure Z/X/Y;
4. insert a known low-strength RY probe and measure Z/X/Y;
5. form the three finite-shot probe deltas and scale them to the target strength.

Frozen defaults:

- probe strength: 0.05;
- target strength: 0.15;
- probe shots per basis per axis: 512;
- nine additional observed probe programs per candidate context, reusing one reference measurement set.

Tier 3 is hardware-valid in protocol form, but the frozen Step-9D QPU artifact does not contain these additional probe programs. Therefore Tier 3 cannot claim QPU validation from existing counts and this script must not submit a new job.

## Primary Step-5 evaluation discipline

Tiers 1–2 fit only on the frozen Step-5 `train` contexts and are evaluated on the frozen Step-5 `validation` contexts. No validation-outcome hyperparameter search is allowed.

For each predicted/empirical approximate frame, report direct fidelity to the exact simulator frame, including mechanism-wise cosine alignment and relative L2 error.

Then substitute the approximate frame into the same finite-shot local decoder and evaluate both cosine and Euclidean decoding.

The frozen exploratory gate is inherited unchanged from Step 9D:

- balanced accuracy >= 0.80;
- cluster-bootstrap 95% lower bound >= 0.75;
- minimum mechanism recall >= 0.70.

## Frozen Step-9D QPU transfer rule

Only after the primary Step-5 validation evaluation, a Tier-1/2 estimator may be refit on all matched Step-5 contexts for the independent frozen Step-9D phase QPU check.

To demonstrate a targeted hardware-deployable frame approximation, the **same decoder** must:

1. pass the frozen Step-5 validation gate; and
2. classify all three frozen Step-9D phase QPU cases correctly using the predicted hardware-valid frame.

The QPU predicted frame is also compared directly against the exact simulator-only Step-9D local frame for audit purposes.

## Decision rules

1. If Tier 1 passes Step 5 and 3/3 QPU for the same decoder, prefer Tier 1 as the minimal deployable representation.
2. Otherwise, if Tier 2 passes both, Tier 2 is the minimal demonstrated representation.
3. If a learned tier passes Step 5 but not QPU, do not retrain; localize simulator-to-QPU/context-domain mismatch first.
4. If Tiers 1–2 fail and Tier 3 passes Step 5, do not retrain; a separately frozen and explicitly authorized bounded QPU probe audit is the next possible step.
5. If all tiers fail, do not retrain; reassess local-frame degeneracy, candidate-context representation, abstention, and diagnostic information/cost.

No Step-9E outcome may retroactively change this protocol or its thresholds.
