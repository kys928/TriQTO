# Step 9G deployment-domain simulator coverage bridge

Status: **FROZEN BEFORE STEP-9G OUTCOME**

Scientific boundary: post-hoc exploratory audit only. No QPU submission, no IBM credentials, no TriQTO checkpoint retraining, no TriQTO weight change, no deployment-threshold change, and no confirmatory interpretation.

## Why this audit exists

Frozen Step 9F established two different facts:

1. A hardware-valid known-axis local-response probe is geometrically viable and passes the inherited Euclidean reference gate in simulation once acquisition SNR is sufficient. The first passing frozen cell is probe strength 0.15 with 2048 shots per basis per axis.
2. The frozen Step-9D phase QPU circuit is extreme out-of-domain relative to the Step-5 circuit-only feature support: standardized nearest-training distance 12.0566 versus a Step-5 validation maximum of 5.5472, and maximum RBF similarity 0.1916 versus a Step-5 validation minimum of 0.7048. RBF, 1-NN, and 5-neighbor transfer all fail.

Therefore Step 9G asks whether Tier-1 failed because the deployment circuit motif was missing from the simulator response library rather than because circuit-conditioned response-frame prediction is intrinsically impossible.

## Frozen bridge design

Generate 240 deterministic two-qubit simulator contexts around the Step-9D phase deployment motif. The bridge is generated entirely from simulator circuit definitions and does not use QPU outcomes or QPU mechanism predictions as training targets.

Four motifs are cycled evenly:

1. `pilot_core_variant`: `H(0) -> RZ(phi,0) -> H(0) -> CX(0,1)`;
2. `spectator_pre`: add a parameterized spectator `RY` on qubit 1 before the core;
3. `spectator_mid`: add a parameterized spectator `RZ` on qubit 1 between the candidate boundary and the closing `H`;
4. `spectator_tail`: add a parameterized spectator `RY` on qubit 1 before the final entangler.

The candidate local context remains qubit 0 immediately after the core `RZ(phi,0)`.

Angles are deterministic pseudo-random draws from frozen seeds. The exact frozen Step-9D pilot angle `phi=0.7` is excluded; the exact pilot circuit must not occur in bridge train or validation.

Split rule: bridge root index modulo 5 equals 0 -> validation; otherwise -> train. This yields 192 bridge-train and 48 bridge-validation contexts, balanced across the four motifs.

## Bridge targets and queries

For every bridge context:

- simulator-privileged exact RZ/RX/RY response vectors at strength 0.15 are the frame targets;
- an independent finite-shot hidden-mechanism query bundle is generated at strength 0.15 with 4096 shots per Z/X/Y basis;
- exact frames are never estimator inputs;
- hidden mechanism labels are never estimator inputs;
- finite-shot query deltas are never estimator inputs.

Candidate qubit/boundary identify the local context being scored; localization remains a separate problem.

## Estimators

Use the **same frozen circuit-only feature map and same fixed RBF kernel-ridge estimator** from Steps 9E/9F. No hyperparameter tuning is allowed.

Frozen ridge alpha: 0.01.

Compare three training supports:

1. `step5_only`: the original 87 frozen Step-5 train contexts;
2. `bridge_only`: the 192 bridge-train contexts;
3. `step5_plus_bridge`: concatenation of both train supports.

Primary evaluation is on the 48 held-out bridge-validation contexts. The untouched frozen Step-9D phase QPU query remains an external targeted transfer check.

## Frozen decoder/gate

Step 9F established Euclidean decoding as the SNR-efficient primary local-frame decoder, so Step 9G predeclares **Euclidean as primary** and cosine as secondary.

The inherited exploratory Euclidean gate remains unchanged:

- balanced accuracy >= 0.80;
- cluster-bootstrap 95% lower bound >= 0.75;
- minimum mechanism recall >= 0.70.

No threshold may be changed after observing Step-9G output.

## Coverage-rescue criterion

A training support demonstrates a **targeted deployment-domain coverage rescue** only if all are true:

1. its predicted frames pass the inherited Euclidean gate on held-out bridge validation;
2. the same Euclidean decoder classifies all three already-frozen Step-9D phase QPU cases correctly using the predicted Step-9D frame;
3. the frozen Step-9D candidate context lies inside the observed held-out bridge support envelope in the fitted feature metric: nearest-training distance <= the maximum bridge-validation distance and maximum-kernel similarity >= the minimum bridge-validation similarity.

Frame cosine to the exact simulator-only frame is reported as an audit diagnostic but is not an additional tunable gate.

## Decision rules

- If `step5_plus_bridge` satisfies the full coverage-rescue criterion, treat missing deployment-domain simulator coverage as the leading explanation of Tier-1 transfer failure. The next dataset/model step may then augment the simulator training distribution and evaluate warm-start reuse of compatible TriQTO checkpoint weights; it does **not** automatically require a new architecture.
- If `bridge_only` passes but `step5_plus_bridge` fails, the deployment-domain response is learnable from circuit context, but one fixed low-capacity response estimator cannot reconcile the old and bridge domains. Investigate mixture/domain-conditioned representation before TriQTO retraining.
- If neither bridge-supported estimator passes, circuit-only response-frame prediction remains insufficient under the tested representation. The Step-9F SNR-qualified known-axis probe becomes the evidence-backed hardware-valid fallback, subject to a separately frozen bounded-QPU protocol.
- No Step-9G outcome authorizes a new QPU job or TriQTO retraining by itself.
