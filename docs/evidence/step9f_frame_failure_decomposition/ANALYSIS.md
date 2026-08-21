# Step 9F zero-QPU frame-failure decomposition — frozen result

Scientific boundary: post-hoc exploratory audit only. No QPU submission, no TriQTO retraining, no TriQTO weight change, no threshold change, and no confirmatory interpretation.

## Frozen artifact identity

Returned artifact: `frame_failure_decomposition_v1.json`

SHA-256:

`242a720fc1173cac9db7dc41f4d1e1eeae09caa0b42ba657514bb6a34c91fdc0`

Returned terminal log SHA-256:

`5ce3ad8c529b23156b9255d6fde2586066ab3c416a6440f40f047f8f24f0dc9b`

Step-5 product: `product_b2d78ad2309b71a55f9bb54f`

Matched contexts: 116 total, 87 frozen train, 29 frozen validation.

## A. Exact probe linearity

Known-axis local response directions are essentially linear over the frozen strength range before shot noise:

- strength 0.025: median frame cosine to the exact 0.15 frame = 0.9995, p10 = 0.9926;
- strength 0.050: median = 0.9997, p10 = 0.9954;
- strength 0.100: median = 0.9999, p10 = 0.9989;
- strength 0.150: median = 1.0000, p10 = 1.0000.

Using these exact scaled frames against the frozen Step-5 finite-shot queries gives Euclidean balanced accuracy 0.9310 at strengths 0.025 and 0.050 and 0.9425 at 0.100 and 0.150. Therefore the Step-9E cheap-probe failure is not explained by geometric nonlinearity between probe strength 0.05 and target strength 0.15.

Frozen diagnostic label:

`LOW_STRENGTH_DIRECTION_IS_APPROXIMATELY_LINEAR`

## B. Finite-shot probe SNR

The known-axis probe frame recovers monotonically as strength and/or shot count increase.

At target strength 0.15:

- 512 shots: frame median cosine 0.6321; Euclidean BA 0.6815;
- 1024 shots: frame median cosine 0.7568; Euclidean BA 0.7495;
- 2048 shots: frame median cosine 0.8544; Euclidean BA 0.8362;
- 4096 shots: frame median cosine 0.9186; Euclidean BA 0.8654;
- 8192 shots: frame median cosine 0.9572; Euclidean BA 0.9037.

The frozen reference gate is BA >= 0.80, cluster-bootstrap 95% lower bound >= 0.75, and minimum mechanism recall >= 0.70.

The cheapest frozen Euclidean cell that passes this full gate is:

- probe strength = 0.15;
- shots per basis per axis = 2048;
- BA = 0.8362068966;
- 95% CI = [0.7868773946, 0.8817169540];
- minimum mechanism recall = 0.7844827586.

A lower-strength alternative also passes: strength 0.10 / 4096 shots, Euclidean BA 0.8141762452 with 95% lower bound 0.7648347701 and minimum recall 0.7772988506.

Cosine requires the strongest tested cell, 0.15 / 8192, to pass the same full gate.

Frozen diagnostic label:

`SIMULATED_PROBE_SNR_RECOVERS_AT_SOME_FROZEN_BUDGET`

Interpretation: the Step-9E Tier-3 failure at 0.05 / 512 was an acquisition-SNR failure under that frozen cheap budget, not evidence that known-axis hardware-valid probing is intrinsically incapable of recovering the local response frame.

## C. Circuit-only estimator / domain coverage

The Step-9D phase QPU circuit context is extreme relative to Step-5 Tier-1 structural support.

Standardized nearest-training-context distance:

- validation median = 2.9294;
- validation p90 = 3.9215;
- validation maximum = 5.5472;
- frozen Step-9D phase QPU context = 12.0566;
- QPU percentile versus validation = 100.0%.

Maximum RBF kernel similarity to Step-5 train:

- validation median = 0.9071;
- validation minimum = 0.7048;
- frozen Step-9D phase QPU context = 0.1916.

Changing the transfer estimator does not materially rescue Tier 1:

- RBF kernel ridge validation cosine/euclidean BA = 0.5977 / 0.5977;
- 1-NN = 0.5747 / 0.5057;
- 5-neighbor kernel average = 0.4598 / 0.4483.

On the frozen three Step-9D phase QPU cases, none of these estimators reaches 3/3 for either decoder.

Frozen diagnostic labels:

`STEP9D_QPU_CONTEXT_IS_EXTREME_IN_TIER1_FEATURE_DISTANCE`

`NEIGHBOR_TRANSFER_DOES_NOT_MATERIALLY_RESCUE_TIER1`

## Decision

Step 9F changes the diagnosis in two important ways:

1. A hardware-valid local-frame probe remains viable in simulation if acquisition strength/SNR is sufficient. The present minimum passing frozen Euclidean point is 0.15 / 2048 shots per basis per axis.
2. Circuit-only frame prediction has not received a fair in-domain test against the Step-9D deployment motif because that QPU circuit is far outside the Step-5 structural support. More estimator tuning on the same Step-5 coverage is not justified by this result.

Do not retrain TriQTO yet and do not submit a new QPU job yet.

The next zero-QPU audit should test whether adding a **frozen deployment-domain simulator bridge** around the Step-9D phase motif can rescue circuit-only local-frame prediction on held-out bridge circuits and the untouched frozen Step-9D QPU query. This separates a dataset/domain-coverage problem from a genuinely missing deployable representation. The probe route remains the hardware-valid fallback and should be taken to QPU only after the coverage-bridge audit determines whether it is actually necessary.
