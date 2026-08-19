# Step 9E deployable local-response-frame audit — frozen outcome

Scientific boundary: post-hoc exploratory audit only. No QPU submission, no TriQTO checkpoint retraining, no TriQTO weight change, no deployment-threshold change, and no confirmatory interpretation.

## Frozen returned artifacts

`deployable_local_frame_v1.json`

SHA-256: `0037914e10f0a8dac3bab00f57839e899a7f12a7b3f43a32a69c38211ba363d7`

`deployable_local_frame_v1.log`

SHA-256: `f9a1fe6c293940b7c5851e1e2b910d1d61fb2cc4bf443f6e2a50d642c3befa56`

The audit used the frozen Step-5 product `product_b2d78ad2309b71a55f9bb54f`, with 116 matched contexts: 87 train and 29 validation.

## Outcome

Decision status:

`NO_DEPLOYABLE_LOCAL_FRAME_DEMONSTRATED__DO_NOT_RETRAIN`

No Tier-1, Tier-2, or Tier-3 approximation passed the inherited Step-9D exploratory gate.

### Tier 1 — circuit only

Frame alignment to the exact local frame was moderate on median but unstable in the lower tail:

- median cosine across mechanisms/validation contexts: 0.6704;
- p10 cosine: -0.7731;
- mechanism median cosines: RZ=0.5891, RX=0.7125, RY=0.5667.

Decoder performance:

- cosine BA=0.5977, 95% cluster CI=[0.4943, 0.7011];
- Euclidean BA=0.5977, 95% cluster CI=[0.4943, 0.7011].

Frozen phase-QPU transfer:

- cosine: 1/3 correct;
- Euclidean: 0/3 correct;
- predicted-frame median cosine against the exact Step-9D QPU-context frame is approximately 0.

Interpretation: intended circuit structure contains some information about the local frame, but the fixed audit estimator does not generalize reliably enough across held-out Step-5 contexts and transfers especially poorly to the frozen Step-9D phase circuit. This result does not establish that circuit structure is irrelevant; it establishes that this representation/estimator/data regime is insufficient.

### Tier 2 — circuit plus baseline Born context

Adding the independent baseline Born snapshot did not help and degraded validation performance:

- median frame cosine: 0.4319;
- p10 cosine: -0.7476;
- cosine BA=0.5057;
- Euclidean BA=0.5057;
- QPU: 0/3 for both decoders;
- predicted-frame median cosine against the exact QPU-context frame: -0.3195.

Interpretation: the chosen absolute baseline Born features do not provide a useful correction to the circuit-only estimator in this audit. They may be noisy, weakly informative for a *counterfactual response frame*, or harmful in the fixed small-sample RBF representation. This result does not establish that all Born context is useless.

### Tier 3 — cheap known-axis finite-shot probes

Frozen probe contract:

- probe strength=0.05;
- target strength=0.15;
- 512 shots per basis per probe axis;
- one shared reference measurement set;
- nine additional observed probe programs per candidate context.

Outcome:

- median frame cosine=0.1887;
- p10 cosine=-0.3646;
- cosine BA=0.5287, CI=[0.4483, 0.6092];
- Euclidean BA=0.4483, CI=[0.3908, 0.5172].

The Tier-3 failure must not be overinterpreted. A strength-0.05 response measured with only 512 shots is a deliberately cheap, low-SNR probe. Scaling a noisy 0.05 response to represent a 0.15 target does not restore lost directional information. In addition, finite-strength response may be nonlinear between 0.05 and 0.15. Therefore this audit rules out the **specific frozen Tier-3 budget**, not hardware-valid local probing in general.

No frozen QPU Tier-3 evidence exists because the Step-9D QPU artifact did not include those additional probe programs and this audit correctly submitted no new job.

## What Step 9E establishes

1. The exact simulator local frame remains sufficient from Step 9D, but none of the tested hardware-valid approximations reproduces it well enough.
2. Circuit-only prediction is partially informative but strongly unstable and does not transfer to the frozen Step-9D phase circuit.
3. The particular absolute baseline Born context used by Tier 2 does not rescue the predictor.
4. The particular 0.05/512-shot Tier-3 probe contract is too weak to recover a reliable response frame.
5. The correct next action is **not** TriQTO retraining. Retraining against a representation that has not been shown to carry the required local coordinate information would confound representation failure with model-learning failure.

## Hard next action

Before architecture changes or any new QPU acquisition, run a zero-QPU decomposition audit that separates three questions:

1. **Probe linearity:** compare the exact simulator response frame at probe strengths 0.025, 0.05, 0.10, and 0.15 directly against the exact 0.15 target frame. This determines whether low-strength probing fails because the local direction itself rotates/nonlinearizes with strength.
2. **Probe SNR:** for each probe strength, simulate finite-shot frames across a fixed shot ladder such as 512, 1024, 2048, 4096, and 8192 shots and measure frame cosine plus downstream decoder BA. This determines the minimum acquisition budget required before discussing QPU feasibility.
3. **Estimator/domain decomposition:** audit Tier-1 prediction separately within Step-5 validation and on the Step-9D pilot circuit using nearest-context distance/coverage and simple context-stratified errors. Determine whether the QPU failure is mainly out-of-distribution circuit context, insufficient training coverage, or estimator capacity.

Only after these decomposition results are frozen should the project choose between: (a) improved deployable representation, (b) bounded diagnostic probing, (c) an explicit localization/response-frame head, or (d) abandoning mechanism labels in favor of a different correction target.
