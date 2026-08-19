# Step 9H post-hoc representation-contract ablation

Status: **FROZEN BEFORE STEP-9H OUTCOME**

Scientific boundary: post-hoc exploratory audit only. No QPU submission, no IBM credentials, no TriQTO checkpoint retraining, no TriQTO weight change, no deployment-threshold change, and no confirmatory interpretation.

## Frozen upstream evidence

Step 9G returned:

`DEPLOYMENT_DOMAIN_COVERAGE_RESCUES_CIRCUIT_ONLY_FRAME_IN_TARGETED_AUDIT`

Frozen returned artifact hashes:

- JSON: `71b16f01745d18765ea4114264299210a7d437c4caa525741def9b9899ebbd3f`
- log: `2be7b3794623261f35cb3b321292e3587eb26db826dc528edabe4db7732a92f4`

The deployment bridge strongly rescued held-out bridge/QPU frame prediction, but that result used a hand-built circuit feature map richer than the exact current Step-7 adapter contract and did not demonstrate one global frame estimator across the original Step-5 and bridge domains.

## Question

Which part of the Step-9G representation is actually necessary for the deployment-domain rescue?

This audit isolates representation **information content** while keeping the Step-9G bridge population, estimator family, ridge alpha, decoder, and thresholds fixed.

## Representation tiers

### R0 — current Step-7 information-equivalent circuit contract

Use only circuit information already represented by the current Step-7 graph adapter in deployable form:

- gate identities;
- arity and qubit incidence;
- gate order / logical layer;
- control/symmetric-gate semantics;
- per-qubit gate-incidence statistics;
- per-qubit aggregate angular `sum(sin θ)` / `sum(cos θ)` values;
- interaction-neighbor and active-layer summaries.

Do **not** use:
- per-gate angle phasors;
- candidate affected qubit;
- candidate insertion boundary;
- finite-shot query delta as a frame-predictor input;
- mechanism label;
- exact response vector as an input.

This is an information-equivalence audit, not an execution of the neural `CircuitGraphEncoder`.

### R1 — current contract + gate-level phasor preservation

Add the per-gate angular information that is already persisted in raw Step-5 graph artifacts but is not currently preserved at the gate-feature level by the Step-7 adapter:

- gate-level `sin θ`;
- gate-level `cos θ`;
- angular parameter count.

No candidate distortion location is supplied.

If R1 succeeds when R0 fails, the evidence favors a **minimal graph-adapter/input-projection change**, not a wholesale architecture replacement.

### R2 — gate-level phasors + candidate local context

Add:
- candidate qubit;
- candidate insertion boundary;
- before/after-boundary indicators.

This mirrors the local-context information used by the successful Step-9G feature map.

The candidate context is an audit upper bound. A future deployable implementation may enumerate candidate contexts or learn a separate localization/routing mechanism; the hidden true location is not assumed to be magically known at inference.

If only R2 succeeds, an explicit local-context mechanism is likely required.

## Frozen data/evaluation

Use the exact frozen Step-9G bridge generator:

- 240 deterministic bridge roots;
- 192 bridge-train roots;
- 48 held-out bridge-validation roots;
- exact frozen Step-9D pilot circuit excluded;
- finite bridge query evidence at 4096 shots/basis;
- target response strength 0.15.

Fit the same fixed RBF kernel-ridge estimator (`alpha=0.01`) separately for R0, R1 and R2 on bridge-train exact frames only.

Primary decoder: Euclidean.

Inherited gate:
- balanced accuracy >= 0.80;
- cluster-bootstrap 95% lower bound >= 0.75;
- minimum mechanism recall >= 0.70;
- frozen Step-9D phase QPU cases = 3/3 for the same primary decoder.

No hyperparameter search is allowed.

## Decision rule

1. If R0 passes the full targeted gate, the leading conclusion is:
   `CURRENT_STEP7_GRAPH_INFORMATION_IS_SUFFICIENT__COVERAGE_IS_PRIMARY_GAP`.
   This supports dataset redesign with the current graph information contract and makes warm-start checkpoint reuse the preferred first training experiment.

2. If R0 fails but R1 passes:
   `GATE_LEVEL_PHASOR_PRESERVATION_IS_REQUIRED`.
   Add gate-level phasor features to the adapter/model input. Reuse all checkpoint weights that remain shape-compatible; expand/reinitialize only the affected input projection before comparing warm-start versus scratch.

3. If only R2 passes:
   `EXPLICIT_LOCAL_CONTEXT_MECHANISM_IS_REQUIRED`.
   The project needs candidate enumeration/localization or a local-context head before retraining the main model.

4. If none pass:
   `CIRCUIT_ONLY_REPRESENTATION_NOT_SUFFICIENT_IN_TARGETED_AUDIT`.
   Do not retrain the main model; retain the Step-9F SNR-qualified known-axis probe route as the leading hardware-valid fallback.

No Step-9H outcome may retroactively change this protocol.
