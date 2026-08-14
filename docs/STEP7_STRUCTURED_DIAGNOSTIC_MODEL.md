# Step 7 — Structured graph-conditioned diagnostic model

Status: **FROZEN BEFORE ANY STEP-7 NEURAL OUTCOME**

Step 7 is the first serious neural diagnostic test built from the Step 5 v3 finite-shot hardware-facing cohort. It is development architecture selection, not confirmatory evidence.

## Why Step 7 exists

Step 6 established four relevant facts:

1. finite-shot `B_delta` carries real mechanism information above chance;
2. simple circuit geometry adds reproducible mechanism value beyond finite-shot diagnostics;
3. a shot-aware diagnostic magnitude path materially improves effect/null detection;
4. generic diagonal quadratic nonlinearity does not solve mechanism diagnosis.

Therefore Step 7 tests a narrower architectural hypothesis: **diagnostic evidence should be interpreted relative to the circuit geometry it belongs to**.

## Source data

Frozen source product:

`product_b2d78ad2309b71a55f9bb54f`

The Step 5 dataset, labels, split membership, reference semantics, masking, and acquisition schedule are immutable.

Step 7 must not access:

- historical v0.1 test data;
- the spent Phase 15.6 confirmatory cohort;
- statevectors or exact diagnostics as model inputs;
- distortion mechanism metadata;
- affected qubit;
- insertion depth;
- injected strength;
- circuit family labels;
- raw reference-window identifiers.

The graph input is always the intended/reference clean circuit. Hidden injected RZ/RX/RY gates never enter the graph.

## Dedicated diagnostic boundary

Signed relational `B_delta` is **not Born probability evidence**. It therefore does not enter `BornTensorBatch` or `BornEncoder`.

Step 7 introduces `DiagnosticTensorBatch` with:

- per-qubit signed Z/X/Y local expectation deltas;
- per-qubit-pair signed ZZ/XX/YY-compatible same-basis correlation deltas, canonically reordered to Z/X/Y;
- global Z/X/Y parity deltas;
- explicit pair endpoints;
- basis identity;
- observed/reference shots;
- reference availability;
- reference-kind code.

The adapter reconstructs the existing Phase-8 graph node/gate/edge features from the Step-5 clean event arrays and feeds them through the existing `CircuitGraphEncoder`. This preserves the existing variable-size graph representation instead of inventing a second graph network.

## Two distinct diagnostic problems

### Effect/null pathway

Step 6B showed that effect presence is strongly magnitude/shot dependent. The effect pathway therefore receives explicit learned embeddings of:

- local RMS;
- pairwise RMS;
- parity RMS;
- total RMS;
- `log2(observed_shots)`;
- inverse square-root shots;
- `total_rms * sqrt(observed_shots)`;
- `log2(reference_shots)`.

These are inputs/inductive bias, not a hard threshold rule.

The output is an effect logit/probability. Predictive uncertainty is the Bernoulli entropy of that learned probability; Step 7 does not fabricate an uncertainty supervision target.

### Mechanism pathway

Mechanism logits are RZ/RX/RY and are supervised **only** where `mechanism_loss_mask=true`.

Clean controls and simulator-truth negligible interventions contribute to effect/null learning but do not force a mechanism answer.

The mechanism head does not receive the explicit magnitude branch directly. It must interpret the signed diagnostic representation and circuit geometry.

## Frozen neural variants

### `diagnostic_only`

Dedicated diagnostic encoder + magnitude-aware effect head. No circuit graph information reaches the representation.

### `graph_only`

Existing circuit graph encoder only. No diagnostic or explicit magnitude information reaches the heads.

### `late_concat`

Graph and diagnostic encoders operate independently. Their graph-level embeddings meet only after pooling in a nonlinear MLP. This is the generic neural nonlinearity/capacity control.

### `structured_interaction`

The actual Step-7 TriQTO candidate:

- local diagnostic embeddings interact with the corresponding qubit-node embeddings;
- pair diagnostic embeddings interact with geometry derived from their two endpoint node embeddings;
- parity diagnostics interact with the pooled circuit-graph embedding;
- local, pair, global, diagnostic-only, and graph representations are fused after these matched interactions.

The interactions are gated residual products, not plain concatenation.

## Predeclared ablations

- `structured_no_magnitude`
- `structured_no_pairwise`
- `structured_no_parity`

Pairwise/parity ablations zero those evidence branches while keeping the same model modules/parameterization in place, so the comparison does not receive a parameter-count advantage from deleting modules.

## Development split

The Step-5 validation cohort is **not** described as an untouched test set. It was already inspected throughout Step 6.

Step 7 therefore uses:

- fit: 3,000 Step-5 training roots with family-occurrence residue 1, 2, or 3;
- internal selection: 1,000 Step-5 training roots with residue 4;
- outer development validation: the existing 1,000 Step-5 validation roots with residue 0.

All 13 derivatives of one clean circuit remain in the same root partition.

A genuinely new confirmatory cohort will be required after the Step-7 architecture is frozen if the project later wants a confirmation/generalization claim.

## Training contract

Frozen primary seeds: `1701, 1702, 1703`.

Frozen primary variants: diagnostic-only, graph-only, late-concat, structured interaction.

Frozen optimizer/schedule:

- AdamW;
- learning rate `3e-4`;
- weight decay `1e-4`;
- root batch size 32;
- maximum 20 epochs;
- early-stopping patience 4;
- gradient clipping 1.0;
- class-balanced effect BCE;
- class-balanced mechanism CE under `mechanism_loss_mask`;
- equal effect/mechanism loss weights.

Epoch selection uses the internal selection roots only, prioritizing mechanism balanced accuracy, then effect balanced accuracy, then the earlier epoch.

Outer validation does not choose epochs or hyperparameters.

## Interpretation gates

The central architecture-specific gate is deliberately stricter than "beats ridge":

> Structured interaction earns a specific architectural signal only if its mechanism balanced-accuracy improvement over `late_concat` has a paired clean-root bootstrap 95% CI whose lower bound is greater than zero.

Additional comparisons are diagnostic-only and graph-only.

For effect detection, the learned structured model should be non-inferior to Step 6B's shot-normalized SNR proxy within a frozen margin of 0.01 BA under paired root bootstrap.

If late concatenation matches structured interaction, Step 7 must report that the specialized interaction was not demonstrated to be necessary.

No Step-7 outcome is a confirmatory TriQTO claim.

## Smoke gate before full training

Before any scientific neural run, execute `scripts/v0_2/smoke_step7_structured_diagnostic_model.py`.

The smoke gate:

- accesses exactly 8 fit roots / 104 examples;
- accesses zero selection roots and zero outer-validation artifacts;
- SHA-verifies every used artifact;
- checks all 7 frozen variants;
- runs forward, masked losses, backward, finite-gradient validation, gradient clipping, and one AdamW step;
- reports parameter counts;
- makes **no model-quality claim**.

Only `STEP7_SMOKE_PASS` allows implementation of/execution of the full development training runner.

## Files

- `configs/v0_2/step7_structured_diagnostic_model.json`
- `configs/v0_2/step7_structured_diagnostic_smoke.json`
- `src/triqto/step7/contracts.py`
- `src/triqto/step7/graph_adapter.py`
- `src/triqto/step7/model.py`
- `scripts/v0_2/smoke_step7_structured_diagnostic_model.py`
- `tests/test_step7_structured_diagnostic_model.py`

This architecture is frozen before the smoke result. Smoke failures may repair implementation defects, but scientific changes to the architecture/ablation contract must be versioned and disclosed rather than silently edited after seeing model-quality results.
