# Step 6 — Cheap baseline benchmark

Status: **FROZEN BEFORE BASELINE OUTCOME**

Step 6 measures what is already learnable from the accepted Step 5 v3 development cohort before changing the TriQTO architecture.

Source product:

`product_b2d78ad2309b71a55f9bb54f`

The source contains 5,000 independent clean-circuit roots and 65,000 finite-shot examples. Step 6 must not regenerate, relabel, resplit, or otherwise modify this cohort.

## Questions

Step 6 separates three evaluation problems.

### 1. Effect detection

Use every example and predict:

- `no_effect` — clean control or privileged-ground-truth negligible intervention;
- `effect` — privileged-ground-truth observable intervention.

This is deliberately separate from mechanism classification. A model should not be punished for refusing to infer RZ/RX/RY when the injected simulator intervention produced no meaningful observable effect.

### 2. Mechanism diagnosis

Use only examples where `y__mechanism_loss_mask=true` and predict:

- `rz_drift`;
- `rx_overrotation`;
- `ry_overrotation`.

Clean controls and negligible interventions are excluded from mechanism loss/evaluation exactly as frozen in Step 5.

### 3. Integrated diagnosis

Combine the effect detector and corresponding mechanism classifier into four deployment-facing outputs:

- `no_effect`;
- `rz_drift`;
- `rx_overrotation`;
- `ry_overrotation`.

The effect prediction acts as an abstention gate. This makes the end-to-end baseline directly comparable to the later TriQTO diagnostic policy without forcing mechanisms onto null evidence.

## Baseline ladder

Two non-learned sanity controls are reported:

1. `majority_prior`;
2. `stratified_random`.

Deployable linear baselines use only Step 5 `x__` data:

1. `context_only` — qubit count and log2 shots;
2. `graph_stats_only` — simple statistics derived from the intended/reference clean circuit graph;
3. `diag_local` — local X/Y/Z finite-shot deltas;
4. `diag_local_pairwise` — local + same-basis pairwise deltas;
5. `diag_full` — local + pairwise + global parity;
6. `diag_full_context`;
7. `diag_full_graph`;
8. `diag_full_context_graph`.

The graph-stat baseline is intentionally not a GNN. It uses only simple counts/depth/connectivity summaries so Step 6 remains a cheap benchmark rather than premature architecture training.

Two privileged analysis ceilings are also reported:

- `exact_diag_full_oracle` — the same diagnostic representation with simulator-exact `audit__exact_delta_*` values instead of finite-shot values;
- `family_oracle` — one-hot clean-circuit generator family from the manifest.

These two variants are **never deployable evidence** and must never be compared as if they were hardware-facing models.

## Linear classifier

All learned Step 6 baselines use a deterministic class-balanced ridge least-squares classifier.

Reasons:

- no new dependency;
- no neural architecture;
- CPU-cheap at 65k examples;
- deterministic closed-form fit;
- interpretable comparison across feature sets.

Features are standardized using training-fold statistics. Class weights are inverse-frequency. The intercept is not regularized.

Frozen ridge candidates:

`1e-4, 1e-3, 1e-2, 1e-1, 1, 10`

## Validation isolation

The Step 5 validation split is evaluation-only.

Model selection occurs entirely inside the 4,000 Step 5 training roots by a deterministic four-fold root-grouped OOF procedure. The held-out fold is determined by the Step 5 `family_occurrence_index mod 5` residue among the four training residues `1,2,3,4`.

All 13 derivatives of one clean root remain together.

For effect detection, the decision threshold is selected from OOF training scores only. The policy maximizes minimum class recall, then balanced accuracy, then macro-F1, then closeness to 0.5.

After ridge/threshold selection, the final linear coefficients are fit on all eligible Step 5 training examples and the frozen Step 5 validation split is evaluated once.

No validation label is used for hyperparameter or threshold selection.

## Metrics

Primary metrics:

- balanced accuracy;
- macro-F1.

Secondary metrics:

- per-class recall;
- binary ROC-AUC for effect detection;
- macro one-vs-rest ROC-AUC for mechanism diagnosis.

Balanced accuracy and macro-F1 receive 95% confidence intervals from 1,000 clean-root-group bootstrap replicates.

The benchmark also performs paired clean-root bootstrap differences for:

- local+pairwise minus local;
- full diagnostic minus local+pairwise — global-parity contribution;
- full+context minus full;
- full+graph-stats minus full;
- full+context+graph-stats minus full;
- exact diagnostic oracle minus finite-shot full diagnostic — shot-noise headroom.

## Stratification

`diag_full` and `diag_full_context_graph` are stratified on validation by:

- circuit family;
- qubit count;
- shots;
- intervention strength;
- insertion-depth bin.

Strata with fewer than 30 eligible examples are not reported.

These strata are analysis metadata, not extra model inputs unless explicitly part of the deployable context/graph representation.

## Interpreting Step 6

The benchmark has no arbitrary architecture-promotion score. A successful run means the benchmark contract completed without touching forbidden data.

The result should answer:

1. Does finite-shot `diag_full` beat trivial and context/graph-only baselines for effect detection?
2. Does finite-shot `diag_full` carry mechanism information above chance-like controls on effectful examples?
3. Do pairwise correlations add value beyond local evidence?
4. Does global parity add value beyond local+pairwise evidence?
5. Do simple context or graph summaries add reproducible value beyond `diag_full`?
6. How much stronger is the exact diagnostic oracle, i.e. how much headroom is attributable to finite-shot noise?
7. Does effect-gated four-class diagnosis remain useful end-to-end?

Possible interpretations include:

- **finite-shot diagnostic baseline strong:** Step 7 must beat a serious cheap baseline, not merely prove learnability;
- **exact oracle strong, finite-shot baseline weak:** acquisition/shot noise is a major bottleneck;
- **both exact and finite-shot linear baselines weak:** nonlinear context-dependent interpretation may be required, but a cheap nonlinear baseline should be considered before attributing the gain specifically to a GNN;
- **simple graph stats improve finite-shot diagnostics:** circuit geometry already helps interpret B-delta and motivates the later graph+diagnostic architecture comparison;
- **graph/context-only unexpectedly strong:** inspect physics-driven selection effects and possible residual shortcuts before celebrating diagnosis performance.

Step 6 does not access the historical v0.1 test or the spent Phase 15.6 confirmatory cohort and does not execute quantum hardware.

## Run

From the Step 6 branch:

```bash
cd /workspace/triqto

git fetch origin
git switch agent/step6-cheap-baselines
git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_step6_cheap_baselines.py
```

Then run the benchmark:

```bash
PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/benchmark_step6_cheap_baselines.py \
  --product-dir /workspace/triqto-data/step5_matched_diagnostic_training_v3/product_b2d78ad2309b71a55f9bb54f
```

The runner writes a result directory under:

`/workspace/triqto-data/step6_cheap_baselines/benchmark_*`

Expected files:

- `baseline_metrics.csv`;
- `paired_differences.csv`;
- `stratified_metrics.csv`;
- `model_selection.json`;
- `feature_dimensions.json`;
- `validation_predictions.npz`;
- `decision.json`;
- `benchmark_complete.json`.

On successful execution the decision is:

`BASELINE_BENCHMARK_COMPLETE`

The scientific interpretation happens only after inspecting the full result bundle.
