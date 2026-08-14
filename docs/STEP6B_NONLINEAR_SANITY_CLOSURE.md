# Step 6B — nonlinear sanity closure

Step 6A established the first cheap linear baseline benchmark on the frozen Step 5 v3 development cohort. It found above-chance finite-shot mechanism diagnosis, reproducible value from simple graph statistics, and a large exact-vs-finite diagnostic gap. It also exposed an important limitation of the baseline family itself: `effect_present` is a magnitude-like target while the first benchmark used signed linear coordinates.

Step 6B is therefore a **development-only adaptive follow-up**. Step 6A validation results have already been observed, so Step 6B must never be described as pre-registered or confirmatory.

## Purpose

Before training any TriQTO graph neural architecture, answer whether cheap nonlinear structure already explains the gains we might otherwise attribute to a neural model.

The closure adds only:

1. direct finite-shot diagnostic RMS thresholding;
2. a shot-normalized diagnostic SNR proxy `RMS * sqrt(shots)`;
3. diagonal quadratic discriminant analysis (QDA) on graph statistics, full finite-shot diagnostics, and full diagnostics + context + graph statistics;
4. privileged exact-diagnostic RMS/QDA ceilings.

No dataset is regenerated. No label, split, reference semantic, or mechanism mask changes. No historical v0.1 test, spent confirmatory cohort, hardware execution, GNN, or TriQTO architecture change is permitted.

## Why diagonal QDA

Diagonal QDA is deliberately cheap and interpretable. It allows class-specific means and variances, which creates quadratic decision boundaries without introducing a neural network or a large external ML dependency. This is enough to test whether generic magnitude/variance nonlinearity explains much of the remaining signal.

The class-specific diagonal variance is estimated after fold-local standardization and shrunk toward unit variance. Shrinkage is selected only from Step 5 training-root OOF predictions.

## Effect-magnitude baselines

`diag_rms_threshold` computes the RMS only over active diagnostic value coordinates:

- local X/Y/Z deltas for present qubits;
- pairwise XX/YY/ZZ deltas for present qubit pairs;
- global X/Y/Z parity deltas.

Structural masks are excluded from the energy.

`diag_snr_proxy_threshold` multiplies that RMS by `sqrt(observed_shots)`. This is not a calibrated physical SNR estimator; it is a deliberately simple proxy motivated by finite-shot standard error scaling approximately as `1/sqrt(N)`.

`exact_diag_rms_threshold_oracle` applies the same magnitude idea to privileged exact simulator diagnostics and is analysis-only.

## Evaluation protocol

The Step 5 train/validation split remains fixed.

For diagonal QDA:

- all derivatives of a clean root remain in one OOF fold;
- the same four Step 6A training-root folds are reused;
- QDA shrinkage and binary effect thresholds are selected only from OOF training predictions;
- the final selected model is refit on all Step 5 training roots;
- validation is evaluated only after selection.

For direct magnitude scores there is no learned feature transform; the decision threshold is selected on the frozen training population only.

The Step 6A validation set has already influenced the decision to run Step 6B. Therefore all Step 6B validation results are **adaptive development evidence** even though validation is not used to fit or tune the individual Step 6B models.

## Run

```bash
cd /workspace/triqto

git fetch origin
git switch agent/step6-cheap-baselines
git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q \
  tests/test_step6_cheap_baselines.py \
  tests/test_step6b_nonlinear_sanity_closure.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/benchmark_step6b_nonlinear_sanity_closure.py \
  --product-dir /workspace/triqto-data/step5_matched_diagnostic_training_v3/product_b2d78ad2309b71a55f9bb54f \
  --step6a-dir /workspace/triqto-data/step6_cheap_baselines/benchmark_383d4c3070350f0bef6fdb23
```

Expected terminal decision on successful execution:

`NONLINEAR_SANITY_CLOSURE_COMPLETE`

The decision means the pre-architecture nonlinear controls completed successfully. It does **not** automatically unlock or validate any specific Step 7 architecture.

## Interpretation questions

After the run, inspect:

- whether direct RMS/SNR effect detection materially exceeds signed-linear `diag_full`;
- whether diagonal-QDA improves finite-shot mechanism BA beyond linear `diag_full`;
- whether graph statistics still add value after nonlinear modeling;
- whether exact-diagnostic QDA remains materially above finite-shot QDA;
- whether the remaining gap is large enough and structured enough to justify the dedicated graph + diagnostic TriQTO architecture in Step 7.
