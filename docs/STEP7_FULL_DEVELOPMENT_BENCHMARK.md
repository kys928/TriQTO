# Step 7 full development benchmark

Status: **FROZEN_AFTER_SMOKE_BEFORE_FULL_NEURAL_OUTCOME**

This is the full development benchmark for the first serious TriQTO diagnostic architecture. It is not confirmatory evidence. The Step-5 outer validation cohort was already inspected during Step 6, so Step 7 uses it only for frozen architecture comparison after fit/selection decisions are complete.

## Preconditions

1. Step 5 v3 final product is frozen: `product_b2d78ad2309b71a55f9bb54f`.
2. Step 6A and Step 6B are merged and remain the cheap/reference baselines.
3. Step 7 architecture config remains byte-identical to the config used by the accepted smoke gate.
4. Real-artifact smoke `smoke_185f69415ea6bb082dd93ef7` returned `STEP7_SMOKE_PASS`.
5. Historical v0.1 test, spent Phase 15.6 confirmatory data, exact diagnostics, statevectors, family labels, affected qubits, insertion depth and strength remain forbidden as model inputs.

## Frozen variants

Primary, three seeds each (`1701`, `1702`, `1703`):

- `diagnostic_only`
- `graph_only`
- `late_concat`
- `structured_interaction`

Predeclared ablations, seed `1701` only:

- `structured_no_magnitude`
- `structured_no_pairwise`
- `structured_no_parity`

All variants must retain exactly **453,829 trainable parameters**, as established by the accepted smoke gate.

## Split and selection protocol

- **fit:** 3,000 Step-5 training roots, family-occurrence residues 1/2/3;
- **internal selection:** 1,000 Step-5 training roots, residue 4;
- **outer development validation:** existing 1,000 Step-5 validation roots, residue 0.

All 13 derivatives of a root stay in one root block.

The runner SHA-verifies and converts every Step-5 artifact once, then caches deterministic 32-root CPU blocks. Fit blocks are shuffled only at the block level with a seed+epoch deterministic schedule. Root composition never changes.

For each seed/variant:

1. train on fit roots only;
2. after each epoch evaluate internal selection roots;
3. choose the effect threshold on selection roots only by the frozen minimum-recall / BA / macro-F1 / closeness-to-0.5 rule inherited from Step 6;
4. select the checkpoint by mechanism BA first, then effect BA, then earlier epoch;
5. early-stop with frozen patience/min-delta;
6. restore the selected checkpoint;
7. evaluate outer development validation exactly once for that frozen checkpoint.

For primary 3-seed aggregate metrics, mean selection logits choose the ensemble effect threshold; mean outer-validation logits produce final aggregate predictions. Outer validation never selects a threshold, epoch, architecture, or hyperparameter.

## Architecture-specific claim gate

The structured interaction earns a development architecture signal only if the paired clean-root bootstrap 95% CI lower bound for

`mechanism BA(structured_interaction) - mechanism BA(late_concat)`

is greater than zero.

This is deliberately stronger than merely beating ridge/QDA: `late_concat` is a matched nonlinear neural control with the same graph encoder, diagnostic encoder and parameter count but without node/pair graph-diagnostic alignment.

Additional paired comparisons report structured vs diagnostic-only, graph-only, the Step-6A mechanism reference, and the Step-6B SNR effect baseline. No single metric automatically unlocks Step 8.

## Run

```bash
cd /workspace/triqto

git fetch origin
git switch agent/step7-structured-diagnostic-model
git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q \
  tests/test_step7_structured_diagnostic_model.py \
  tests/test_step7_full_development_benchmark.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step7_full_development_benchmark.py \
  --product-dir /workspace/triqto-data/step5_matched_diagnostic_training_v3/product_b2d78ad2309b71a55f9bb54f \
  --smoke-dir /workspace/triqto-data/step7_structured_diagnostic_smoke/smoke_185f69415ea6bb082dd93ef7 \
  --step6a-dir /workspace/triqto-data/step6_cheap_baselines/benchmark_383d4c3070350f0bef6fdb23 \
  --step6b-dir /workspace/triqto-data/step6b_nonlinear_sanity/closure_e7de7cdd47142287352f8de8
```

The runner automatically chooses CUDA when available, otherwise CPU. `--device cpu` or `--device cuda` may be used only to select execution hardware; it does not change the frozen architecture/training contract.

## Required outputs

The result directory contains:

- `aggregate_metrics.csv`
- `seed_metrics.csv`
- `paired_differences.csv`
- `ablation_metrics.csv`
- `stratified_metrics.csv`
- `training_history.csv`
- `model_selection.json`
- `decision.json`
- `validation_predictions.npz`
- `benchmark_complete.json`

The completion marker must say `STEP7_DEVELOPMENT_BENCHMARK_COMPLETE`. This means the frozen development benchmark executed correctly; it does **not** mean the structured architecture succeeded.

Interpretation comes from the frozen paired gates in `decision.json`.

## Scientific boundary after completion

Even a positive architecture signal remains development evidence because the outer validation set has prior development exposure. Once Step 7 architecture selection is finished, a new untouched confirmatory cohort must be generated/frozen before any stronger generalization or reviewer-facing confirmation claim.
