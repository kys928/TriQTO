# Step 9A — deployment model freeze

Status: **FROZEN POST-CONFIRMATION FIXED-EPOCH REFIT, BEFORE HARDWARE TRANSFER**

Step 8 confirmed the already-selected `late_concat` architecture/training procedure on an untouched 2,000-root same-generator cohort. The exact Step-8 checkpoint weights were not serialized. A later CUDA replay of the original seeded training procedure selected epoch 19 for seed 1701 instead of the archived Step-8 epoch 8, demonstrating that exact checkpoint reconstruction is not supported.

Step 9A therefore does **not** claim to recover the exact confirmed weights. It creates one new deployment refit from development data only, with every adaptive choice inherited from Step 8 and frozen before the refit runs.

## Fixed deployment recipe

The deployment ensemble uses:

- architecture: `late_concat`;
- seeds: `1701`, `1702`, `1703`;
- fixed training epochs: `8`, `11`, `17`, inherited from the archived Step-8 selected epochs;
- trainable parameters per seed: `453,829`;
- ensemble aggregation: mean logits;
- deployed effect threshold: the archived Step-8 value `0.05939410626888275`;
- mechanism class order: `rz_drift`, `rx_overrotation`, `ry_overrotation`.

The current refit may not select a new epoch or threshold. It trains each seed for exactly its archived epoch count. The development selection partition is evaluated once after the fixed epoch only to report descriptive drift from the archived Step-8 selection metrics.

The spent Step-8 confirmatory cohort is not read.

## Why this correction is necessary

The Step-8 training run seeded Python, NumPy and PyTorch, but it did not enable deterministic CUDA algorithms. The model also uses segmented reductions such as `index_add_`, so exact CUDA optimization trajectories cannot be assumed reproducible merely from the seed.

The original Step-9A v1 runner correctly failed closed when seed 1701 replay selected epoch 19 rather than epoch 8. No deployment bundle was written by that failed attempt.

The scientifically correct response is not to loosen the replay tolerance. The exact Step-8 weights are irrecoverable because they were never serialized. The deployment artifact is therefore explicitly labeled a **post-confirmation fixed-epoch refit**.

## Weight identity and one-bundle rule

CUDA refit weights themselves need not be bitwise reproducible across a second run. Therefore:

1. the actual SHA-256 hashes of `seed1701.pt`, `seed1702.pt`, and `seed1703.pt` are part of the deployment bundle identity;
2. the first successfully completed bundle becomes authoritative for Step 9B and the later exploratory hardware pilot;
3. the runner refuses to produce a second successful deployment candidate in the same output parent;
4. every saved checkpoint is reloaded into a fresh CPU model and required to match its just-saved state tensors exactly.

This freezes actual weights rather than pretending the weights can be regenerated later from a seed alone.

## Scientific boundary

Step 9A is not a new architecture experiment:

- no new training data;
- no architecture change;
- no hyperparameter selection;
- no new epoch selection;
- no new threshold selection;
- no spent-confirmatory reuse;
- no hardware execution.

It is, however, transparently a new **post-confirmation fixed-epoch refit**. The exact Step-8 checkpoint weights are not claimed.

## Output bundle

A successful run writes:

- `seed1701.pt`;
- `seed1702.pt`;
- `seed1703.pt`;
- `training_history.csv`;
- `model_selection.json`;
- `inference_contract.json`;
- `bundle_complete.json`.

`bundle_complete.json` records the actual checkpoint hashes, source confirmation identities, weight provenance, fixed epochs, archived deployment threshold, and the explicit `exact_step8_checkpoint_weights: false` boundary.

## Run after the correction PR merges

```bash
cd /workspace/triqto
git fetch origin
git switch main
git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_step9a_deployment_freeze.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/freeze_step9a_deployment_bundle.py \
  --development-product-dir /workspace/triqto-data/step5_matched_diagnostic_training_v3/product_b2d78ad2309b71a55f9bb54f \
  --device cuda
```

A successful run ends with `TRIQTO STEP 9A DEPLOYMENT REFIT BUNDLE FROZEN`, reports that no current epoch or threshold was selected, and prints the hashed deployment bundle path.

Upload that deployment bundle ZIP for independent audit before Step 9B hardware acquisition consumes it.
