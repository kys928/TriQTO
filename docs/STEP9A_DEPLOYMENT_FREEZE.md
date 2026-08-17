# Step 9A — deployment model freeze

Status: **FROZEN AFTER STEP 8 CONFIRMATION, BEFORE HARDWARE TRANSFER**

Step 8 confirmed the already-selected `late_concat` ensemble on an untouched 2,000-root same-generator cohort. Step 9A does not perform new model development. It converts that confirmed ensemble into a reusable deployment artifact.

## Frozen model identity

The deployment ensemble is exactly:

- architecture: `late_concat`;
- seeds: `1701`, `1702`, `1703`;
- selected epochs: `8`, `11`, `17`;
- trainable parameters per seed: `453,829`;
- ensemble aggregation: mean logits;
- frozen effect threshold: `0.05939410626888275`;
- mechanism class order: `rz_drift`, `rx_overrotation`, `ry_overrotation`.

The Step-8 confirmatory cohort is not read by Step 9A. Only the original development fit and internal-selection roots are materialized.

## Fail-closed reproduction

The run fails unless the frozen Step-7/8 config identities match, every seed reproduces its archived selected epoch and selection metrics, the ensemble threshold reproduces within the numerical tolerance, and every saved checkpoint reloads with exact tensor equality.

The deployed threshold itself is the exact archived Step-8 value, not a newly selected value.

## Output bundle

A successful run writes `seed1701.pt`, `seed1702.pt`, `seed1703.pt`, `model_selection.json`, `inference_contract.json`, and `bundle_complete.json`, with SHA-256 hashes recorded in the completion marker.

## Scientific boundary

This is artifact packaging, not another experiment: no new data, architecture, hyperparameters, epoch choice, threshold choice, confirmatory reuse, or hardware execution.

## Run after merge

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

A successful run ends with `TRIQTO STEP 9A DEPLOYMENT BUNDLE FROZEN`, selection reproduction PASS, checkpoint reload exactness PASS, no new tuning, and no spent-confirmatory access.

Upload the resulting deployment bundle ZIP for independent audit before Step 9B hardware acquisition consumes it.
