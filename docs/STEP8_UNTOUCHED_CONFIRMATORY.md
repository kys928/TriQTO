# Step 8 — untouched confirmatory evaluation

Status: **FROZEN BEFORE CONFIRMATORY COHORT GENERATION**

Evidence status: **predeclared confirmatory same-generator replication**.

Step 7 and Step 7.1 are complete. Architecture search is closed. The selected final development architecture is `late_concat` with the dedicated signed diagnostic stream, generic local/pairwise/global-parity diagnostic encoding, graph encoding, nonlinear late fusion, and a separate magnitude/shot-aware effect pathway.

There is no Step 7.2 on the already-inspected development validation cohort.

## Scientific question

Does the already-selected `late_concat` architecture retain useful effect and mechanism diagnosis on newly generated independent clean-circuit roots from the same frozen Step-5 v3 simulator population?

This stage tests **same-generator replication/generalization to new roots**. It does not test real hardware, calibration drift, new circuit families, quantum advantage, fault tolerance, or universal mechanism identification.

## Cohort

The confirmatory cohort contains:

- 2,000 independent clean-circuit roots;
- 13 examples per root;
- 26,000 examples total;
- the same family cycle, qubit universe, intervention strengths, depth targets, RZ/RX/RY matched mechanisms, finite-shot levels, paired-reference semantics, local diagnostics, pairwise diagnostics, and global parity diagnostics as accepted Step 5 v3.

The simulator-generation root namespace is shifted by `+1,000,000`. The offset is a multiple of the frozen 20-root family cycle, so it preserves family scheduling while changing every root-specific RNG stream. The generator fails closed if any confirmatory clean graph hash or clean-group ID overlaps the accepted 5,000-root development product.

## Blinding and seal

Generation creates a `SEALED_UNEVALUATED` cohort.

The human-visible `example_manifest.csv` intentionally omits:

- mechanism;
- effect-present target;
- mechanism-loss mask;
- phenomenology;
- privileged population/phase/overlap targets.

Manifest rows are shuffled after generation so row order does not reveal the deterministic RZ/RX/RY loop ordering.

`sealed_complete.json` reports only cohort identity, counts, source identity, design/integrity checks, manifest hashes, and zero development overlap. It reports no target summaries and no model metrics.

Do **not** inspect `y__*` arrays or derive target summaries before the one-shot evaluator.

## Frozen model training

The evaluator uses exactly `late_concat` and seeds:

- 1701
- 1702
- 1703

Before confirmatory access, each seed is trained using only the original development data and the exact Step-7 training protocol:

- 3,000 fit roots;
- 1,000 internal-selection roots;
- the old 1,000 outer-development roots are not used for training or selection;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- root batch size 32;
- maximum 20 epochs;
- patience 4;
- mechanism BA selects checkpoints first, then effect BA;
- effect threshold is selected from mean logits on the development-selection roots only.

The confirmatory cohort selects no epoch, threshold, architecture, hyperparameter, seed, loss, or representation.

## Irreversible one-shot boundary

After all three development-trained models and the ensemble effect threshold are frozen, the evaluator writes:

`CONFIRMATORY_ACCESS_STARTED__<cohort_id>.json`

Only then does it load confirmatory NPZ artifacts and expose `y__*` targets to evaluation code.

If the evaluator fails after this marker is written, the cohort is scientifically spent. The evaluator refuses a second attempt. A completed evaluation additionally writes:

`CONFIRMATORY_SPENT__<cohort_id>.json`

The cohort may never again support a confirmatory claim, regardless of outcome.

## Predeclared decisions

### Primary — mechanism diagnosis

Primary mechanism confirmation is **SUPPORTED** only if both hold:

1. ensemble mechanism balanced-accuracy clean-root bootstrap 95% CI lower bound is strictly greater than `0.45`;
2. each RZ/RX/RY mechanism recall is at least `0.40`.

Otherwise the primary claim is **NOT SUPPORTED**.

### Secondary — effect detection

Supported only if effect-detection balanced-accuracy 95% CI lower bound is strictly greater than `0.65`.

### Secondary — integrated diagnosis

Supported only if four-class integrated-diagnosis balanced-accuracy 95% CI lower bound is strictly greater than `0.40`.

Bootstrap unit: independent clean root. Replicates: 5,000. Confidence level: 95%.

The Step-7 development mechanism BA (`0.5118`) is reported descriptively only. It is not used to tune the confirmation threshold after outcome.

## Protocol workflow

**Do not generate the cohort from an unmerged protocol branch.**

First merge the PR that freezes this file, the config, generator, evaluator, and tests. Only after the protocol is on `main` should the cohort exist.

After protocol merge:

```bash
cd /workspace/triqto
git fetch origin
git switch main
git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_step8_untouched_confirmatory.py
```

Generate and seal the cohort:

```bash
PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/generate_step8_untouched_confirmatory_cohort.py \
  --development-product-dir /workspace/triqto-data/step5_matched_diagnostic_training_v3/product_b2d78ad2309b71a55f9bb54f
```

The terminal must end with `SEALED_UNEVALUATED`, zero development graph overlap, no target summaries, and no model evaluation.

Then, without inspecting target arrays or deriving target summaries, run the evaluator exactly once using the sealed cohort path printed by the generator:

```bash
PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/run_step8_one_shot_confirmatory_evaluation.py \
  --development-product-dir /workspace/triqto-data/step5_matched_diagnostic_training_v3/product_b2d78ad2309b71a55f9bb54f \
  --confirmatory-cohort-dir /workspace/triqto-data/step8_untouched_confirmatory/<confirm_id>
```

Expected completion marker:

`STEP8_CONFIRMATORY_EVALUATION_COMPLETE`

The result must be reported exactly as observed. No failed gate may be repaired by changing the architecture, threshold, training protocol, cohort, or decision rule and reusing the same cohort as confirmatory evidence.
