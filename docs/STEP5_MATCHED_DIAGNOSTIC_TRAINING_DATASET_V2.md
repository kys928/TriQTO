# Step 5 v2 — repaired matched diagnostic training dataset

## Why v2 exists

The first 500-root Step 5 product (`product_bb5cbdcd33a591f2f8c97b7c`) passed its original publication checks but failed a deeper full-artifact EDA for two deterministic design reasons:

1. global `root_index % 5` split assignment aliased with the deterministic family cycle, leaving several families train-only or validation-only;
2. the fixed strength schedule was perfectly confounded with insertion depth.

The v1 artifacts themselves were intact. All 6,500 uploaded NPZs matched their manifest hashes, loaded pickle-free, used one schema, matched targets/audit metadata, and contained finite bounded diagnostic arrays. V2 therefore fixes the **data design**, not corruption.

The full v1 EDA is archived under:

`docs/evidence/step5_500_v1_rejected/EDA_SUMMARY.md`

## V2 clean-root universe

Allowed stages remain:

`500 -> 1000 -> 2000 -> 5000` independent clean-circuit roots.

Derived perturbations never count as independent roots.

Circuit families remain:

- Bell-like;
- GHZ;
- hardware-efficient ansatz;
- phase-interference;
- QAOA-like;
- QFT-like;
- random-shallow.

Non-Bell qubit choices are now:

`2, 3, 4, 6, 8`.

Bell-like is still forced to two qubits.

At the deterministic 500-root v2 plan, 3q coverage is 72 train roots and 16 validation roots.

## Repaired nested split

The split unit remains the independent clean root, but split assignment now uses the zero-based occurrence index **within each circuit family**:

```text
validation iff family_occurrence_index % 5 == 0
train otherwise
```

At 500 roots this produces exact 80/20 coverage in every family:

| family | train | validation |
|---|---:|---:|
| Bell-like | 20 | 5 |
| GHZ | 60 | 15 |
| HEA | 60 | 15 |
| phase-interference | 60 | 15 |
| QAOA-like | 60 | 15 |
| QFT-like | 60 | 15 |
| random-shallow | 80 | 20 |

The first 500 assignments remain unchanged when planning 1000, 2000 or 5000 roots.

## Repaired strength/depth factorial

Each root still contributes four matched intervention contexts near 25%, 50%, 75% and terminal unitary depth.

Strengths are `0.05` and `0.15`, but the schedule alternates by within-family occurrence parity:

```text
even family occurrence: 0.05, 0.15, 0.05, 0.15
odd family occurrence:  0.15, 0.05, 0.15, 0.05
```

Thus weak and strong interventions both occur at early, middle, late and terminal locations, in every family and both train/validation splits.

RZ/RX/RY remain matched within each clean-root/qubit/depth/strength/reference context.

## Per-root examples

Each root still produces 13 examples:

- 1 clean/no-distortion control;
- 4 matched contexts x 3 mechanisms = 12 injected examples.

At 500 roots:

- 500 clean controls;
- 2,000 RZ;
- 2,000 RX;
- 2,000 RY;
- 6,500 total examples.

## Hardware-facing diagnostic input

The Step 4.1 selected core is unchanged. For X/Y/Z, persist finite-shot paired-reference deltas for:

- local one-body expectations;
- all same-basis two-body correlations;
- global parity.

The clean intended/reference circuit graph is the only deployable graph. The hidden simulator perturbation gate is not exposed.

Primary acquisition still cycles 512, 1024, 2048 and 4096 shots. Clean controls use independent observed/reference samples so they contain finite-shot noise rather than exact-zero `B_delta`.

## Privileged supervision

Statevectors are transient generation machinery only. Persisted targets include:

- clean-control target;
- effect-present/negligible target;
- mechanism target;
- mechanism-loss mask;
- phenomenology target;
- population/phase components;
- total overlap loss.

Injected-but-negligible examples remain in the product but have mechanism loss disabled.

## Raw reference-window IDs are not features

The manifest may contain `meta_reference_window_id` for provenance/matching. The semantic string is **meta/audit only** and is not persisted under `x__`.

Any future model-facing timing information must be neutral numerical metadata such as reference age, observed/reference time difference or same-calibration-window status.

## Generate the repaired 500-root stage

```bash
cd /workspace/triqto

git fetch origin

git switch agent/step5-matched-diagnostic-training-dataset 2>/dev/null || \
  git switch --track origin/agent/step5-matched-diagnostic-training-dataset

git pull --ff-only

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_step5_matched_diagnostic_training_dataset_v2.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/generate_step5_matched_diagnostic_training_dataset_v2.py \
  --clean-circuit-roots 500
```

V2 products are written under:

`/workspace/triqto-data/step5_matched_diagnostic_training_v2/product_*`

## Mandatory full-artifact EDA

Do not unlock 1000 roots merely because generation finishes.

Run:

```bash
PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_step5_training_dataset_eda.py
```

The EDA scans every NPZ and verifies:

- artifact SHA against manifest;
- pickle-free loading and schema consistency;
- target/audit/identity agreement with manifest;
- absence of privileged truth in `x__` names;
- finite/bounded local, pairwise and parity diagnostics;
- family/split association;
- 3q presence in both splits;
- depth/strength association and missing cells;
- empirical-vs-exact shot error by shot count and strength.

Possible decisions:

- `PROMOTION_READY`;
- `BLOCKED`.

Only `PROMOTION_READY` unlocks the 1000-root stage.

## Finite-shot EDA boundary

V1 showed that shot error often exceeds the exact signal for weak perturbations, particularly at low shots. V2 reports this explicitly but does not use it as a dataset rejection gate by itself.

This is deliberate. Exact simulator quantities may quantify the challenge but may not become deployable features. Step 6 cheap baselines determine whether the finite-shot evidence is learnable enough to justify later model training.

## Scientific boundaries

- no historical v0.1 test access;
- no spent confirmatory reuse;
- no model/classifier training during Step 5 generation/EDA;
- no QPU execution;
- no layout/noisy-hardware robustness claim;
- v1 remains an immutable rejected data-quality candidate rather than being rewritten after the fact.
