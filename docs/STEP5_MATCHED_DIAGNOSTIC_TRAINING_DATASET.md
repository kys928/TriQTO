# Step 5 — matched diagnostic training dataset

## Purpose

Steps 3–4 established three things before any new TriQTO diagnostic model is trained:

1. exact relational `B_delta` evidence carries highly identifiable RZ/RX/RY mechanism information across generalized qubits and insertion depths;
2. injected mechanism and observed phenomenology are not the same thing, and some injected perturbations are physically negligible;
3. the smallest tested hardware-valid correlation core that survives GHZ is:
   - local X/Y/Z expectation deltas;
   - all same-basis pairwise XX/YY/ZZ correlation deltas;
   - global X/Y/Z parity deltas.

Step 5 now turns those results into the first actual train/validation dataset intended for Step 6 baselines and Step 7 `Graph + Diagnostic B_delta` training.

This stage still makes no model-performance claim.

## What `500 -> 1k -> 2k -> 5k` means

The progression counts **independent clean-circuit roots**, not distorted derivatives.

For the first stage:

- 500 independent clean circuit roots;
- 400 roots assigned to train;
- 100 roots assigned to validation;
- every derivative of one root remains in exactly one split.

The stages are nested by root index and support:

- 500;
- 1,000;
- 2,000;
- 5,000 roots.

A large number of matched perturbations derived from one circuit never increases the independent-root count.

## Clean circuit universe

The generator creates fresh deterministic simulator circuits rather than recycling the 161 Step 3.5 roots as the entire training universe.

Families:

- Bell-like;
- GHZ;
- hardware-efficient ansatz;
- phase-interference;
- QAOA-like;
- QFT-like;
- random-shallow.

Supported widths are 2, 4, 6 and 8 qubits. Bell-like roots are intentionally limited to two qubits and occupy only 5% of the deterministic family cycle so they do not dominate the training distribution.

Every clean circuit has randomized continuous parameters from a root-specific deterministic seed. Duplicate clean graph identities are rejected.

## The anti-leakage graph rule

This is a hard scientific requirement.

The deployable graph input represents the **intended/reference clean circuit**.

The simulator-only hidden intervention is inserted only while generating the observed state and its diagnostic evidence:

```text
input graph seen by model:

intended clean circuit C

hidden simulator execution:

prefix(C) -> hidden RZ/RX/RY perturbation -> suffix(C)
```

The injected gate is never serialized into `x__graph_*`.

Likewise, affected qubit, insertion depth and perturbation strength are not deployable graph metadata. They remain `audit__` fields / manifest metadata only.

Otherwise a classifier could learn the answer from the synthetic intervention construction instead of diagnosing it from evidence.

## Matched perturbation design per clean root

Each clean root produces four matched perturbation contexts near:

- 25% unitary depth;
- 50%;
- 75%;
- terminal unitary depth.

Affected qubit cycles deterministically as:

```text
(root_index + context_index) mod n_qubits
```

Strengths are balanced within each root:

```text
0.05, 0.15, 0.05, 0.15 radians
```

Within one context, clean circuit, affected qubit, insertion boundary, strength, shot count and sampled reference bundle are fixed. Only mechanism changes:

```text
RZ drift
RX overrotation
RY overrotation
```

This yields 12 injected examples per root.

Each root also contributes one clean/no-distortion control, for 13 examples per root total.

Therefore the 500-root stage contains:

- 500 clean controls;
- 2,000 RZ examples;
- 2,000 RX examples;
- 2,000 RY examples;
- 6,500 total examples.

## Finite-shot primary input

Step 5 does not train on exact simulator probabilities as its primary diagnostic input.

Each observed/reference logical example uses an empirical shot count drawn deterministically from:

```text
512, 1024, 2048, 4096
```

For each side, three basis programs are emulated:

- Z;
- X;
- Y.

Within an injected matched context, the three RZ/RX/RY siblings share the same independently sampled clean/reference bundle. Each observed mechanism side is sampled independently.

A clean control uses two independent finite-shot samples of the same clean state. Its `B_delta` therefore contains ordinary sampling noise around zero rather than an unrealistic exact-zero signature.

Exact probabilities and exact statevectors are generation/audit machinery only.

## Primary deployable diagnostic arrays

For each basis Z/X/Y the artifact contains signed observed-minus-reference deltas for:

### Local expectations

```text
Delta<P_i>
```

### Same-basis pairwise correlations

```text
Delta<P_i P_j>, i < j
```

### Global parity

```text
Delta<P_0 P_1 ... P_n-1>
```

where `P` is X, Y or Z.

All are derived from the same basis-specific sampled bitstrings. No extra basis programs are required beyond paired Z/X/Y acquisition.

The representation scales as `O(n^2)` rather than `O(2^n)`.

## Example artifact namespace

Artifacts are compressed pickle-free NPZ files.

### `x__` — deployable inputs

Includes:

- clean/intended circuit graph;
- logical-to-physical layout mapping;
- explicit basis codes;
- finite-shot local deltas;
- pair indices and finite-shot pairwise deltas;
- finite-shot global parity deltas;
- observed/reference shot counts;
- reference availability mask;
- reference-kind code.

No statevector, mechanism label, effect label, affected-qubit truth, insertion truth or perturbation strength is permitted in `x__`.

### `y__` — training supervision

Includes:

- clean-control target;
- effect-present target;
- mechanism target;
- mechanism-loss mask;
- phenomenology target;
- population component;
- phase component;
- dominance log-ratio;
- total overlap loss.

### `audit__` — privileged generation/audit metadata

Includes:

- exact correlation deltas for later shot-noise analysis;
- affected qubit;
- insertion boundary;
- perturbation strength.

Audit fields are not deployable model inputs.

## Negligible intervention policy

The privileged exact-state overlap floor remains `1e-8`.

For an injected perturbation:

```text
effect_present_target = total_overlap_loss >= 1e-8
```

Mechanism supervision is active only when an observable effect exists:

```text
mechanism_loss_mask = effect_present_target
```

Thus injected-but-negligible examples remain in the dataset but do not train the model to hallucinate a mechanism from essentially no physical effect.

Clean controls also have `mechanism_loss_mask = false`, but remain explicitly distinguishable from injected-negligible examples through `y__clean_control_target` and audit metadata.

Phenomenology remains separate from mechanism identity:

- phase-dominant;
- mixed;
- population-dominant;
- negligible;
- clean control.

## Split policy

Split assignment is deterministic from clean-root index:

```text
validation if root_index mod 5 == 0
train otherwise
```

This gives exactly 80/20 roots for every allowed stage and remains stable when the dataset expands from 500 to 1k to 2k to 5k.

No test split is created in Step 5.

The historical v0.1 test and the spent Phase 15.6 confirmatory holdout remain untouched.

## First-stage validation gate

Before the 500-root dataset is accepted, the generator checks:

- exactly 500 unique clean graph roots;
- exactly 6,500 examples;
- exactly 400 train roots and 100 validation roots;
- exactly 2,000 examples for each injected mechanism;
- exactly 500 clean controls;
- no clean root crossing train/validation;
- identical clean graph hash across all 13 derivatives of one root;
- no duplicated clean graph identities;
- mechanism loss disabled for clean and negligible examples;
- all diagnostic values finite and bounded by the physical delta range;
- minimum family and qubit-count coverage;
- no privileged/target-derived name in the deployable `x__` namespace.

A failed gate prevents publication.

## Product layout

```text
product_<content-id>/
  dataset_complete.json
  stage_validation.json
  manifests/
    clean_circuit_manifest.csv
    example_manifest.csv
    family_summary.csv
    split_summary.csv
    mechanism_summary.csv
  artifacts/
    train/*.npz
    validation/*.npz
```

The completion marker binds the configuration, runner and manifests by SHA-256.

## Run the 500-root stage

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_step5_matched_diagnostic_training_dataset.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/generate_step5_matched_diagnostic_training_dataset.py \
  --clean-circuit-roots 500
```

Output parent:

```text
/workspace/triqto-data/step5_matched_diagnostic_training
```

After the 500-root stage passes and its summary is archived, the same frozen generator can be run with 1000, 2000 and 5000 roots in sequence.

## Scientific boundaries

Step 5 establishes a training-data product, not a diagnosis-performance result.

It does not establish:

- baseline accuracy;
- TriQTO accuracy;
- finite-shot robustness across untrained shot regimes;
- noisy-backend robustness;
- calibration-drift robustness;
- real-hardware utility.

Those remain Step 6, Step 7 and later stages.
