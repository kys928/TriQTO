# Step 10 — leakage-safe deployment-domain training distribution and warm-start benchmark

Status: **FROZEN BEFORE STEP-10 DATASET OR TRAINING OUTCOME**

## Purpose

Steps 9G–9H isolated deployment-domain simulator coverage as the leading explanation for the frozen Step-9A phase-mechanism failure, while Step 9H showed that the current Step-7 graph-information contract is sufficient in the targeted bridge audit. Step 10 therefore changes the training distribution before changing the model architecture.

Step 10 has two ordered parts:

1. **Step 10A — dataset freeze.** Build an immutable mixture product that references the frozen Step-5 v3 product without modifying it and adds a larger simulator-only deployment-domain bridge family.
2. **Step 10B — initialization benchmark.** Train the unchanged `late_concat` Step-7 model on that mixture from (a) the frozen Step-9A deployment weights and (b) scratch, with identical training/evaluation rules.

No QPU submission or IBM credential is permitted in Step 10.

## Frozen scientific requirements

### Existing domain is retained, not replaced

The original domain remains the frozen Step-5 v3 product:

- product: `product_b2d78ad2309b71a55f9bb54f`
- 5,000 clean roots / 65,000 examples
- original root-safe Step-7 fit / selection / outer-development-validation partitions remain unchanged.

The Step-10 product references this frozen product; it does not regenerate or relabel it.

### Deployment-domain bridge family

The bridge is a larger simulator-only family based on the successful Step-9G local phase-interference motif. It is not a copy of the three frozen Step-9D QPU cases.

Frozen default size:

- 300 parent groups
- 8 nearby circuit variants per parent
- 2,400 clean bridge roots
- 13 examples per root (1 clean control + 4 matched contexts × RZ/RX/RY)
- 31,200 bridge examples.

Bridge roots vary the phase-interference angle, spectator rotations, motif placement, qubit count (2–4), spectator structure, intervention context, strength, and finite-shot acquisition. The exact frozen Step-9D pilot graph is forbidden and checked explicitly.

### Leakage-safe split unit

The split unit is the **parent bridge group**, not the individual generated root.

All eight nearby variants from one parent group must remain together. Parent fold is fixed by `parent_group_index mod 5`:

- folds 1, 2, 3 → fit
- fold 4 → selection
- fold 0 → outer validation.

Therefore neighboring variants intentionally created from the same parent cannot cross fit/selection/validation boundaries. All 13 derivatives of each clean root also remain together.

Expected bridge counts:

- fit: 1,440 roots
- selection: 480 roots
- outer validation: 480 roots.

### Model input boundary

The Step-10 neural benchmark uses the unchanged current Step-7 input/model contract:

- intended/reference clean circuit graph only;
- finite-shot signed Z/X/Y local expectation deltas;
- same-basis pairwise correlation deltas;
- global parity deltas;
- acquisition/reference metadata already allowed by Step 7.

The model does **not** receive hidden mechanism, affected qubit, insertion depth/boundary, strength, family label, exact simulator diagnostic, statevector, or parent-group identity.

No graph-feature dimension or model-architecture change is allowed in Step 10.

## Step 10B initialization comparison

Architecture is frozen to the existing `late_concat` Step-7 model with 453,829 trainable parameters.

Seeds are frozen to `1701, 1702, 1703`.

Two initialization conditions are compared:

- **warm_start**: load the corresponding serialized Step-9A `seed1701.pt`, `seed1702.pt`, `seed1703.pt` `state_dict`, then create a new AdamW optimizer;
- **scratch**: instantiate the identical architecture from the same seed and create the same new AdamW optimizer.

This is warm-starting, not optimizer-state resume. No Step-9A optimizer state is claimed or required.

Training hyperparameters inherit the frozen Step-7 development values unless explicitly fixed in the Step-10 config: AdamW, learning rate 3e-4, weight decay 1e-4, gradient clip 1.0, maximum 20 epochs, patience 4, root batch size 32.

### Domain-balanced training schedule

Original-domain and bridge-domain fit blocks are shuffled independently. Each epoch uses an equal number of optimizer blocks from each domain; the shorter block list is cycled deterministically. This prevents the larger original domain from numerically drowning the new bridge domain.

### Independent selection reporting

At every epoch, original-domain selection and bridge-domain selection metrics are computed separately.

The frozen warm-start epoch-0 original-domain selection performance defines a per-seed retention floor. A trained checkpoint is selection-eligible only if:

- original mechanism balanced accuracy ≥ epoch-0 warm-start original mechanism BA − 0.02; and
- original effect balanced accuracy ≥ epoch-0 warm-start original effect BA − 0.02.

The corresponding scratch run must satisfy the same warm-start-derived floors. Among eligible checkpoints, bridge mechanism balanced accuracy is primary; ties use the minimum of the two domain mechanism BAs, then bridge effect BA, then earlier epoch.

Outer validation never selects epochs, thresholds, learning rates, mixture ratios, or any other hyperparameter.

## Final evaluation and anti-forgetting gate

After checkpoint selection, both domains are evaluated independently on untouched outer-development validation roots.

For each initialization condition report at minimum:

- effect balanced accuracy and class recalls;
- mechanism balanced accuracy and RZ/RX/RY recalls;
- integrated diagnosis balanced accuracy;
- 95% clean-root/parent-group bootstrap intervals;
- individual-seed and three-seed mean-logit ensemble results.

A single effect threshold is selected from the combined original+bridge **selection** logits for each initialization ensemble and is then frozen for both outer domains.

The deployment-domain mechanism gate inherits the targeted audit reference:

- balanced accuracy ≥ 0.80;
- 95% bootstrap lower bound ≥ 0.75;
- minimum mechanism recall ≥ 0.70.

Original-domain retention is evaluated relative to the untouched Step-9A warm-start ensemble baseline on the same original outer-validation cohort. Step-10 is not considered successful if improvement on the bridge is purchased by an original-domain mechanism or effect BA drop greater than 0.02.

## Interpretation boundary

A successful Step-10 warm-start result would support the claim that the frozen TriQTO phase weakness can be addressed by leakage-safe simulator coverage while preserving the current architecture and reusing prior weights.

It would **not** establish universal generalization, quantum advantage, full hardware robustness, or confirmatory QPU performance. The Step-9D QPU pilot remains exploratory and is not reused for Step-10 training or model selection.
