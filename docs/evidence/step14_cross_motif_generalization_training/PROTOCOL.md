# Step 14 — Cross-motif mechanism-generalization training protocol

Status: **FROZEN BEFORE DATASET GENERATION**

## 1. Why Step 14 exists

Step 12 failed its predeclared cross-backend phase-generalization gate: the frozen Step-10C ensemble achieved 8/18 mechanism classifications, while the report-only Step-9A baseline achieved 9/18. Step 13 then decomposed that failure using the already-acquired Step-12 data.

Two results are decisive for the Step-14 design:

1. the measured Step-12 Pauli diagnostic geometry retained strong mechanism-specific structure, including cases that the learned model misclassified; and
2. replacing the real QPU diagnostics with exact statevector diagnostics did not improve Step-10C mechanism correctness: hardware = 8/18, fully ideal = 8/18, ideal-local-only = 8/18, ideal-pair/parity-only = 8/18. The `cz_echo_ramsey` family remained 0/6 under exact ideal diagnostics.

Therefore Step 14 tests the narrowest plausible repair first: **broaden training context while preserving the architecture and hardware-facing input contract**. Step 14 does not assume that the architecture is proven sufficient; it tests whether the observed failure can be repaired by coverage/generalization alone before any representation redesign is justified.

## 2. Scientific hypothesis

The frozen Step-10C model has learned mechanism mappings that are too entangled with circuit context. If that is the primary problem, then warm-start fine-tuning on a much broader, leakage-safe distribution of circuit skeletons should improve mechanism generalization on completely held-out circuit families without materially degrading the legacy domains.

The intervention is therefore deliberately data-first:

- no new architecture;
- no graph-contract change;
- no new privileged inputs;
- no new loss terms;
- no scratch-vs-warm initialization experiment;
- no QPU data in training;
- no Step-12 case replay during model selection.

If this intervention fails under the frozen outer gate, the next step is not another dataset tweak. The next step is a representation/fusion/head decomposition on fixed data.

## 3. Frozen starting model

Step 14 warm-starts the three frozen Step-10C warm-start checkpoints, seeds 1701, 1702, and 1703. State dictionaries must load strictly. Optimizer state is not reused; each seed receives a fresh AdamW optimizer.

The model remains `late_concat` with 453,829 trainable parameters. Every existing parameter remains trainable. No layer is added, removed, frozen, widened, narrowed, or reparameterized.

The frozen Step-10C checkpoints remain the primary baseline for all Step-14 outer comparisons.

## 4. Dataset composition

Training consists of three domains:

1. the original Step-10 original-domain fit partition, unchanged;
2. the original Step-10 deployment bridge fit partition, unchanged; and
3. a new Step-14 cross-motif domain.

The first two are reused byte-for-byte from the immutable Step-10 mixture. No spent Step-10B or Step-10C outer cohort is read or reused.

### 4.1 Cross-motif family hierarchy

The new domain contains 1,050 independently generated circuit families. Each family defines one structural reference skeleton and one injection-context class. Four roots are produced from each family by changing continuous gate angles and allowed spectator single-qubit slots while preserving the family skeleton.

Every root creates exactly 13 examples:

- one clean control;
- four RZ-drift examples;
- four RX-overrotation examples;
- four RY-overrotation examples.

The four distortion strengths are fully crossed with the three mechanisms. Thus mechanism identity cannot be inferred from strength or family membership.

The family is the leakage unit. All variants and every clean/distorted derivative of a family remain in one partition.

### 4.2 Family split

Partition assignment is deterministic from `family_index mod 7`:

- folds 0,1,2,3: fit — 600 families, 2,400 roots, 31,200 examples;
- fold 4: selection — 150 families, 600 roots, 7,800 examples;
- fold 5: untouched simulator outer — 150 families, 600 roots, 7,800 examples;
- fold 6: future hardware reserve — 150 families, 600 roots, 7,800 possible examples.

Only fit and selection are materialized before training. Simulator outer is materialized only after all three selected checkpoints and the ensemble effect threshold are frozen. The future hardware reserve is not materialized, inspected, summarized, or scored during Step 14.

This separation is intentional. Step 14 must improve on unseen circuit families, not merely on new angle values for familiar skeletons.

## 5. Circuit grammar

The generator samples 2-, 3-, 4-, and 5-qubit reference circuits from six topology classes:

- path;
- hub-spoke;
- two-branch;
- staggered path;
- directed fan;
- mixed sparse.

Reference circuits contain 4–9 logical layers and at least two entangling operations. Single-qubit operations are sampled from H/RX/RY/RZ with frozen probabilities 0.15/0.25/0.25/0.35; parameterized angles are sampled over [-pi, pi]. Two-qubit operations are CX or CZ with equal probability.

The injected mechanism acts on one active logical qubit at a legal boundary drawn from three context classes: pre-entangling, inter-entangling, or post-entangling recombination. A legal boundary requires operations before the injection, a noncommuting operation after the injection on the affected qubit, and entangling structure in the reference circuit. RZ, RX, and RY variants for the same root and strength use the same affected qubit and the same injection boundary.

Exact Step-12 reference signatures and the exact Step-11 phase-interference signature are excluded. Step-12 examples themselves are never inserted into Step-14 training.

## 6. Strength and shot design

Each root receives one deterministic uniform strength draw from each ordered interval:

- [0.05, 0.12);
- [0.12, 0.20);
- [0.20, 0.28);
- [0.28, 0.36].

The half-open convention on the first three intervals makes the bins disjoint. The same four strengths are used for all three mechanisms within the root.

Finite-shot counts use 512, 1024, 2048, and 4096 shots. Within a root, the four strengths are assigned the four shot levels by a root-seeded permutation, and that same permutation is shared by RZ/RX/RY. The clean example receives a root-seeded shot level sampled from the same set.

Therefore shot count cannot act as a mechanism label proxy and cannot deterministically distinguish clean from distorted examples.

Reference and observed counts are sampled independently from the corresponding simulator probabilities.

## 7. Layout semantics under the frozen graph adapter

Step 13 leaves **graph-conditioned circuit context** as a leading failure candidate. Under the current frozen Step-7 graph adapter, that context consists of the logical reference-circuit structure represented by nodes, gate events, logical incidences, ordering/layers, gate types, and gate parameters.

The current adapter reads `x__layout_logical_to_physical` to determine the logical qubit count, but it does **not** expose the absolute physical-qubit identifiers themselves as node, edge, or gate features. Therefore Step 14 does not randomize synthetic physical IDs and does not claim to train invariance to absolute IBM qubit identity. Doing so would be scientifically inert under the present input contract.

Step 14's generalization target is consequently circuit/motif context under the existing deployable representation—not backend-ID or calibration-context invariance. If later evidence shows that physical layout or hardware metadata must be represented explicitly, that would be a separate input-contract/architecture question and cannot be smuggled into Step 14.

## 8. Model-blind identifiability admission

Before a generated family is admitted, a statevector-only audit is run at probe strength 0.12. No model prediction is used.

A family is accepted only if:

- each true mechanism produces diagnostic delta norm >= 0.02; and
- every pair of RZ/RX/RY mechanism diagnostic templates is separated by distance >= 0.05.

A failing family is rejected wholesale and generation advances deterministically to the next family seed.

This is not a performance filter. It only prevents the training set from being dominated by cases where the frozen diagnostic contract carries essentially no mechanism information.

## 9. Training schedule

Each Step-14 seed starts from the matching Step-10C checkpoint with a fresh optimizer.

Frozen optimization settings:

- AdamW;
- learning rate 1e-4, constant;
- weight decay 1e-4;
- root batch size 32;
- gradient clip norm 1.0;
- effect-loss weight 1.0;
- mechanism-loss weight 1.0;
- maximum 30 epochs;
- early-stopping patience 5;
- early-stopping minimum delta 0.0005.

Every epoch contains equal optimizer-block counts from the legacy original, legacy Step-10 bridge, and new cross-motif fit domains. Shorter block lists cycle deterministically. No domain-weight, learning-rate, clipping, batch-size, or loss-weight sweep is permitted.

## 10. Per-seed checkpoint selection

Before training, each matching Step-10C seed checkpoint is evaluated on the three development selection domains to establish baseline metrics.

A Step-14 epoch is eligible only if its mechanism balanced accuracy on each legacy selection domain remains within 0.02 of the matching-seed Step-10C baseline.

Among eligible epochs, checkpoint order is fixed:

1. highest Step-14 cross-motif selection mechanism balanced accuracy;
2. highest minimum per-mechanism recall on cross-motif selection;
3. highest minimum of legacy-original and legacy-bridge mechanism balanced accuracy;
4. earliest epoch.

Outer data cannot select a checkpoint. Step-12 data cannot select a checkpoint. Human monitoring cannot override this rule.

## 11. Ensemble effect threshold

After all three seed checkpoints are frozen, one ensemble effect threshold is selected from the concatenated development selection logits of all three domains.

The primary objective is maximum macro-average effect balanced accuracy across the three domains. Ties are broken by:

1. lower macro-average false-positive rate;
2. threshold closest to the frozen Step-10C threshold;
3. smaller numerical threshold.

The chosen threshold is written to `selection_freeze.json` and is immutable before any Step-14 outer cohort is materialized.

## 12. Fresh outer evaluation

Step 14 uses two untouched outer components.

### 12.1 Cross-motif outer

This is family fold 5 from the frozen Step-14 grammar: 150 unseen families, 600 roots, 7,800 examples.

### 12.2 Fresh legacy-retention outer

A new seed namespace generates:

- 500 fresh original-domain clean roots = 6,500 examples;
- 60 fresh bridge parent groups x 8 variants = 480 roots = 6,240 examples.

These use the same legacy generator semantics and label definitions, but cannot overlap any spent Step-10B or Step-10C outer product.

The Step-14 ensemble and the frozen Step-10C ensemble are evaluated on exactly the same newly materialized outer data. Neither model gets a different cohort.

## 13. Statistical unit and bootstrap

All reported uncertainty uses 2,000 bootstrap replicates with frozen seed 2026082503 and 95% confidence intervals.

Resampling units preserve dependence:

- cross-motif outer: family ID;
- legacy original outer: clean root;
- legacy bridge outer: parent group.

The candidate-minus-Step-10C improvement uses paired resampling on the same cross-motif family draws.

## 14. Predeclared support gate

Step 14 supports a simulator generalization repair only if **every** criterion passes:

- cross-motif mechanism BA >= 0.80;
- cross-motif mechanism BA bootstrap lower bound >= 0.75;
- minimum RZ/RX/RY recall >= 0.70;
- cross-motif effect BA >= 0.90;
- candidate minus Step-10C cross-motif mechanism BA >= +0.05;
- paired-bootstrap lower bound for candidate minus Step-10C mechanism BA > 0;
- fresh original-domain mechanism BA drop vs Step-10C <= 0.02;
- fresh original-domain effect BA drop vs Step-10C <= 0.02;
- fresh legacy-bridge mechanism BA drop vs Step-10C <= 0.02;
- fresh legacy-bridge effect BA drop vs Step-10C <= 0.02.

If all pass, the only allowed primary interpretation is:

`CROSS_MOTIF_GENERALIZATION_REPAIR_SUPPORTED_IN_SIMULATION`

Otherwise:

`CROSS_MOTIF_GENERALIZATION_REPAIR_NOT_SUPPORTED_IN_SIMULATION`

The threshold cannot be relaxed after seeing the outer result.

## 15. Step-12 and QPU boundary

The Step-11 and Step-12 QPU counts are forbidden as training examples. Step-12 predictions, logits, and case-level outcomes are forbidden from model selection.

After the Step-14 outer result has been frozen, Step-12 may be replayed once as a **report-only known-failure diagnostic** to answer whether the known `cz_echo_ramsey` failure was repaired. That replay cannot alter Step-14 selection, threshold, gate, or scientific conclusion.

No QPU execution occurs in Step 14.

## 16. What happens next

If the complete Step-14 support gate passes, Step 15 may freeze a new blind hardware protocol using only the untouched fold-6 reserve. That future protocol must be frozen before materializing the reserved cases.

If the Step-14 support gate fails, Step 15 QPU execution is not justified. The next stage must isolate graph encoder, diagnostic encoder, fusion, and mechanism head behavior using fixed Step-14 data before any architecture modification is proposed.

## 17. Claim boundary

A successful Step 14 would show that a data-first cross-context intervention repaired simulator mechanism generalization under a predeclared held-out-family test. It would **not** prove physical-hardware generalization, full TriQTO correctness, quantum advantage, universal circuit optimization, or architecture optimality.

A failed Step 14 would falsify the narrow claim that broader context coverage alone is sufficient under this frozen training intervention. It would not prove that the entire TriQTO hypothesis is false.
