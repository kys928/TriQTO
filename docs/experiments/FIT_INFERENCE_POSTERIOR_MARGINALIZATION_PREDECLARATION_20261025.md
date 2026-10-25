# Inference-stage uncertainty marginalization, FIT-only prospective diagnostic

**Pre-run specification, 2026-10-25. Status: new exploratory project; the previous SELECTION cohort is SPENT.** No changes to the original R0/R1/R2 results or the immutable selected R0 freeze. The original TEST (label-leaky) and prospective 60-family blind replacement TEST remain **unread, unacquired and unspent**.

## Mechanism and why it differs from R1/R2

R1/R2 attenuated observed quantum-response feature vectors directly, and failed to demonstrate a stable inference benefit on the 60-family SELECTION cohort. This study instead asks whether numerical decision noise can be reduced by **marginalizing the frozen mechanism network's posterior predictive output over measurement uncertainty**, without modifying the original observed feature tensor, optimizer, frozen network weights, or shot budget.

For a single acquisition's observed calibration counts `n` per candidate/circuit/axis/basis, use the explicitly **assumed Jeffreys Dirichlet posterior** `p|n ~ Dirichlet(n+1/2)`. This is a model of uncertainty, **not** a certified posterior for hardware. We draw 64 conditional replicates with independent deterministic private domain-separated SHA-derived RNG seeds per root/acquisition/observation/method, propagate each candidate's hypothetical calibration probabilities through the full frozen canonical 24D feature calculation and three exact SHA-256 checkpoint heads, keeping the **observed** signed-amplitude profiles and frame distance/gates fixed. Then average **post-softmax pooled class probabilities** across 64 replicates before taking a single three-class argmax.

Compare:
- **I0 (baseline R0):** unchanged measured candidate features and frozen kernel, score and argmax. Must bitwise match the original available R0 score path.
- **I1 (latent-posterior integration):** each replicate uses a Dirichlet draw of the unknown calibration outcome probabilities to reconstruct hypothetical expected Jacobian, without resimulating shots; propagate through frozen class decision and integrate softmax class probabilities.
- **I2 (posterior-predictive integration):** same Dirichlet draw, followed by a same-N **simulated** multinomial acquisition per calibration table before propagating feature/score, with no extra physical shots. It may count the observed posterior parameter uncertainty twice for the wrong estimand and may *not* improve performance.

The observed calibration counts and diagnostics remain identical across methods. We **do not** average observed quantum-coordinate features into a new deployed vector, attenuate them, change the original evidence profile, pair geometry/gating, retrain heads, touch the Step-15 rank adapter, or optimize any threshold based on the prior spent SELECTION verdict. I1/I2 are **score-path inference marginalization prototypes**, not full end-to-end uncertainty propagation of the frozen geometry.

## FIT-only cohorts, finite selection and falsifiability

Original immutable hash-verified finite-shot cohort FIT 120 families. Prior R0/R1/R2 pilot used families 0:40, so this experiment uses **only fresh FIT family indices 40:72** in deterministic manifest order, first root per family; 16 families 40:56 for comparative exploratory training, 16 families 56:72 for family-disjoint FIT screening. The first **two injected observation rows per class** are used (six total mechanism examples per root); no clean control is part of primary three-class accuracy denominator. All 16 independent calibration acquisitions at **N=4096**, original `delta`, reference and observed diagnostic count evidence, no released TEST or SELECTION. The six chosen FIT truth labels are used *only* to calculate descriptive outcomes after predictions are produced, not in any inference procedure.

No hyperparameter grid: I1 vs I2 only, exactly 64 Monte Carlo samples, Jeffreys concentration 0.5, no temperature tuning or confidence threshold fitting. Training-group decision heuristic: require at least 0.1 percentage point decrease in mean pairwise 16-draw **disagreement** versus I0 and no more than 1.0 percentage point absolute **draw-level mechanism accuracy** loss, else no FIT-supported candidate. Rank passing methods by greatest disagreement reduction; tie prefer simpler I1. Screening group reports paired accuracy/disagreement; **no retuning on screening**. Neither screening nor any mean reduction is statistically independent external confirmation; method eligibility for a future SELECTION is contingent on stability and accuracy plus a fresh development-selection cohort, with design frozen before generating that cohort.

Additional descriptive diagnostics: integrated predictive entropy, mean confidence and plurality-vote dispersion, wall-time/replicate cost, deterministic replay (same seed -> same decisions), one-acquisition-only input, and numerical finite/normalization checks. Family is the unit for any later bootstrap comparisons; 16 draws are repeated acquisitions, not 16 families.

## Stop conditions / prohibitions

No use whatsoever of original SPENT SELECTION, leaked TEST, prospective blind TEST or four protected historical/reserve namespaces. No QPU access, S3 writes, new physical shots, model retraining, or original frozen score-geometry mutation. No significance or 'method winner' claims from FIT-only exploratory work. If neither method passes FIT training/screening safeguards, report a negative result and retain frozen R0 as current reference. If method appears promising, separately design a *fresh structurally independent SELECTION* set with secret mechanism scheduling, and freeze its protocol before acquisition.

**Engineering priority:** verify correct outcome-axis count shape and reproduce I0 baseline bitwise before interpreting results.
