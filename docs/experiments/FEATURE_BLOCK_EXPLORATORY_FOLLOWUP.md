# Exploratory feature-block crossed replay — protocol recorded before this execution

Status: **data-informed exploratory follow-up**, reconstructed after prior locally
committed Work-session files were unavailable. Original broad [0,15) and strict
[0,9) results are inherited as unverified handoff observations, not the outcome
of this script. Do not call this an independent confirmatory test.

## Fixed questions and feature groups

Frozen 24-coordinate canonical feature vector as implemented in
`analyze_step14_local_frame_canonicalization.canonicalize_evidence`:

* A=[0,9): inferred coordinates, |coordinates|, normalized coordinates.
* B=[9,15): signed matched responses and absolute matched responses.
* C=[15,24): singular values, log response norms, condition, residual, data norm.

For each selected circuit and each 32-draw N/2N/4N budget, hold all pooling
inputs (profiles, pair distances, pair gates) at archived reference j. Replace
only each nonempty subset of A/B/C with values from archived draw i. Keep frozen
network weights, normalization, masks, other features and candidate support fixed.

Evaluate all seven nonempty subsets A, B, C, AB, AC, BC, ABC and each
(i,j) pair. For each reference j summarize per-example disagreement among
the 32 variant draws i. Average over all 32 references; report group contrasts,
descriptive circuit-root clustered 95% intervals (2000 bootstrap replicates,
fixed seed 20261023), and nonadditive interactions. Separate the natural
baseline from ABC-with-fixed-pooling. Require every archival natural baseline
logit/label to reproduce and fixed-fixed counterfactuals to be deterministic.

## Frozen-network small-step probes

For every reference j, use the next archived draw i=(j+1) modulo 32 and
artificial linear interpolation of all 24 feature coordinates at fractions
t in {0, 0.001, 0.01, 0.05, 0.1, 0.25, 1}. Pooling is fixed at j.
Report standardized feature distances, candidate-score response norms,
pooled logit responses, and near-tie-associated flips. Compare observed
network response gain with the conservative per-seed spectral weight bound.
These interpolations are artificial diagnostic probes, not physical finite-shot
experiments or a license to call the network hypersensitive.

## Scientific safeguards

Read only previously archived and hash-checked evidence; do not touch sealed
labels, true locations, whole holdout scoring, architecture, frozen weights,
QPU, RunPod pods, or volume objects. The archive is a selected cohort
(36 examples, 33 roots), not a population sample. Repeated draws are not
independent circuits. Compute no accuracy metric and select no remedy on
this cohort. All output is diagnostic and version/hash tracked. Do not
publish raw cohort inputs; save aggregate results to a private/ephemeral
Actions artifact and publish only non-sensitive summaries after review.
