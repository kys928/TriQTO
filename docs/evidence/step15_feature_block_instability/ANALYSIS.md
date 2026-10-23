# TriQTO — frozen Step-15 feature-block instability localization (exploratory report)

**Date:** 2026-10-23  
**Status:** Completed exploratory diagnostic; **not** a validation of a remedy, correctness, model hypersensitivity, or generalization.  
**Target:** Frozen Step-14 candidate mechanism network used by the Step-15 diagnostic pipeline (three fixed network seeds 6201/6202/6203); frozen equivalence-aware pooling.  
**No additional quantum hardware access, no new calibration acquisition, no model/optimizer training, no sealed labels.**

## Research question and context

Following the 96-draw calibration-resampling pilot and crossed score/pooling replay, determine which components of the **24-dimensional canonical finite-shot geometry feature vector** transmit decision instability through the frozen candidate-network score path. This complements the distinct variable pooling-input path (profiles, pair distances, gates). The previous crossed replay reported conditional N disagreement of 18.86% when scores varied under fixed pooling and 13.37% when pooling inputs varied under fixed scores. These earlier crossed numbers are a supplied prior report, **not independently regenerated here**; 18.86% is independently reproduced in the present all-feature fixed-pooling replay.

The experiment is **data-informed exploratory**. Original broad/strict feature-group results from an interrupted Work session motivated this follow-up and must not be represented as an independent preregistered hypothesis test.

## Evidence, sample, and fixed components

- 36 selected mechanism-diagnostic examples from 33 circuit roots; 32 archived finite-shot resamples per shot budget N, 2N, 4N (**96 draws per experiment, each involving the same selected cohort**). Repeated resamples are not independent circuits.
- Independent GitHub releases: [fixed calibration evidence](https://github.com/kys928/TriQTO/releases/tag/calibration-evidence-20261019) and [96-draw pilot archive](https://github.com/kys928/TriQTO/releases/tag/calibration-pilot-20261019). Both were SHA-256 checked by the two GitHub workflow jobs prior to extraction.
- Frozen candidate networks, normalization, candidate sets/masks, profiles/distances/gates, pooled mechanism-logit implementation, tau and temperature. For every reference draw j and replacement draw i the intervention replaces only selected candidate feature columns, then recomputes the network and pooling with **P_j held fixed**.
- Reproducible source: [feature-block script](../../scripts/v0_2/feature_block_exploratory_replay.py), [small-step script](../../scripts/v0_2/feature_block_small_step_probes.py), and [recorded protocol](../experiments/FEATURE_BLOCK_EXPLORATORY_FOLLOWUP.md). Protocol commit `e7516ad8abddd2abe0d201cdcbb848db3330c970` precedes both scripts' experiment runs.
- Completed [block replay GitHub Action](https://github.com/kys928/TriQTO/actions/runs/39990791770), run head `f5e4788b772b03b6ac2fee0312433651c11e1e55`; archived action artifact `feature-block-aggregate-results` (ID 12517861811, expiring retention).
- Completed [synthetic small-step Action](https://github.com/kys928/TriQTO/actions/runs/39991030177), run head `cdf0f789ba520de9376a3b6372e655b3aecf237a`; artifact `synthetic-small-step-aggregates` (ID 12517593353, expiring retention).
- This document preserves the aggregate results from the successful job logs so the scientific conclusions do not depend solely on expiring Actions artifacts. Full raw measurements remain in the release archives; diagnostic raw inputs are not republished here.

## Methods

The exact feature construction is defined by `canonicalize_evidence` in `scripts/v0_2/analyze_step14_local_frame_canonicalization.py`. Indexing is zero-based, half-open:

| Block | Indices | Content |
|---|---|---|
| A | 0–8 | Estimated coordinates (3), absolute coordinates (3), unit directions (3) |
| B | 9–14 | Matched responses (3), absolute matched responses (3) |
| C | 15–23 | Singular values (3), log column norms (3), log condition, residual and data norm (3) |

For each arm N/2N/4N, each fixed pooling reference j in 0…31, each replacement i in 0…31, recompute the final prediction using candidate features from i only in A, B, C, AB, AC, BC or ABC; all non-selected features, masks and pooling inputs remain at j. The reported conditional within-example disagreement uses distinct replacement draw pairs (no self-pairs), averaged over examples and all 32 references.

**Important:** crossing feature blocks disrupts correlations within physically co-generated measurements. This identifies conditional algorithmic dependence under intervention, not a unique additive causal attribution or physically realized measurement change.

## Results: feature-block crossed replay

Numbers below are percentages of **pairwise final-label disagreement**, not misclassification, model accuracy, or probability uncertainty.

| Variable feature block, pooling fixed | N | 2N | 4N |
|---|---:|---:|---:|
| A — coordinates | 7.89% | 9.45% | 10.77% |
| B — matched responses | **15.13%** | **13.13%** | 10.72% |
| C — remaining geometry | 8.77% | 8.03% | 7.15% |
| A+B | 16.12% | 15.07% | 14.68% |
| A+C | 11.47% | 12.18% | 12.53% |
| B+C | 18.67% | 16.18% | 13.42% |
| A+B+C — all features, pooling fixed | 18.86% | 17.24% | 15.27% |
| Unmodified full pipeline (scores and pooling co-vary) | 26.00% | 23.48% | 20.46% |

Full-precision rates stored in the run log and aggregate artifact. At N, B exceeds A by **7.25 percentage points** and C by **6.36 points**, descriptively; at 2N, B again leads; at 4N, A and B are approximately equal (10.77% versus 10.72%).

At N, B+C captures almost all of ABC's conditional disagreement (18.67% vs 18.86%) while A alone yields 7.89%. However, the difference in conditional disagreement **must not** be interpreted as a variance-explained percentage or imply that coordinates are harmless.

Interactions are non-additive: the empirical fraction of (reference, replacement, example) triples in which ABC flips relative to the fixed reference while **none** of A/B/C alone flips was **3.57% (N), 3.14% (2N), 2.81% (4N)**. This “ABC-only flip” event is a concrete interaction example; the statistic is not a full Shapley decomposition or physical causal effect.

The reported contrasts are descriptive on the selected cohort. In particular, A and B changing direction with shot budget suggests different noise-response behavior but does not establish convergence or hardware transfer.

## Synthetic small-step probes (distinct diagnostic)

With pooling fixed at reference j, the frozen network was given the artificial interpolation `x(t) = x_j + t (x_(j+1 mod 32) - x_j)` for t = 0, .001, .01, .05, .1, .25, 1. The quantity below is the percentage of final decisions changed **relative to the reference at t=0**, not the pairwise disagreement statistic in the previous table.

| Interpolation fraction t | N | 2N | 4N |
|---|---:|---:|---:|
| 0 | 0.00% | 0.00% | 0.00% |
| 0.001 | 0.00% | 0.09% | 0.00% |
| 0.01 | 0.09% | 0.43% | 0.52% |
| 0.05 | 1.22% | 0.95% | 1.30% |
| 0.1 | 2.00% | 2.34% | 2.26% |
| 0.25 | 5.21% | 5.82% | 5.47% |
| 1 | 19.62% | 17.97% | 15.97% |

The smallest synthetic feature moves rarely flip a label in this setup; no numerical criterion for “model hypersensitivity” was established. Small changes may cross a naturally small decision margin. Physical realizability, input scaling and local score gain must be established independently before labeling the network excessively sensitive.

## Execution checks and deviations

Both Actions completed **successfully** with hash-verified input release archives and frozen-network inference. The block-swap implementation checked the candidate masks for equality across draws, compared one original reference (j=0) against stored logits to tolerance 1e-4, and verified self-swap (i=j) labels for **every** group/reference. The small-step implementation checked t=0 logits against stored predictions at each reference. These checks support replay consistency; **the script did not independently re-evaluate every unmodified archived draw against the historical logits**, so a stronger claim of exact all-draw logit reproduction would be unwarranted.

**Deviations/incomplete planned analyses:**
1. The protocol proposed 2,000 root-cluster bootstrap intervals; the executed script only saved point estimates and reference-level min/max. No inferential 95% intervals for the new contrasts are available.
2. The protocol requested standardized feature-change scale and an explicit spectral-weight upper bound on candidate-score gain; the small-step script saved raw feature norms, candidate/pooled changes and conditional flip margins, **but did not implement standardized distances, local ReLU response checks or spectral bounds**.
3. No paired accuracy, correctness or calibration-quality evaluation was run, and no remedial hyperparameters were selected.
4. No independently enforced new regression suite was added for these exploratory scripts; successful Actions execution is not equivalent to full algorithmic test coverage.

These omissions must be repaired or preregistered as an explicitly **post-hoc** refinement; they cannot be silently backfilled as part of the original protocol.

## Interpretation, limitations and scientific decision

**Supported, within this selected cohort:** matched-response features B transmit the largest *individual* fixed-pooling decision-instability effect at N and 2N, while coordinates A are similarly consequential at 4N. Candidate network score instability is distributed across feature blocks; interactions matter. Tiny artificially interpolated steps generally flip fewer labels than full archived-draw displacements.

**Not supported:** “matched response noise is the unique cause,” “the model is hypersensitive,” “the quantum state itself changed,” “the intervention improves accuracy,” or “a hardware-compatible fix has been validated.” No improvement in generalization or physical robustness follows from this replay alone. Pooling profiles remain an independently known contributor from earlier crossed-path work.

**Decision:** Retain frozen Step-15. Advance to [uncertainty-aware matched-response stabilization protocol](../experiments/MATCHED_RESPONSE_UNCERTAINTY_STABILIZATION_PROTOCOL.md), with separate development families/circuits and family-level selection/holdout controls. Treat this diagnostic cohort as **explanation-only, not a method-selection set**.

## Reproducibility and provenance

- Frozen scientific feature definition: `scripts/v0_2/analyze_step14_local_frame_canonicalization.py`.
- Selected-circuit release archives carry independent SHA256 receipts, manifests and hashes; the Actions scripts verify the archives before use.
- Block run: [39990791770](https://github.com/kys928/TriQTO/actions/runs/39990791770), exploratory script source commit `8c31961bafdbfa08731b988505021e425c1d5285`; workflow commit `f5e4788b772b03b6ac2fee0312433651c11e1e55`.
- Step run: [39991030177](https://github.com/kys928/TriQTO/actions/runs/39991030177), script commit `75d5e0bc54af70ce2423d6861a07eca2baa48555`; workflow commit `cdf0f789ba520de9376a3b6372e655b3aecf237a`.
- No raw sealed truth, location labels or private credentials are included in this report.
