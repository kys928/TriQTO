# TriQTO — R0/R1/R2 paired stabilization comparison: prospective locked decision rules

**Research status:** Written 2026-10-25, *before any R0/R1/R2 network scoring on the new cohort*. Comparison design frozen; **candidate-specific implementation, covariance estimator transfer coefficients, and selected remedy remain UNFROZEN**, and this document is not permission to use quarantined SELECTION/TEST. This paper is an amendment to [MATCHED_RESPONSE_UNCERTAINTY_STABILIZATION_PROTOCOL.md](MATCHED_RESPONSE_UNCERTAINTY_STABILIZATION_PROTOCOL.md).

## Aim and comparison unit

Preserve the frozen Step-15 checkpoint, normalizer, candidate masks, original 24-dimensional canonical features, profile/pair/gate model and pooling contract unless a specifically declared intervention requires recomputation. For every legal circuit candidate and original calibration draw compare remedies with **exactly paired** source diagnostics and observed calibration counts at N=4096. Repeat at 2N=8192 and 4N=16384 as secondary analyses. Sixteen calibration draws per root are *repeated acquisitions*, not sixteen independent circuit families. Each family is the clustering unit for all uncertainty intervals, model comparison and decisions.

## Interventions

- **R0:** original feature and fully propagated pipeline without any change. Preserve hash-bound replay outputs; this is the compulsory null remedy.
- **R1:** attenuation of signed matched-response features 9:12 using per-axis calibrated observed-count uncertainty, with absolute features 12:15 recomputed from the attenuated signed values. A and C unchanged. No exact simulator probabilities. Never interpret a score-only feature substitution as evidence that changed profile/distance/gating is correct.
- **R2:** joint, covariance-aware stabilization of `c[0:3]` and `m[9:12]`, using the **full 6×6 observed-count covariance** and signed values. Any modified coordinates must have `abs(c)` and `c/||c||` recomputed; any modified m must regenerate `abs(m)`. Preserve coherent coordinate/matched covariance via one transformation. If a proposed upstream geometric modification alters Jacobians, recompute ALL affected 24 features plus profiles, pair distances, gate evidence, and pooling. A score-only R2 proxy is useful as an ablation, but cannot establish complete deployable geometry stabilization.

A future remedy grid/regularization implementation must be separately preregistered with hashes **before** any SELECTION model-score evaluation. No hyperparameters may be inferred from the old 36 diagnostics, leaked TEST, external reserves, or sealed truths. Nonlinear uncertainty covariance must be validated for PSD, prediction error, joint cross-covariance and calibrated per-axis variability before declaring R2 eligible.

## Ordered measures and decision

**Primary at N:** within-example pairwise disagreement rate across the 16 independent calibration draws for the final three-class mechanism decision, excluding clean controls from the three-class accuracy denominator. Prediction *stability alone* cannot establish success: constant always-one-class output is an unacceptable degenerate fix.

**Accuracy guard:** family-clustered paired balanced accuracy over the three injected mechanisms, with predefined **2.0 percentage point absolute noninferiority margin** vs R0. A remedy passes only if the lower bound of the 95% family-cluster paired confidence interval for `BA(remedy)-BA(R0)` exceeds **-0.020**. The 2pp value is a prospective research safeguard, not a previously measured effect. If the available 60 SELECTION families provide insufficient precision to establish it, **INCONCLUSIVE**, not 'pass'.

**Instability benefit gate:** report the within-family paired difference `Disagreement(remedy)-Disagreement(R0)`. Require the **upper** bound of its 95% family-cluster paired bootstrap interval at N to be below zero, not merely a favorable point estimate. Use 2,000 fixed-seed family-cluster resamples; resample entire families and keep all their four variants/observations/draws/candidates together. Report both point and interval, plus effect in percentage points.

**Ordered choice:** among remedies satisfying **both** gates on independent verified SELECTION, choose the one with the greatest absolute decrease in N disagreement; tie-break by higher observed mechanism-balanced accuracy, then simpler method (R1 before R2). If none clears both, retain **R0** and explicitly report no demonstrated correction. Testing multiple method settings/variants without multiplicity control invalidates the nominal 95% claim; before looking at SELECTION, prespecify at most one frozen R1 and one frozen R2 and apply a family-wise multiplicity adjustment or simultaneous confidence intervals for two comparisons. A recommended conservative adjustment is two-sided **97.5% intervals for each contrast** (Bonferroni), with the corresponding one-sided bounds, while still publishing ordinary 95% descriptive intervals.

## Secondary analysis (cannot rescue a failed primary gate)

Evaluate N/2N/4N; candidate top-two margin, score covariance, position/routing gates, pair profiles and distances, wrong-class stability, near-zero/missing-axis strata, small vs large qubit and nominal topology, calibration SNR, and feature shift relative to frozen network's training normalization. Score-only ablations are distinguished from fully propagated inference. Track physical shot cost, solver failures and finite/nonnegative constraints. Preserve clean control false-positive outputs in a separate denominator. No post-hoc selection by subset.

## Mandatory independent-source and TEST controls

1. The current 240-family corpus is quarantined because independent historical/outer/reserve **identity-only** coverage is still incomplete. Its original TEST is **not blind**, as the public deterministic permutation predicts its mechanism class exactly; no confirmatory use.
2. A new replacement TEST must receive a **separate** identity-only custodian overlap certificate on its **own 60-family candidate identities**. Certificate for FIT/SELECTION families does not certify future TEST identities. The new TEST mechanisms are blinded by a secret custodian-held permutation, not by public root IDs, and scoring remains blind. Do not claim location blindness: original first-RY proxy reveals intended qubit.
3. Never run the replacement TEST until the chosen R0/R1/R2 code hash, estimator transfer/calibration coefficients, comparison scripts, frozen model weights, primary metrics, noninferiority margin, multiplicity adjustments and source-registry attestations are committed. A separate TEST-only audit must ensure no schedule reconstruction from public metadata.
4. The independent TEST is evaluated **once** after method freeze. No post-test retuning. If the frozen remedy fails TEST noninferiority or disagrees with the primary stability hypothesis, report failure rather than recycling TEST for development.

**Current approval:** `COMPARISON_RULES_DEFINED_NO_REMEDY_SELECTED_AND_NO_TEST_AUTHORIZED`. No model scores, SELECTION/TEST labels, sealed truth or QPU work are generated by this document.
