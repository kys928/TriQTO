# TriQTO — uncertainty-aware matched-response stabilization: development protocol

**Written:** 2026-10-23. **Status:** Pre-implementation research protocol; no candidate has been evaluated or selected.  
**Source of hypothesis:** Exploratory Step-15 [feature-block localization report](../evidence/step15_feature_block_instability/ANALYSIS.md).  
**Question:** Can measurement-derived uncertainty estimates reduce frozen-candidate-network decision instability **without degrading diagnostic accuracy**, when changing only deployable feature extraction and preserving the frozen Step-15 candidate network and pooling?

## Why this experiment

In a 36-example selected diagnostic cohort, matched-response features B (indices 9:15 of the finite-shot 24-D canonical feature) created the largest individual score-path conditional label disagreement at N and 2N; inferred coordinate features A (indices 0:9) were also consequential, particularly at 4N. Artificial block swaps disrupt the covariance among features. The next test must address the **shared finite-shot Jacobian** behind A and B, not assume B can be manipulated independently on actual hardware.

Original transformation (see `canonicalize_evidence`):
`J_w = diag(w) J`, `U[:,k] = J_w[:,k] / max(||J_w[:,k]||, 1e-12)`,
`d_w = w ⊙ d`, `G = U^T U`, `c = (G + λI)^(-1) U^T d_w`,
`m = U^T d_w / max(||d_w||, 1e-12)`.
Output A contains `c, abs(c), c / ||c||`; B contains `m, abs(m)`. Thus both depend nonlinearly on U and can be correlated under finite-shot error. C contains singular values, raw-column log norms, condition, residual, observed-data norm. The frozen candidate network consumes exactly 24 features with frozen training normalization.

## Mandatory dataset boundary BEFORE method selection

**Forbidden:** the 36 selected calibration-pilot examples/33 roots, all N/2N/4N archived draws from this cohort, spent confirmatory fresh holdout, sealed truth, future hardware reserve, and any outer evaluation. None may select a hyperparameter, threshold, scoring rule, or candidate remedy.

**Preferred development source:** generate a *new*, reproducible development family cohort from the frozen Step-14 circuit grammar with a new seed namespace and content-hash manifest. Split by **family**, not by resampled circuit/draw, into remedy-development FIT / remedy-SELECTION / untouched remedy-TEST. Explicitly exclude circuit-graph, family-signature, and data-product overlaps with original frozen network training/selection products, the 36 diagnostic examples, and existing fresh holdout/reserves. Require documented checked disjointness before experimental acquisition or score calculations.

**Fallback source assessment, not automatic authorization:** existing historical `development_b087edfd6629ac250299391d` with 600 fit and 150 selection families. It was used in original model-development work; it is *not a virgin method-selection cohort*. A clean separation from the 36 diagnostic examples would be necessary but not sufficient to call it independent of model fitting. If used at all, flag as **reused development**, restrict remedy work to legitimate development roles, and create a genuinely new family-held-out validation set for external support.

Keep ground-truth mechanism/location metadata out of geometry extraction and feature uncertainty estimators. Permitted development labels may be used **only** after predictions are produced, for predefined remedy-selection accuracy safeguards. No selected 36-example diagnostic labels are required or allowed.

## Proposed uncertainty estimators: hardware-compatible

With an explicit finite-shot call contract, per-call shot counts and observed eigenvalue samples/count histograms, estimate uncertainty in each basis statistic and the central-difference candidate Jacobian. Two estimators are admissible:

1. **Analytic count covariance (primary feasibility study):** for empirical outcome probability vector `p_hat = n/N` and observable-value vectors `f_a(y)`, estimate `Cov(f_a, f_b) = (E_hat[f_a f_b] - E_hat[f_a]E_hat[f_b]) / N` (document any finite-sample bias corrections). For separate plus/minus calibration shots at the same probe ε, combine independent covariances as `Cov(J[:,k]) = (Cov(s_plus)+Cov(s_minus))/(2ε)^2`; preserve cross-statistic covariance within each basis and the actual multi-basis grouping.
2. **Plug-in multinomial count bootstrap (sanity/covariance cross-check):** resample counts with original per-call N and empirical `p_hat` only, for a fixed predeclared number of replicates. Independently reconstruct U, c, m, and corresponding vector covariance. This uses no ideal probabilities, exact Jacobians, simulator truth or additional QPU shots. Check failures near zero norms, singular geometry, and known physical bounds.

Raw matched-response uncertainty (`m`) and coordinate uncertainty (`c`) must be estimated **jointly**, because both arise from the same J. Report covariance, correlated sign/direction flips, magnitude/variance ratio, and near-zero response columns. Do not assume independent feature noise or that `abs(m)` is unbiased.

## Minimal remedy candidates (predeclare before evaluation)

All preserve the frozen 24-wide feature contract, normalizer, network weights and pooling algorithm; only the extracted inputs change.

- **R0:** original canonicalization (mandatory unchanged baseline).
- **R1:** uncertainty-dependent, *smooth* attenuation of matched-response features 9:15 only, using bounded data-derived reliability `r_k∈[0,1]`; recompute both `m_k` and `abs(m_k)` consistently. Coordinate block unchanged. Reliability must use calibrated shot-count uncertainty; no exact/noiseless simulator oracle.
- **R2:** joint attenuation of correlated A+B components, with coordinate vector/norm/unit direction and matched-response values handled **consistently** from the same modified geometric estimate. Do not shrink `|c|` independently of `c` or `|m|` independently of `m`. Specify robust limits at near-zero coordinate magnitude.
- **R3:** optional pre-normalization reliability/shrinkage of weak Jacobian directions followed by *full* recomputation of all 24 canonical features and pooling evidence, using explicit hardware-available covariance. If used, differentiate from previously failed binary null-column suppression; analyze regularized inverse conditioning and zero-response guard.

Possible reliability family (development-tunable only): `r = signal_power / (signal_power + α * estimated_sampling_variance + ε)`, bounded to [0,1], with scales and ε determined before confirmation. Use a short fixed grid chosen before observing new development evaluation; no thresholds tuned on the selected 36-example diagnostic cohort. Report impact on original network’s input distribution (training normalization was frozen): large out-of-distribution feature shifts can cause spurious stability.

**Important engineering contract:** Profiles, pair distances and gates depend on calibration geometry. The primary R1/R2 tests may deliberately hold those separate pooling inputs at each original realization to isolate the **score path**, but these are algorithmic interventions, not complete deployable remedies. A secondary **fully propagated** test must recompute any affected profiles/distances/gates and assess both paths together. A method that stabilizes one path while damaging the other is not ready.

## Dataset, measurement, and evaluation plan

1. **Feasibility & freeze**: independent root/family split manifest with dataset content hashes and overlap checks, code/protocol hashes, baseline checkpoint SHA256, shot counts. No new pod unless a resource check proves CPU impractical and a bounded GPU plan is documented.
2. **Mechanics validation**: unit tests for multinomial covariance, paired plus/minus difference, observed-only bootstrap, 0/1/low-count edge cases, consistency of signed/absolute/normalized outputs, dimension and finite-value preservation, symmetry and positive semidefiniteness (up to numerical tolerance). Synthetic known-binomial tests are allowed for code correctness, not choice of remedy.
3. **Development acquisition**: use same fixed circuit families and observed evidence for every remedy, independent measurement seeds at N, 2N, 4N, unchanged total quantum shot budget per comparison. If simulation is used, emulate finite shots and **never feed exact probabilities to the estimator**.
4. **Within FIT families**, use predeclared family folds for choosing reliability estimator settings and the small remedy grid; selection families choose one frozen method exactly once according to explicit ordered criteria. No use of the reserved independent TEST, diagnostic or confirmatory results in tuning.
5. **Primary outcomes:** within-example pairwise final mechanism-label disagreement at N, plus paired balanced accuracy (or an available predeclared task correctness metric) against permitted development labels. Report per-family/root clustered intervals; do **not** reward stable constant predictions. Secondary: same statistics at 2N/4N, candidate-score perturbation vs reference, top-two margins/near-tie flips, B and A uncertainty calibration, candidate winner changes, pooling-path effects, feature shift.
6. **Selection gate:** favor remedies with statistically supported reduction of N disagreement and no material loss in task performance. Pre-specify the acceptable performance noninferiority margin **numerically before any development scores**. Use paired confidence intervals and do not silently switch the primary metric; if no candidate clears gates, select **R0 (no fix)**.
7. **Independent evaluation:** after remedy code, settings, selected model, hash manifest, and claimed estimands are frozen, perform **one** untouched family-level test. Diagnostic 36-example cohort is explanation-only; revisiting it is optional non-selecting exploratory reporting after independent freeze, never reconfirmation.
8. **Interpretation:** a success in finite-shot simulation is *not* a QPU/hardware validation claim; accuracy improvement, robustness, geometry preservation and transport to unseen layouts remain distinct hypotheses.

## Decision gates before executing remedy selection

- No stable, verified disjoint family/development dataset ⇒ **STOP**: report data-availability blocker, do not select.
- No observed-count uncertainty estimator/covariance check ⇒ **STOP**: implement or audit mechanics first.
- Source archive/input/checkpoint mismatch, changed masks or candidate ordering ⇒ **STOP**.
- Label leakage, benchmark access to sealed truth or cross-family data contamination ⇒ **STOP**.
- A remedy improves disagreement only by erasing useful evidence/accuracy ⇒ **REJECT**.
- No support beyond the selected diagnostic cohort ⇒ **NO SCIENTIFIC SUPPORT** for deployment claims.

## Requested first implementation milestone

Perform a **read-only development-data inventory and leakage preflight** first. Then implement analytic/bootstrapped finite-shot covariance helpers and standalone unit tests on synthetic counts. Only after preflight and tests should we generate the new disjoint development cohort and run the smallest R0/R1/R2 paired comparison. Do not retrain Step-15.

## What is deliberately not claimed

This protocol is a research plan, not a completed remedy or even evidence that attenuation is preferable to preserving the existing features. The earlier N=15.13% matched-response conditional disagreement is **not an expected improvement size**. The 36-example diagnostic examples are not an independent validation source.

## Cohort construction record (2026-10-25; not remedy-selection authorization)

The new [240-family cohort, EDA and independent audit](../evidence/finite_shot_cohort/README.md) are documented in PR #108. Internal construction/integrity verification passed, but protected-reserve and complete historical structural exclusion remain unverified. The corpus is quarantined; this record does not lift the mandatory independent-data gate above. No R0/R1/R2 comparison has been performed.
