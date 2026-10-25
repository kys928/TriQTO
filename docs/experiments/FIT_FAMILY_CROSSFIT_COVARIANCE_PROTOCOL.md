# TriQTO — 5-fold same-shot covariance calibration pilot (FIT only)

**Status**: prospective analysis specification, fixed before the crossfit CI. A data-informed *exploratory* estimator informed by earlier FIT results. No independent confirmation or remedy selection is claimed. The old original TEST remains disqualified; SELECTION and sealed reserves are not used.

## Question

Can an observed-count bootstrap covariance of the joint inferred coordinates and matched responses be calibrated by learning just two *global* covariance-scale factors from **other** circuit families, without increasing the future measurement shot budget or changing any frozen TriQTO feature/model weight?

## Data boundaries

Use original hash-verified finite-shot archive, 120 FIT families only. First root, first candidate, first diagnostic observation in each family. For each N=4096, 2N=8192 and 4N=16384, use all 16 independent calibration draws. Preselect count reference indices 0,7,15; from each reference individually run 192 N-shot multinomial resamples from its observed frequencies to estimate the full 6x6 covariance of (z=(c_Z,c_X,c_Y,m_Z,m_X,m_Y)). The empirical evaluation target for each reference is the 15 **other** archived independent draws, avoiding a shared reference in the target. No exact simulation probabilities enter the estimator. No oracle data, SELECTION, TEST, labels, mechanism-head logits or model scores.

## Family-disjoint crossfit

Assign all roots and reference draws of family f to the same fold by `SHA256(["triqto-fit-crossfit-covariance-v1","fold",family_id]) mod 5`. For each held-out fold and shot budget, fit separately one positive global factor for coordinate covariance trace and matched-response covariance trace on the other 4 folds: per family median empirical/plug-in ratio over its 3 references, then median over training families, clipped to [0.25,8]. Apply (\Sigma'=D\Sigma D), (D=\operatorname{diag}(\sqrt{r_c}I_3,\sqrt{r_m}I_3)), preserving PSD and coordinate/matched cross-correlations. Predeclared metrics: median and p10/p90 of out-of-fold empirical / predicted trace ratios and median absolute log ratio, both blocks at all shots; numerical PSD/finite checks. Keep all family counts; no performance-based omissions.

## Interpretation, future gate

This calibrator is trained on **FIT families only** and produces a covariance prediction from a **single** observed calibration draw at equal shot budget. However the calibration coefficients come from prior independent FIT acquisition replicates: the method is only *deployable* once those global coefficients are frozen. Out-of-fold metrics are internal FIT exploratory evidence, not independent TEST or SELECTION confirmation. Training normalizer/network/24-feature contract unchanged.

Two global scaling factors cannot generally calibrate anisotropic 6x6 covariance; trace agreement is not enough for R2 reliability. Follow-up requirements: full-matrix/orientation and marginal/predictive coverage, pairwise correlations, cross-family generalization, calibration at all legal candidates and diagnostic shots, no material inflation of linear Jacobian uncertainty; only afterward assess on independent SELECTION if historical/reserve identity attestation is complete.

This diagnostic **does not** alter R0/R1/R2 features, define an R2 attenuation hyperparameter, unseal TEST, train the Step-15 model or authorize a confirmatory evaluation.
