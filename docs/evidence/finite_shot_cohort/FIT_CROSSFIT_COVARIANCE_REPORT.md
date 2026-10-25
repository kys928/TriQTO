# FIT-only family-held-out observed-count covariance calibration

**Exploratory and data-informed scientific result, 2026-10-25.** The source hypothesis emerged after the original FIT nonlinear covariance discrepancy, so the internal 5-fold experiment is *not untouched confirmatory validation*. Original immutable cohort: `sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d`.

Protocol: [FIT_FAMILY_CROSSFIT_COVARIANCE_PROTOCOL.md](../../experiments/FIT_FAMILY_CROSSFIT_COVARIANCE_PROTOCOL.md). Source code: `scripts/cohort/crossfit_nonlinear_covariance_fit.py`, tests `tests/test_fit_covariance_crossfit.py`. Successful hash-verified, FIT-only GitHub Actions [run 40213473847](https://github.com/kys928/TriQTO/actions/runs/40213473847): **3 mechanics tests passed, 120 distinct FIT families processed**, result artifact `five-fold-fit-covariance-calibration` (ID 12606896074, 30-day retention).

## Method

For each FIT family, the first root, candidate and diagnostic observation was used. At N=4096, 2N and 4N, each of 3 predeclared reference acquisitions 0, 7 and 15 supplies only its own **observed measurement counts** for a 192-replicate multinomial nonlinear-bootstrap 6×6 covariance. Ground-truth labels, ideal simulator probabilities and other acquisitions do not enter this estimate. Empirical reference covariance is estimated from the **15 remaining independent calibration draws**, excluding the reference draw.

Five folds are assigned by SHA256 of the family ID; every observation/reference for a family belongs in one fold. Two training-fold robust median trace-ratio correction factors per shot budget, `f_c` and `f_m`, are fit on the four other family folds (training medians over reference draws and then families; clipping [0.25,8]). On the held-out families, full six-dimensional covariance is updated by congruence `D Sigma D` with `D=diag(sqrt(f_c) I3,sqrt(f_m) I3)`, preserving PSD and cross-block correlations. Predictions require only one count draw and previously fitted global factors: **no extra QPU shots**. This *fits the covariance estimator*, NOT the frozen TriQTO neural network or R2 reliability weights.

## Results

Median actual (15-draw) empirical variance divided by the predicted variance:

| Shot budget | Uncorrected coordinates | Out-of-fold corrected coordinates | Uncorrected matched | Out-of-fold corrected matched |
|---|---:|---:|---:|---:|
| N | 1.9089 | **1.0031** | 1.3909 | **1.0184** |
| 2N | 2.2579 | **0.9700** | 1.2670 | **0.9889** |
| 4N | 2.2878 | **0.9931** | 1.2205 | **0.9662** |

Median **absolute log trace calibration error** improved for coordinates: N 0.6499→0.4103, 2N 0.8144→0.4001, 4N 0.8336→0.4106. For matched responses: N 0.3581→0.2142, 2N 0.3247→0.2968, 4N 0.3412→0.2964. The fold assignment excludes any family from learning its own correction.

## Interpretation and remaining scientific limitations

These results demonstrate **descriptive median trace calibration on FIT-held-out families**, not full covariance reliability or prediction accuracy. Global block scaling preserves PSD but **cannot repair incorrect eigenvectors, within-block anisotropy, empirical sign/direction changes, or calibration failures at other candidates/diagnostic settings**. The reference 15-draw covariance is a finite and potentially noisy estimate; medians can mask major tails. Fit families were already explored to identify this idea. The broad source cohort remains subject to unresolved external historical/reserve source attestation; the original TEST is label-leaky.

**Current verdict: `PROMISING_FIT_CROSSFIT_NOT_EXTERNALLY_VALIDATED`.** No current R2 remedy is selected. No R0/R1/R2 scores, selection-family outcomes, hidden TEST truth, QPU, pod or hardware noise were accessed. Next: numerical full-matrix and coverage evaluation on disjoint FIT folds as a screening diagnostic; *only after a real independent source attestation* evaluate fixed coefficient estimation on SELECTION and freeze one remedy before a completely new blind TEST.
