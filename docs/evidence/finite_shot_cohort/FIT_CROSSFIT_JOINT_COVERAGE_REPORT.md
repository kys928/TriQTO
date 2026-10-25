# FIT-only six-dimensional covariance and predictive-region assessment

**Source:** successful [GitHub Actions run 40214131307](https://github.com/kys928/TriQTO/actions/runs/40214131307), original cohort SHA256-verified, 120 FIT families, first root/candidate/diagnostic, 16 acquisition draws at N/2N/4N. Program: `scripts/cohort/check_covariance_coverage_fit.py`. Protocol committed before this run: [FIT_CROSSFIT_COVARIANCE_COVERAGE_PROTOCOL.md](../../experiments/FIT_CROSSFIT_COVARIANCE_COVERAGE_PROTOCOL.md). Run artifact `fit-joint-covariance-coverage` (ID 12607101522, retention limited).

## Scientific objective

The preceding [family-crossfit covariance report](FIT_CROSSFIT_COVARIANCE_REPORT.md) found corrected coordinate and matched **trace** ratios close to 1 across five held-out FIT folds. Trace-only correction can hide wrong covariance directions or confidence coverage. We therefore tested a fixed Gaussian/chi-square *diagnostic*, not a certified Gaussian generative model.

For reference calibration draws 0,7,15, compare each 6D feature vector to the average of the **other fifteen independent acquisitions**, with estimated difference covariance `(1+1/15) Σ`. Evaluate 95% chi-square ellipsoidal containment for the 3D coordinate block, 3D matched block and joint 6D vector. All five folds train the same two global block multipliers exclusively on other family folds. Compare 6x6 matrix Frobenius error to the empirical 15-other-draw covariance. No TEST, SELECTION, truth, ideal simulator probabilities or frozen candidate-network scores were loaded.

## Descriptive held-out FIT results

| Shots | Method | 3D coordinate region coverage | 3D matched region coverage | 6D joint region coverage | Median relative Frobenius error |
|---|---|---:|---:|---:|---:|
| N=4096 | Original observed bootstrap | 82.50% | 85.00% | 80.00% | 0.6141 |
| N=4096 | Family-crossfit covariance | **97.22%** | 91.67% | **91.94%** | **0.5595** |
| 2N=8192 | Original observed bootstrap | 83.89% | 88.06% | 83.89% | 0.6413 |
| 2N=8192 | Family-crossfit covariance | **98.06%** | 95.28% | **90.56%** | **0.5066** |
| 4N=16384 | Original observed bootstrap | 85.56% | 92.22% | 82.22% | 0.6345 |
| 4N=16384 | Family-crossfit covariance | **99.44%** | 95.28% | **90.56%** | **0.4791** |

All percentages are within this finite simulator cohort and computed over 120 families with three preselected calibration references per family; reference draws within a family are dependent. The 95% target is merely the nominal Gaussian benchmark. The run stores additional family-cluster descriptive intervals and counts; it is not a formal population-hypothesis test.

## Interpretation

Block scaling produces a substantial numerical improvement but does **not** fully calibrate the joint 6D geometry. The coordinate block becomes relatively conservative under Gaussian ellipsoids (97–99%), while full joint coverage remains below the nominal 95% (about 90–92%). This is consistent with remaining anisotropy, inter-block covariance/orientation error, nonlinear non-Gaussian behavior, or finite 15-draw reference noise; these possibilities are not uniquely identifiable here.

**Verdict: `FIT_CROSSFIT_PROMISING_NOT_FULL_JOINT_COVARIANCE_CALIBRATED`.** A global 2-scale correction is a useful exploratory baseline and may improve statistical stability without extra shots; it is **not validated as a universal same-shot joint reliability estimate** for R2. We must not turn its near-1 trace ratios into confirmation. A separate, identity-certified independent development evaluation is still required before selecting a method. All network weights and original source data remain frozen.
