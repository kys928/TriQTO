# Prospective FIT-only nonlinear covariance source-center diagnostic — added after M0/M1 results

**Scope:** exploratory mechanism isolation only, no R0/R1/R2 scoring and no remedy selection. The original FIT-only M0/M1 analysis found persistent coordinate underdispersion; posterior predictive partially helped but overdispersed matched responses at 2N/4N.

## Source-centering hypothesis

A single noisy empirical frequency table can be an unstable plug-in center for the nonlinear normalized local-frame-to-coordinate map. Compare pooled frequency estimates from K independent observed calibration realizations, at K={1,2,4,8}. For **each** K pool exactly the first K archived calibration draws, estimate their per-call multinomial probabilities, simulate 512 independent *N-shot* feature realizations, and compare the 6D covariance estimate with an unchanged evaluation target from the **last eight independent archived draws** (indices 8–15). Use one first FIT family root, first candidate, first observed diagnostic bundle; all 120 FIT families × N,2N,4N. No source probability oracle, no labels, no model predictions, no SELECTION or TEST.

### Interpretation caveat / cost

The calibration table used to estimate p at K=8 costs **eight times** the per-candidate shots of K=1, although the synthetic replicated future feature measurement uses N shots. This experiment diagnoses center-estimation error rather than demonstrating a budget-matched deployable estimator. Keep the evaluation draws identical for each K; do not pool any test/holdout count table into the estimator. If estimates approach empirical covariance as K increases, that supports noisy source-centering as a contributor; it does not establish a physically affordable remedy.

Report median empirical/estimated coordinate and matched covariance trace ratios and median absolute log-ratio, with the eight-draw reference sampling variance limitation. Do not select a K for R2 or claim validation outside FIT. This design and K grid are fixed before the script is run.
