# FIT-only covariance correlation-shrinkage experiment

**Date:** 2026-10-25. Study predeclared in [FIT_COVARIANCE_CORRELATION_SHRINKAGE_PROTOCOL.md](../../experiments/FIT_COVARIANCE_CORRELATION_SHRINKAGE_PROTOCOL.md) before this run. Successful FIT-only [GitHub Actions run 40214461272](https://github.com/kys928/TriQTO/actions/runs/40214461272); artifact `fit-shrinkage-coverage-and-pseudo-score` (ID 12606842960, retention-limited). **No SELECTION, TEST, reserve, ideal simulator probabilities, candidate neural model scores, or labels accessed.**

## Hypothesis and tests

The previous FIT crossfit global two-block covariance estimator achieved near-unity marginal empirical/estimated variance trace ratios, but only ~90–92% nominal 95% **joint** Gaussian ellipsoid coverage. As an exploratory source of residual error, we tested covariance correlation shrinkage with `Sigma_beta=(1-beta)*Sigma+beta*diag(Sigma)`, keeping each diagonal variance and preserving PSD.

For each of five held-out FIT-family folds, choose beta from fixed `{0,0.25,0.5,0.75,1.0}` by mean per-family Gaussian pseudo-NLL on the other four training-family folds only. Evaluate the selected beta on held-out families. Original single-calibration observed-count estimator, paired 16-acquisition target and two trained coordinate/matched scale factors are unchanged. A fixed diagonal ridge prevents singular covariance inversions. This evaluates *internal* FIT folds, not independent confirmation.

## Outcome

| Shots | Five selected beta values | Mean held-out pseudo-NLL (beta=0 vs selected) | Joint 6D nominal-95% coverage (beta=0 vs selected) |
|---|---|---|---|
| N | 0,0,0,0,0 | 0.7414 vs 0.7414 | 91.94% vs 91.94% |
| 2N | 0,0,0,0,0 | 0.0227 vs 0.0227 | 90.56% vs 90.56% |
| 4N | 0,0,0,0,0 | -0.8080 vs -0.8080 | 90.56% vs 90.56% |

All 15 fold/budget decisions retained **beta=0 (no off-diagonal shrinkage)** on their training pseudolikelihood. There is no evidence under this prespecified score that arbitrarily deleting coordinate/matched correlations improves joint calibration. The remaining joint coverage discrepancy is not fixed by this simple global correlation shrinkage.

**Scientific verdict:** `NO_CORRELATION_SHRINKAGE_BENEFIT_ON_FIT_PSEUDO_NLL`. The two-block calibrated covariance remains exploratory but promising. Its full 6x6 predictive distribution remains imperfect; it must not be declared a universally reliable R2 estimator, and this negative result must not be erased by post-hoc tuning to nominal coverage. Later work may evaluate physically informed source inference or nonlinear distributions under an independently attested cohort.

No R0/R1/R2 remediation settings were selected. Original TEST remains compromised, protected source attestation remains absent, and no new TEST measurements were acquired.
