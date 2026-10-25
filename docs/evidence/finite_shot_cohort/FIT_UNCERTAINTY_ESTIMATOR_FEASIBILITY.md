# TriQTO — exploratory FIT-only nonlinear uncertainty estimator feasibility

**2026-10-25.** Cohort identity `sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d`. **This is a post-diagnostic, data-informed FIT-only engineering analysis**, not preregistered confirmation or a selected R2 estimator.

Original **hash-verified** archive; successful [GitHub Actions run 40207050186](https://github.com/kys928/TriQTO/actions/runs/40207050186), results artifact `stabilization-fit-feasibility-and-blind-candidate-identities` (ID 12603982880, expires). The run checked seven original synthetic attestation/covariance regressions, processed 120 FIT families and exported a *public identity-only* candidate manifest intended for an authorized external custodian. No original TEST data arrays, truth, decryption key, SELECTION, sealed reserve or oracle simulator probabilities were read for this estimator comparison.

## Estimands and method contract

Per FIT family: first circuit root, first ordered plausible candidate, first diagnostic observation; original 16 independent calibration realizations at N=4096, 2N=8192 and 4N=16384; diagnostics fixed. The empirical 16-draw covariance of the jointly computed 3 coordinates and 3 matched responses is the descriptive *evaluation target*, not input to either estimator. Exactly 192 resamples were generated for each of 3 reference count realizations (indices 0, 7, 15), and their 6×6 covariance matrices were averaged. Both estimator families use **one reference draw at a time**, preserving the single-realization measurement acquisition contract:

- **M0**: multinomial resampling from observed normalized frequencies (existing plug-in bootstrap).
- **M1**: Jeffreys prior posterior predictive, (p^*\sim\mathrm{Dirichlet}(n+\frac12)), (n^*\sim\mathrm{Multinomial}(N,p^*)). This includes uncertainty about the unknown probabilities and *another* future shot draw. It does not claim frequentist unbiasedness for fixed-source acquisition covariance.

A ratio `empirical repeated-acquisition covariance trace / estimated covariance trace` near 1 is desirable; >1 denotes underdispersion, <1 overdispersion. Reported medians are per-family and no multiple resamples are treated as independent circuits.

## Results

| Budget | M0 coordinate ratio | M1 coordinate ratio | M0 matched ratio | M1 matched ratio |
|---|---:|---:|---:|---:|
| N | 1.945 | 1.600 | 1.368 | 0.857 |
| 2N | 2.172 | 2.122 | 1.302 | 0.770 |
| 4N | 2.290 | 2.001 | 1.232 | 0.718 |

The median per-family **absolute log calibration error** for coordinates declined with M1: 0.665→0.510 (N), 0.776→0.752 (2N), 0.840→0.693 (4N). For matched responses it was 0.328→0.317 (N), **worsened** 0.334→0.381 (2N) and 0.299→0.480 (4N). Thus M1 improves some coordinate trace calibration but does not solve coordinate underdispersion, and its matched-response covariance becomes increasingly overdispersed with larger shot budgets.

**Research verdict:** `EXPLORATORY_FIT_ONLY_NO_ESTIMATOR_VALIDATED`. Do not choose a calibrated R2 feature reliability weight from these results. No Step-15 inference, R0/R1/R2 comparison or test-blindness verdict was made by this run. The earlier [true-centered control](../evidence/finite_shot_cohort/ADVERSARIAL_POST_RELEASE_FINDINGS.md) remains descriptive mechanism evidence only.

## Scientific limitations / next validation

- The ratio uses covariance across **16**, not infinitely many, acquisitions; the denominator uses only 192 resamples per reference and can be noisy or distorted in nonlinear/singular regions.
- M1 is a **posterior predictive for future measurements with uncertain source probabilities**, which differs mathematically from conditional sampling covariance at a fixed physical source. Improving one trace does not establish correct uncertainty geometry or downstream decision stability.
- Trace alone ignores six-dimensional anisotropy and coordinate/matched covariance block directions. Test covariance eigenstructure, interval/ellipsoid coverage and empirical predictive validity; track observed-count estimators' linear Jacobian inflation.
- The source cohort remains quarantined for historical/reserve identity attestation, and its publicly released TEST has deterministic mechanism-label leakage. Nothing here lifts either gate.
- SELECTION was not read and is available only after estimator-selection acceptance and external-source identity attestation. Predefine numerical acceptance criteria and any candidate estimator modifications in a prospective amendment **before** evaluating SELECTION.

**Decision:** Retain M0 and M1 as explicit feasibility baselines; do not tune their hyperparameters on the 36 selected diagnostics, publicly released TEST or existing report ratios.
