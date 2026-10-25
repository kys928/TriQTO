# FIT-only cross-fitted covariance coverage / full-matrix audit

**Prospective auxiliary specification committed before this diagnostic runs; 2026-10-25.** The existing FIT five-fold experiment showed improved marginal covariance trace estimates, but that does not establish predictive uncertainty shape.

Reuse the exact family SHA256 five-fold assignment, 120 FIT families and single-observed-draw 192-resample 6x6 bootstrap inputs at N/2N/4N, first root/candidate/observation per family, predeclared source-reference indices 0,7,15. Train only two global coordinate/matched PSD-preserving covariance factors on other families (as in `FIT_FAMILY_CROSSFIT_COVARIANCE_PROTOCOL.md`), never from each held-out family's outcomes.

For a held-out reference, compute the six-dimensional difference between its extracted features and the average of the other 15 independent calibration draws. Under the benchmark idealization of independent same-source samples, its covariance is approximately `(1+1/15) × Σ`. Evaluate 3D-coordinate, 3D-matched and full 6D quadratic regions using 95% chi-squared cutoffs 7.8147 and 12.5916, respectively. Report **descriptive observed coverage** alongside nominal 95%; nonlinear distribution may be strongly non-Gaussian, so chi-square coverage is a diagnostic, not a guaranteed validity criterion.

Also compare Frobenius norm error of predicted 6x6 matrix vs the 15-other-draw covariance, normalized by the reference's Frobenius norm, and report numerical rank/singular flag rate and eigenvalue PSD errors. No outliers are dropped; near-singular matrices use a frozen pseudoinverse with relative singular-value cutoff 1e-10, and report such cases instead of pretending standard chi-square degrees of freedom apply. Include all N/2N/4N and family-block bootstrap uncertainty (2,000 families resamples) for scalar coverages.

All calculations strictly FIT, no oracle probabilities, SELECTION/TEST, labels, protected reserves, candidate-network scores or QPU calls. The source cohort historical/reserve identity gate remains unresolved. This is a post-hoc methodological investigation, not independent confirmation or R0/R1/R2 selection.
