# Finite-shot stabilization cohort — constructed, internally audited, quarantined

The existing Step-14 task was preserved. A new 240-family cohort was built with 960 roots, 12,480 observations, and 16 independent calibration realizations at each of N/2N/4N. The 120/60/60 family split and 192/48 representative/stress sampling counts are exact. TEST labels and generation recipes are encrypted. Step-15 was not evaluated or changed.

**Research verdict:** `INTERNALLY_VERIFIED_QUARANTINED_EXTERNAL_INDEPENDENCE_UNVERIFIED`. Construction and internal verification are complete. Full historical/reserve exclusion is not, so **do not select stabilization remedies on this corpus yet**.

- [Cohort protocol](COHORT_PROTOCOL.md): preregistered design, admission, acquisition, boundaries and prospective storage amendment.
- [Extensive EDA](EDA_REPORT.md): five figures, joint coverage, balance/confounding, geometry, count covariance, nonlinear joint uncertainty and scientific limitations.
- [Independent audit](INDEPENDENT_AUDIT.md): exhaustive/sampled checks, expanded identity coverage and remaining gate.
- [Supplemental audit](SUPPLEMENTAL_AUDIT.md): partition-specific structural tables, additional exclusion checks and independent diagnostic replay.
- [Acceptance record](COHORT_ACCEPTANCE.json), [immutable manifest](DATASET_MANIFEST.json), [shot/resource accounting](FINAL_ACQUISITION_ACCOUNTING.json), and [reproduction](REPRODUCTION.md).

## Principal observations

Analytic Jacobian covariance tracks repeated-shot variance (N median trace ratio 0.996), but nonlinear coordinate uncertainty needs work: median empirical/bootstrap variance ratios are 1.96 for coordinates and 1.34 for matched responses at N in the deterministic single-candidate audit sample. Coordinate/matched uncertainty is strongly coupled (median maximum absolute bootstrap correlation 0.805).

The representative sample is often geometrically difficult: 72.9% of representative development families meet the stress rule. More calibration shots reduce matched-response feature SD but do not monotonically stabilize coordinates. These are **feature/uncertainty findings, not model performance or remedy results**.

The original first-RY gate also exposes a proxy for the intended affected qubit. It was retained to preserve the agreed task and is documented as a limitation.

## Reproducible evidence

- Materialization identity: `sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d`.
- Statistical evidence identity: `sha256:9f889092113d9c65549d0191158b25ed0a845f2cce7769192d449d0564ccb6fa`.
- [Versioned raw archive and checksum](https://github.com/kys928/TriQTO/releases/tag/finite-shot-cohort-20261025-1448ca1ebbdf): 369,819,600 bytes; SHA256 `fbdf4f351771e5b1779173b1e085eadd249ecb72d059d0fb051a3f55b44adce9`.
- [Successful primary run](https://github.com/kys928/TriQTO/actions/runs/40188027634): 15 tests, 3,366 file checks, 24 independent calibration replays, 384 independent canonical reconstructions, full development recipe reconstruction and EDA.
- [Expanded identity inventory](https://github.com/kys928/TriQTO/actions/runs/40188502697), plus 156 additional independent diagnostic replays and 720 development truth-schedule checks in the supplemental audit.

All checked identities were disjoint. Inaccessible reserve identities, unenumerated historical products and missing early-product structural signatures remain the blocker. No sealed historical labels were accessed. No training, model inference, remedy tuning, QPU work or RunPod pod was used.
