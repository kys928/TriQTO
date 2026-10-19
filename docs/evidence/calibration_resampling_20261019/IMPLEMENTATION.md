# Portable replay and calibration resampling implementation

Implemented after the verified count-substitution experiment, under the protocol
in `docs/experiments/CALIBRATION_RESAMPLING_PILOT.md`. This is an implementation
record, not an empirical result report.

## Implemented

- Read-only fixed-experiment S3 exporter: original five probability/count bundles,
  their circuit/measurement/shot/schema provenance, selected 36 x-only payloads,
  frozen checkpoint ensemble, hash-verified source reports, and dependency inventory.
  No sealed truth, unselected payloads, pod provisioning, or S3 writes.
- Independent GitHub release archive with a fresh-download SHA256 verification.
  This is only complete after the archival workflow passes; workflow creation
  does not itself constitute archival success.
- Explicit CPU recorded-count replay entry point that never simulates or samples,
  verifies the complete call contract, requires exact donor Jacobian hashes,
  checks both extraction paths, and gates on frozen prediction recovery.
- Same-process original-seed multinomial reproducer selecting the first historical
  count-different call by numeric root and ordinal. It checks exact conditional
  binomial reconstruction and probes the p=0.5 direct/complement boundary.
- Preregistered 32 realizations at each of N, 2N, 4N, with fixed observed inputs,
  probe strength, checkpoints, normalization, and pooling. Every new count/Jacobian,
  feature, candidate logit, and pooled output is retained. Reports include root-
  clustered descriptive intervals and explicitly distinguish stability from accuracy.

## Validation and execution boundary

700 local repository tests passed (Python 3.12, NumPy 2.3.2, PyTorch 2.8.0+cpu).
The 54 warnings are from existing tensor/noise-model/multiprocessing tests.
A synthetic binary sampler test verifies the boundary probe; it does not establish
that this mechanism explains the captured historical case.

The initial queue-only status has been superseded by completed workflows. The
primary archive request succeeded and its durable release has been downloaded
and independently verified. The alternate archive request refused to overwrite
that existing release. The initial pilot stopped at the omitted product-manifest
CSV dependency; commit d43063b4da37a0180f0d7a40dd4cb8b8e2b32374 repairs that
portable API boundary using the authenticated selected rows.

The corrected workflow completed all 96 draws, verified durable publication,
and passed all 704 repository CI tests. Actual archived evidence passes the
replay gate (33 exact Jacobians, zero frozen-label disagreements). See
`REPLAY_REPAIR.md`, `ANALYSIS.md`, `PILOT_RESULT_SUMMARY.json`, and
`SAMPLER_REPRODUCER.json` for verified results and interpretation limits.

- [Successful primary archive](https://github.com/kys928/TriQTO/actions/runs/39280314165)
- [Successful corrected pilot](https://github.com/kys928/TriQTO/actions/runs/39287947712)
- [Durable pilot evidence](https://github.com/kys928/TriQTO/releases/tag/calibration-pilot-20261019)

No GPU pod was launched for this work, no weights or frozen predictions were
changed, and no full holdout was rescored.
