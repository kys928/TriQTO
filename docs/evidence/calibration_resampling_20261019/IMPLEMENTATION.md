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

At the last execution check on 2026-10-19, GitHub had not assigned runners to the
archival or pilot jobs. The raw probability arrays are held behind the existing
Actions/S3 control plane and are unavailable locally until export succeeds.
Therefore no historical sampler-boundary conclusion or fresh-calibration stability
result has been established by this new work. The preceding count-substitution
experiment remains the latest completed empirical evidence.

- [Primary archive request](https://github.com/kys928/TriQTO/actions/runs/39280314165)
- [Alternate archival runner request](https://github.com/kys928/TriQTO/actions/runs/39281742338)
- [Corrected pilot request](https://github.com/kys928/TriQTO/actions/runs/39281334105)

The earlier queued pilot request predates the frozen 17-operation compatibility
bound fix; its outcome must not be used as the pilot result. The corrected request
above includes that fix. No GPU pod was launched for this work, no weights or
frozen predictions were changed, and no full holdout was rescored.
