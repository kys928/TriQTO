# Finite-shot calibration count substitution

This fixed-cohort diagnostic tests whether changing calibration evidence explains
the host-dependent Step-14 replay failures. It does not fit any model, inspect
sealed truth, select hyperparameters, or rescore the spent holdout.

Both hosts use one digest-pinned image overlay on the previously tested numerical
image `sha256:5fa24fff53342967a59b837405c9ebadc51ceeb5d5581a88e8cbc2c721d01425`.
Only diagnostic scripts and their tests are overlaid; numerical dependencies and
frozen model/configuration files are inherited unchanged.

1. Capture probabilities and integer counts on an RTX 5090 for the existing
   36-example, 33-root cohort, using both independent extraction paths. Require
   zero frozen-prediction disagreements and exact count/Jacobian hash parity with
   the previously independently verified 5090 diagnostic. Stop on failure.
2. On that same 5090 host run fresh subprocesses with single-thread numerical
   libraries, then OpenBLAS Haswell dispatch with the same thread setting, then
   NumPy AVX-512 dispatch disabled with the same thread setting. Record the
   actual runtime dispatch and compare probabilities/counts to the native run.
3. On an RTX 4090 capture native probabilities/counts, then run both extraction
   paths with the donor's exact integer counts substituted at empirical_stats.
   The recipient still simulates its own probabilities but does not sample them
   in the substitution arm. Reconstruct statistics and Jacobians locally and
   run the same frozen CPU/CUDA classifiers.
4. Verify donor-count hashes, compare Jacobian hashes, and report native and
   substituted prediction disagreements and logit errors. Numeric probability
   comparisons include maximum absolute differences, L1/total variation per
   call, and count changes per root. Verify the Step-15 freeze before and after.

Bundles consist of a canonical JSON manifest and a hash-bound, pickle-free NPZ
archive of float64 probabilities and int64 counts. Their directory is the SHA256
of the manifest. Provenance binds the clean circuit's ordered operations/wires,
global phase/parameters, ordered candidates, calibration settings, measurement
eigenvalues and pair ordering, basis/mechanism/sign/call ordering, seeds, shots,
cohort source files, method bundle, and checkpoint hashes. Readers require the
manifest hash, validate the archive and shot totals, reject missing/extra calls
or roots, and check complete coverage on both extraction paths.

Results are new immutable directories under
`/workspace/triqto-data/calibration_substitution/<experiment>/<role>/`. Old
replay pointers, predictions, weights, and calibration artifacts are untouched.
The dedicated workflow preserves verified JSON results as Actions artifacts.
One pod runs at a time; each host has a 60-minute execution budget and cleanup
runs on success/failure. Controller records also allow the existing supervisor
to clean terminal workers if the controller is interrupted.

Interpretation: recovery after a count-only intervention supports calibration
counts as a causal explanation on this cohort. It does not establish universal
model accuracy, hardware portability for all circuits, or an exclusive CPU
cause. CPU ablations are controlled within one host; inspect actual dispatch
records before attributing a change to a requested environment setting. These
ablations do not isolate every Qiskit, compiler, libm, or instruction-selection
path. Preserved source counts are observational experimental evidence, not a
replacement for independent fresh-shot robustness evaluation.
