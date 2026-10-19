# Calibration count substitution: controlled replay result

The 4090 replay's 16 prediction disagreements were removed by substituting the
verified 5090 integer calibration counts. On the unchanged 5090 host, selecting
OpenBLAS Haswell dispatch reproduced the independent 4090 calibration trace
exactly across all 33 circuit roots. This identifies OpenBLAS CPU dispatch as a
sufficient trigger for the observed calibration divergence in this experiment.

[Experiment workflow and artifacts](https://github.com/kys928/TriQTO/actions/runs/39267478899)

## Fixed experimental conditions

The existing targeted cohort contains 36 examples across 33 roots and 13,590
calibration sampling calls per extraction path. Both independent extraction
paths were run for every profile. They agreed exactly within every run.

Both hosts requested the same digest-pinned audit image, overlaid on the
previously tested numerical image. The 5090 donor was required to reproduce
both the previous verified count/Jacobian hashes and all frozen predictions
before the recipient could launch. Frozen checkpoints, configurations, seeds,
cohort files, and measurement provenance were checked. No fitting, sealed-truth
access, hyperparameter selection, full spent-holdout rescoring, or QPU execution
occurred. The Step-15 adapter freeze passed verification before and after each
host's experiment.

## Prediction replay

All disagreements below are against frozen Phase-A predictions, not ground-truth
classification errors. Logit errors are maximum absolute CPU inference errors.

| Host / intervention | OpenBLAS dispatch | Threads | Disagreements / 36 | Max logit error |
| --- | --- | ---: | ---: | ---: |
| 5090 + EPYC 9554, native | SkylakeX | 64 | 0 | 2.31823e-6 |
| Same host, single thread | SkylakeX | 1 | 0 | 2.31823e-6 |
| Same host, Haswell dispatch | Haswell | 1 | 16 | 2.46949 |
| Same host, NumPy AVX-512 disabled | SkylakeX | 1 | 0 | 2.31823e-6 |
| 4090 + EPYC 7763, native | Haswell | 64 | 16 | 2.46949 |
| Same 4090, donor counts substituted | Haswell | 64 | 0 | 2.31823e-6 |

GPU inference also recovered all predictions after substitution; its maximum
absolute logit error was 2.95398e-6. The recipient retained its own simulated
probabilities and used donor integer counts without calling the sampler in the
substitution arm. All 33 reconstructed finite-shot Jacobians matched the donor
exactly, and every donor count hash matched.

## Numeric probability comparison and CPU isolation

The largest absolute probability difference between native 5090 and native
4090 was **4.440892098500626e-16**. The largest per-call total variation distance
was also **4.440892098500626e-16**. Sample counts differed in 16 of 33 roots.

The within-host Haswell ablation had these same maxima and count-root changes.
More strongly, its complete per-root probability, seed, count, statistic, and
Jacobian trace records were identical to the independently captured 4090
records across all 33 roots. The single-thread and NumPy feature-disable arms
had zero probability differences and zero changed count roots relative to the
native donor. Runtime snapshots confirmed the requested OpenBLAS dispatch and
thread settings.

These are controlled software-dispatch and evidence-substitution interventions,
so the diagnosis is stronger than a correlation between GPU models. The
experiment does not identify the individual BLAS call/instruction responsible
or trace NumPy's internal multinomial branch decisions. It also does not prove
robustness to fresh finite-shot calibration draws or generalize beyond this
fixed cohort. Persisting counts solves replay of this evidence realization;
independent robustness evaluation remains a separate scientific question.

## Portable evidence and verification

The manifest-addressed donor bundle persists probabilities and integer counts,
with circuit operation/wire/parameter identity, measurement ordering,
calibration settings, candidate/sign/basis/call ordering, seeds, shot budgets,
schema, cohort-source hashes, and checkpoint provenance. Both extraction paths
must consume the complete verified bundle. Manifest/archive integrity and shot
totals are checked before replay; mismatches fail closed.

The exact donor directory and manifest hash, individual diagnostic hashes,
worker result hashes, per-profile environments, and comparison maxima are
recorded in [RESULT_SUMMARY.json](RESULT_SUMMARY.json). The full result artifact
ZIP was downloaded and independently matched its GitHub SHA256 digest:
`68bc29f19deea04b5b5d81797a67aa18759c325c8af0821230adaeeec5b93315`.
Both extracted worker JSON files also matched the terminal status hashes read
separately from the network volume. Raw bundles remain on the persistent
network volume; full JSON results are also preserved in Actions artifacts.

Validation: 32 local focused tests and 691 full repository CI tests passed.
The workflow completed successfully. A separate read-only check confirmed both
experiment pods absent:
[cleanup verification](https://github.com/kys928/TriQTO/actions/runs/39269809592).
