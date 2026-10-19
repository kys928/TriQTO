# Frozen calibration resampling: verified result

Portable replay is repaired and the preregistered 96-draw pilot completed.
Recorded-count replay recovers every frozen prediction, while fresh calibration
realizations produce substantial decision variation on this selected cohort.
Increasing calibration shots reduces some variation but does not eliminate it.
No ground-truth accuracy was measured.

## Replay and execution verification

The archive's authenticated selected rows now supply the independent post-hoc
extractor; the omitted product CSV is no longer a portable dependency. The
original on-volume CSV path is preserved. No archive, weights, calibration
settings, feature definitions, or pooling settings were changed by this repair.

The original release's five bundles and 36 x-only inputs were independently
verified after download. Portable replay needs neither simulation nor sampling:
all 33 donor Jacobian hashes match, both extraction paths agree exactly, all 36
frozen labels return, and maximum absolute logit error is 2.3182305177726903e-6.
27 focused tests and 704 full repository CI tests passed. `REPLAY_REPAIR.md`
records the missing-manifest failure and its regression coverage.

The corrected workflow ran on Python 3.11.17, NumPy 2.3.2, PyTorch 2.8.0+cpu.
All 32 draws at each of N/2N/4N finished, and the result release was independently
downloaded and hashed in the workflow and again outside it. All 101 result
file hashes match. A separate read of all 96 NPZ draws verified count budgets,
nonnegative integer counts, pooled-logit finiteness, and the reported label-change
and pairwise-disagreement metrics.

## First historical sampler discontinuity

For root 40025, call ordinal 0, the native 5090/4090 probability arrays reproduce
the original counts in a single process. At category 2 the conditional-binomial
loop has the same RNG state and 1,314 remaining shots, but conditional p changes
from 0.5 to 0.5000000000000002. NumPy's p <= 0.5 direct/complement branch changes
and counts `[631,683]` become `[683,631]`; the two earlier categories match.
A same-state binary control at adjacent floating-point values reproduces the
swap. `SAMPLER_REPRODUCER.json` preserves the call, exact hex values, counts, and
RNG-state/fixture hashes. This identifies the first differing call's mechanism;
it does not assert that every differing call takes this same branch or that
NumPy samples an incorrect distribution.

## Fresh-calibration stability

The observed x-only evidence, captured probability arrays, model ensemble,
normalization, probe strength, and pooling stayed fixed. Only calibration seed
schedules and the preregistered shot multiplier changed. N is 4,096 shots per
basis per sign; observed-evidence weights stayed fixed at all calibration budgets.

| Calibration shots | Label differences vs recorded realization | Pairwise disagreement between distinct fresh draws | Mean softmax-score standard deviation |
| --- | ---: | ---: | ---: |
| N = 4,096 | 429/1,152 (37.24%) | 26.00% | 0.1159 |
| 2N = 8,192 | 430/1,152 (37.33%) | 23.48% | 0.0882 |
| 4N = 16,384 | 393/1,152 (34.11%) | 20.46% | 0.0755 |

Each denominator repeats the same 36 examples across 32 realizations. There
are still only 33 circuit roots, selected for replay diagnosis. Pairwise means
compare two distinct draws for the same example and then average examples;
these comparisons are dependent. Softmax scores are not certified uncertainty.
These figures are stability measurements, not accuracy or population estimates.

The root-clustered descriptive 95% intervals for differences vs the recorded
realization are 23.84–49.31%, 23.43–49.68%, and 19.91–46.88%, respectively. They
retain all examples and draws of each root. Selection and possible family-level
dependence limit interpretation. No preregistered formal support gate or test of
monotonic improvement was defined, so no such verdict is manufactured here.

The noiseless finite-difference reference differs from the recorded realization
on 16/36 labels. Fresh-draw disagreement against that reference is 50.87%, 50.26%,
and 47.57%. This reference uses privileged captured probabilities at the same
finite epsilon; it is neither an exact derivative nor a ground-truth classifier.

## Downstream amplification: a concrete candidate mechanism

In the analysis-only expected-statistic Jacobians, 273/2,265 candidate-axis
columns have raw norm <= 1e-12 (the existing normalization floor). These are
numerically null responses at the frozen probe strength. Sampling gives these
columns nonzero estimates. Across their weighted example-level counterparts,
median estimated norms fall from 11.23 at N to 7.96 at 2N and 5.63 at 4N, roughly
the expected sampling-noise reduction.

However, the frozen pipeline divides each weighted column by its norm before
regularized inversion. For a null response estimated as eta/sqrt(N), while its
noise exceeds the floor, normalization yields eta/||eta||. Smaller noise amplitude
does not by itself remove its arbitrary direction. This is a structural route
for noise amplification, rather than evidence that Born probabilities changed
meaningfully across hosts.

The retained measurements are consistent with this concern: median normalized-
frame change vs the recorded realization is 1.21, 1.11, and 1.04; standardized
feature-change RMS is 0.80, 0.84, and 0.93. Distance from a single recorded
realization is not an accuracy metric and need not decrease with shots. These
observations and the algebra do not establish that null-column normalization
alone causes every label change; a selective intervention is still needed.

## Scientific next step

Keep the models frozen. Preregister an analysis-only selective-column
intervention on the retained draws to test whether normalizing near-null
responses drives the instability. Such an intervention can use the noiseless
reference as an explicitly privileged diagnostic control; it must not be
presented as a deployable estimator. Choose uncertainty-aware shrinkage, masks,
ensembles, or other remedies on separate development data, then evaluate on an
untouched locked cohort. Do not tune a threshold on these selected examples or
round probabilities to recover a preferred historical seed.

No sealed truth, Step-15 training, full holdout rescoring, or QPU execution took
place. CPU-only resampling provisioned no Runpod pods.

[Successful corrected workflow](https://github.com/kys928/TriQTO/actions/runs/39287947712)
[Durable pilot evidence](https://github.com/kys928/TriQTO/releases/tag/calibration-pilot-20261019)
[Durable source bundles](https://github.com/kys928/TriQTO/releases/tag/calibration-evidence-20261019)
