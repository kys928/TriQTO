# Calibration resampling pilot — protocol frozen before execution

This is a diagnostic intervention on the already-spent, selected 36-example,
33-root replay cohort. It is not a new accuracy evaluation. No sealed labels,
true locations, full holdout scoring, fitting, checkpoint edits, or tuning.

## Fixed evidence and interventions

Use the verified native 5090 probability bundle from counts-substitution-20261016-r1.
Hold its probability arrays, circuit/candidate/measurement order, observed x-only
inputs, checkpoint ensemble, training normalization, pooling, tau, temperature,
probe strength, and all preprocessing constant. First reconstruct recorded-count
Jacobians without simulation or sampling and require exact donor Jacobian hashes,
zero frozen label disagreements, and maximum CPU logit error <= 1e-4. This is an
execution gate, not a stability threshold.

Sample 32 realizations using explicit NumPy PCG64 and per-call SeedSequence input
[20261019, realization_index, root_index, call_ordinal], realization indices 0..31.
Use the same seed schedule separately at shot multipliers 1, 2, 4; this couples
comparisons but does not make different multinomial draws nested shot sequences.
Draws are independent across call keys and realization indices. Preserve the
recorded per-call shot budgets times the multiplier. Do not change the observed
measurement weights when changing the calibration shot budget. The 1x arm is the
primary pilot; 2x and 4x are preregistered descriptive follow-ups, run after 1x.
No choice of remedy or acceptance threshold is optimized on this cohort.

## References and measurements

Recorded counts are the replay reference. Expected statistics computed directly
from captured probabilities yield an analysis-only noiseless finite-difference
reference at the same epsilon, not an exact derivative or hardware-visible input.

For each realization and example, retain pooled logits, softmax probabilities
(uncalibrated scores, not certified uncertainty), top-two margin, mechanism label,
and winning candidate under the frozen ensemble's max-mechanism-logit score.
This candidate proxy is not an action policy or a localization accuracy metric.
For each root/candidate, retain Jacobian relative/Frobenius differences, raw and
shot-weighted column norms, normalized singular values and regularized system
condition. For each example/candidate retain canonical features, profiles, and
candidate scores; summarize changes before and after feature standardization.
No abstention head is invented if the frozen pipeline does not expose one.

Report per-example flip rates against both recorded-count and noiseless
references, within-example pairwise disagreement, probability dispersion, margin
changes, candidate-proxy changes, and descriptive shot-budget trends. Estimate
uncertainty by resampling circuit roots as clusters, preserving all examples and
all draws for a root (2,000 replicates; seed 2026101901). These intervals describe
this selected cohort only; selection and possible circuit-family dependence
prevent population claims. Report seed variation separately. Thirty-two draws
cannot certify rare-event stability. Repeated draws are not independent circuits.

## Sampler micro-reproducer

Select the first count-different call by numeric root order, then call ordinal,
from the original native 5090 and 4090 bundles. Reproduce both recorded draws in
one NumPy 2.3.2 process with each original seed and shot count; retain both exact
binary probability arrays and integer counts. Inspect sequential conditional
binomial probabilities and the NumPy 2.3.2 implementation. Any claimed internal
branch mechanism must be backed by source and a controlled intervention; otherwise
report it as unresolved. This investigation does not alter the sampler in the pilot.

## Decision boundary

The pilot diagnoses stability; it cannot establish correctness. If calibration
changes materially affect decisions, propose remedies for separate development
data and an untouched locked evaluation. If changes are small, still require an
untouched evaluation before generalization claims. Baselines must share observed
evidence and total shot budgets. No Step-15 retraining is authorized by this pilot.
