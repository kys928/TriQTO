# Step 7 full development benchmark

Status: **COMPLETE — DEVELOPMENT EVIDENCE, NOT CONFIRMATORY**

The full frozen Step 7 benchmark completed as `benchmark_9989fbf9ef9feeaf283fe23f` on the accepted Step 5 v3 development cohort.

Authoritative result summary: `docs/evidence/step7_full_development_benchmark/RESULT_SUMMARY.md`.

## Primary result

The predeclared architecture-specific gate did **not** pass.

- `late_concat` mechanism BA: `0.5118`
- `structured_interaction` mechanism BA: `0.5052`
- structured minus late-concat paired BA difference: `-0.0067`
- 95% CI: `[-0.0120, -0.00165]`

Therefore the bespoke structured graph-diagnostic interaction did not earn its complexity over the matched late-concatenation neural control on this development cohort.

This is not a failure of the broader graph+diagnostic neural approach. Both neural joint models substantially outperform the frozen Step 6 cheap mechanism baselines, and the learned effect pathway substantially outperforms the frozen Step 6 SNR threshold.

## Development interpretation

- keep the dedicated signed diagnostic stream;
- keep the magnitude + shot-aware effect pathway;
- treat `late_concat` as the strongest tested Step 7 deployable mechanism architecture;
- do not promote the current `structured_interaction` to confirmation;
- current pairwise gated interaction is not supported by the one-seed ablation;
- global parity is supported as useful in the one-seed structured ablation;
- future architecture work must be frozen against `late_concat` before outcome;
- a new untouched confirmatory cohort remains required after final architecture freeze.

## Frozen execution record

- benchmark: `benchmark_9989fbf9ef9feeaf283fe23f`
- fit / internal selection / outer development roots: `3000 / 1000 / 1000`
- model runs: `15`
- outer validation used for selection: **NO**
- historical v0.1 test accessed: **NO**
- spent confirmatory cohort accessed: **NO**
- new confirmatory cohort accessed: **NO**

## Scientific boundary after completion

Step 7 remains development architecture selection. The current outer development validation cohort has prior development exposure from Step 6 and cannot be relabeled as a confirmatory test.

The next development decision may simplify/revise the architecture using only the predeclared Step-7 evidence, but any revised candidate must be frozen against `late_concat` before outcome. A new untouched confirmatory cohort must remain untouched until the final candidate is frozen.

Step 8 is **not automatically unlocked**.
