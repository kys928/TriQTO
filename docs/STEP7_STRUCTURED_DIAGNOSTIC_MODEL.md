# Step 7 structured diagnostic model

Status: **DEVELOPMENT BENCHMARK COMPLETE**

Step 7 tested whether explicit graph-conditioned interpretation of signed finite-shot diagnostic evidence improves mechanism diagnosis beyond matched neural controls.

The full result is archived at:

`docs/evidence/step7_full_development_benchmark/RESULT_SUMMARY.md`

## Result

The predeclared architecture gate failed:

- `late_concat` mechanism BA: `0.5118`
- `structured_interaction` mechanism BA: `0.5052`
- structured minus late-concat paired 95% CI: `[-0.0120, -0.00165]`

The bespoke node/pair structured interaction therefore did **not** earn its additional architectural specificity on the current development cohort.

However, Step 7 strongly supports learned joint graph + diagnostic modeling in general: both neural joint models substantially outperform the frozen Step 6 cheap mechanism baselines, while the learned effect pathway substantially outperforms the frozen Step 6 SNR threshold.

## What remains valid

- `DiagnosticTensorBatch` / `DiagnosticEncoder` remains the correct semantic boundary for signed relational `B_delta` evidence.
- Signed `B_delta` must not be routed through the Born-probability encoder.
- The magnitude + shot-aware effect pathway remains supported.
- `late_concat` is the strongest tested Step-7 deployable mechanism architecture.
- Graph-only is weak; diagnostic evidence carries the dominant mechanism signal.
- The one-seed parity ablation supports retaining global parity.
- The current pairwise gated interaction is not supported as beneficial.

## Scientific boundary

Step 7 is development evidence, not confirmation. The existing outer-development validation cohort has prior exposure from Step 6 and cannot become a clean confirmatory cohort after the fact.

Any architecture revision must be frozen against `late_concat` before outcome. A genuinely new untouched confirmatory cohort remains required after final architecture freeze.
