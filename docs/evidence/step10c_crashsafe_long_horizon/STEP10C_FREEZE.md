# Step 10C result freeze

This file freezes the completed Step-10C outcome before any follow-up optimization study.

- Benchmark: `benchmark_f9478da45d68795655259054`
- Official decision: `NO_INITIALIZATION_PASSES_FULL_DUAL_DOMAIN_GATE`
- Audit bundle SHA256: `efbb016d7c4f6c0908217ce5cca8ba89e23fe785fa71e247dbbe843b09955ba4`
- Fresh outer product: `product_57ee407d62ea794bfc9ff169`
- Fresh outer used for selection: **NO**
- Fresh outer materialized only after all six selections: **YES**
- Step-10B outer evaluated: **NO**
- QPU executed: **NO**
- Architecture: `late_concat`, 453,829 trainable parameters, unchanged
- Any crash-recovery resume used: **NO**

Frozen primary results:

- warm-start bridge mechanism BA `0.7939042908232706`, CI lower `0.7809313429211779`, minimum recall `0.7493055555555556`, full gate **FAIL**;
- scratch bridge mechanism BA `0.7868371829119972`, CI lower `0.7735037558747904`, minimum recall `0.7645833333333333`, full gate **FAIL**.

The BA >= 0.80 requirement is not rounded, relaxed, or reinterpreted.

Both initializations pass original-domain retention. The paired primary scratch-minus-warm bridge mechanism BA difference is `-0.007154791047562522`, CI `[-0.01498014147048735, 0.0004084862699042057]`, so neither initialization has statistically clear primary-metric superiority.

The 40-epoch ceiling is not frozen as the next intervention: five of six trajectories stopped early under the unchanged patience rule, and the sixth selected epoch 38 before the ceiling. Step 10C therefore closes the max-epoch-only question for this stage.

New telemetry shows 100% of optimizer steps were clipped at norm 1.0 in all six runs. This is frozen as a post-hoc optimization diagnostic, not as evidence that any specific new clip threshold will succeed.

The Step-10C fresh outer cohort is now spent. No follow-up may tune on it or present it as an untouched outer cohort.
