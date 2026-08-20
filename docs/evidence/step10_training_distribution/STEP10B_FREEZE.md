# Step 10B completed-outcome freeze

Status: **FROZEN — COMPLETED BENCHMARK AND POSTHOC AUDIT**

This file is the immutable transition marker between Step 10B and any later development stage. The benchmark outcome and forensic audit below were observed before Step 10C was designed.

- Benchmark: `benchmark_0971e5bb2ec77c2a1bb550d8`
- Official decision: `NO_INITIALIZATION_PASSES_FULL_DUAL_DOMAIN_GATE`
- Audit bundle SHA256: `26aab8b957f3d06d583b347b4095a381c3efbae33d39af9ac0fe52359054d640`
- Step-10 training-mixture product: `product_0f7112597501f7ea5fbe123b`
- Step-9A deployment bundle: `deploy_ac536a74b2f8dd571d353a12`
- Architecture: `late_concat`, 453,829 trainable parameters
- QPU executed: no
- Architecture changed: no
- Optimizer state resumed: no
- Step-10B outer validation used for selection: no

## Frozen gate outcome

Warm-start bridge mechanism BA was `0.7796768617`, 95% CI lower `0.7612315164`, minimum class recall `0.7425191371`; original mechanism/effect BA drops versus untouched Step-9A were `-0.0045710907` / `-0.0025309770`. Scratch bridge mechanism BA was `0.7740111150`, CI lower `0.7557037696`, minimum class recall `0.7432150313`; original mechanism/effect BA drops were `-0.0032765330` / `0.0051324175`.

Both initializations passed the CI-lower, minimum-recall, original-retention, and selected-checkpoint-eligibility subgates. Both failed only the predeclared bridge mechanism BA >= 0.80 gate. The primary paired bootstrap comparison, scratch minus warm-start bridge mechanism BA, was `-0.0056906671` with 95% CI `[-0.0124700217, 0.0019181343]`; therefore no statistically clear primary-metric superiority was established.

## Frozen selected checkpoints

Warm-start selected epochs: seed 1701 = 19, seed 1702 = 17, seed 1703 = 18. Scratch selected epochs: seed 1701 = 20, seed 1702 = 20, seed 1703 = 20. The Step-10B runner already selected the best retention-eligible checkpoint rather than blindly using the last epoch; however, those best states lived in process memory until final writeout.

## Consequence for later stages

The Step-10B original and bridge outer-development cohorts are now **spent for any follow-up motivated by this outcome**. They may remain archived for historical comparison, but they may not be presented as untouched Step-10C outer validation. Any Step-10C design must be frozen separately and must use a new untouched outer cohort.

The detailed immutable forensic record is `STEP10B_ANALYSIS.md` and `STEP10B_RESULT_SUMMARY.json`. No later result may rewrite this Step-10B decision.