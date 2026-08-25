# Step 13 post-replay freeze

Status: **FROZEN AFTER IDEAL-DIAGNOSTIC COUNTERFACTUAL REPLAY**

This document freezes the interpretation of the Step-13 counterfactual replay before any new training intervention.

## Immutable observed outcome

- Step10C hardware replay: 8/18 mechanism classifications correct.
- Step10C exact-ideal diagnostics: 8/18.
- Step10C ideal-local-only: 8/18.
- Step10C ideal-pair/parity-only: 8/18.
- Hardware prediction reproduction: PASS.
- `cz_echo_ramsey`: 0/6 under both hardware and exact-ideal diagnostics.
- Diagnostic conclusion: `IDEAL_DIAGNOSTICS_DO_NOT_RESCUE_MODEL__TRAINING_OR_GRAPH_CONTEXT_GENERALIZATION_PRIMARY`.

## Frozen interpretation

The primary Step-12 mechanism failure is not attributed to finite-shot or physical-QPU evidence collapse. The evidence supports a learned out-of-distribution mapping/generalization failure as the primary failure class. Graph-conditioned circuit-context generalization is a leading candidate because error structure is strongly motif dependent, but the exact neural subcomponent remains unresolved.

## No-retroactive-change declaration

Before this freeze:

- no model weights were changed;
- no new training was run;
- no threshold was changed;
- no checkpoint was switched;
- no architecture was changed;
- no Step-12 QPU case was rerun;
- no Step-12 pass/fail rule was changed;
- no Step-12 outcome was reclassified;
- no posthoc physics-template oracle was promoted into a deployable model or confirmatory metric.

## Next-stage boundary

A new training/generalization stage is now scientifically justified. It must have a new experiment identity and frozen data/split/evaluation contracts before model selection.

The Step-12 hardware cohort is spent as development/diagnostic evidence. It may inform the class of failure and motivate broad distributional coverage, but exact Step-12 cases must not be reused as a future blind validation cohort.

Warm-starting the frozen Step-10C checkpoint is the default unless a separately frozen controlled comparison demonstrates that discarding it is preferable.

A future claim of repaired hardware generalization requires a new independently frozen hardware cohort after training; success on simulator/development data alone is insufficient.
