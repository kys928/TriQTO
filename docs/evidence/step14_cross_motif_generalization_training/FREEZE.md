# Step 14 pre-data freeze

Status: **FROZEN BEFORE ANY STEP-14 DATASET GENERATION OR TRAINING**

This file records the scientific boundary for the Step-14 cross-motif mechanism-generalization intervention.

## Frozen rationale

Step 13 showed that the Step-12 mechanism-classification failure is not primarily explained by disappearance of useful physical diagnostic information. Exact ideal diagnostic replacement also failed to rescue the frozen Step-10C model: Step-10C remained 8/18 overall and `cz_echo_ramsey` remained 0/6. Therefore the next intervention tests broader circuit-context training coverage while holding the existing architecture and deployable input contract fixed.

## Frozen intervention

Step 14 is exactly one data/generalization intervention:

- warm-start the three frozen Step-10C checkpoints;
- retain the legacy original and Step-10 bridge training domains;
- add the predeclared cross-motif family-grammar training domain;
- use held-out family-level selection and a separately materialized untouched simulator outer;
- evaluate against frozen Step-10C on identical new outer cohorts;
- preserve the architecture, graph input contract, loss weights, batch size, gradient clipping, and mechanism classes.

No alternative architecture, scratch initialization, auxiliary invariance loss, domain-weight sweep, LR sweep, or QPU adaptation is part of Step 14.

## Data boundaries

Before this freeze:

- no Step-14 fit family has been generated;
- no Step-14 selection family has been generated;
- no Step-14 simulator outer family has been generated;
- no Step-14 future-hardware-reserve family has been materialized;
- no Step-14 training has run;
- no Step-14 model selection has occurred.

The exact Step-11 and Step-12 QPU counts are forbidden from Step-14 training. The three exact Step-12 circuit signatures and the exact Step-11 phase-interference signature are excluded from the new generator. Step-12 case-level results may motivate the scientific question but may not select examples, checkpoints, thresholds, or hyperparameters.

The fold-6 future hardware reserve is especially protected: it may not be materialized, inspected, summarized, embedded, or scored during Step 14.

## Outer boundary

The fold-5 cross-motif simulator outer and the fresh legacy-retention outer may be generated only after:

1. all three per-seed Step-14 checkpoints are selected by the frozen development-selection rule;
2. the ensemble effect threshold is selected by the frozen development-selection rule; and
3. `selection_freeze.json` records those identities and hashes.

Outer evaluation is one-way. Outer results select nothing and cannot change the effect threshold, epoch, model seed, training schedule, dataset composition, or support gate.

## Frozen gate

A Step-14 simulator repair is supported only if all criteria in `configs/v0_2/step14_cross_motif_generalization_training.json` pass. In particular, the new cross-motif outer must meet the absolute BA/CI/recall/effect requirements, Step-14 must improve mechanism BA by at least 0.05 over frozen Step-10C with a paired-bootstrap lower bound above zero, and both legacy domains must remain within the 0.02 retention tolerances.

No criterion may be weakened after outer access.

## Allowed engineering work after this freeze

Implementation may add the generator, trainer, outer generator, one-shot evaluator, atomic persistence, logging, and runtime-compatibility fixes. Such changes must preserve the exact scientific contract: generated population, partition rule, model inputs, warm-start identities, optimizer settings, domain schedule, selection order, threshold rule, outer gate, and QPU boundary.

If an implementation defect makes the frozen protocol impossible to execute as written, the defect must be documented and a visible protocol amendment must be merged **before** generating affected data or training under altered semantics.

## Next-stage boundary

If Step 14 passes, a separate Step-15 hardware protocol may be frozen around the untouched future-hardware reserve. Passing Step 14 alone does not authorize a hardware-generalization claim.

If Step 14 fails, no Step-15 QPU execution should be used to search for a rescue. The next experiment must decompose graph encoder, diagnostic encoder, fusion, and mechanism head behavior on fixed simulator data before architecture changes are considered.
