# Step 10D pre-outcome freeze

Status: **FROZEN BEFORE STEP-10D OUTCOME**

Step 10D is the final simulator-development intervention before IBM hardware.

## Frozen scientific intervention

Exactly one change is permitted to the Step-10C warm-start training trajectory:

- epochs 1–20: LR `3e-4`;
- epochs 21–40: LR `1e-4`.

No LR sweep, scheduler search, architecture change, dataset redesign, clip-threshold tuning, loss-weight tuning, max-epoch extension, scratch rerun, outer evaluation, or QPU execution is part of Step 10D.

## Frozen execution scope

- warm-start only;
- seeds `1701, 1702, 1703`;
- same Step-10 fit + selection partitions;
- same Step-9A warm-start checkpoints;
- same `late_concat` architecture with `453829` trainable parameters;
- same AdamW, weight decay, batch size, gradient clip, loss weights, domain schedule, early stopping, and checkpoint-selection rule;
- crash-safe best/resume persistence retained;
- Step-10C warm checkpoints may be re-evaluated only on the same development selection cohorts for the predeclared hardware-candidate comparison;
- spent Step-10C outer access is forbidden.

## Frozen hardware-candidate rule

Step 10D warm becomes the primary IBM-hardware candidate only if:

1. all three selected Step-10D checkpoints are retention-eligible;
2. the Step-10D ensemble passes the same `0.02` original-domain development retention tolerance;
3. Step-10D bridge-selection mechanism BA exceeds the re-evaluated Step-10C warm bridge-selection mechanism BA by more than `0.0005`.

Otherwise the frozen Step-10C warm ensemble remains primary.

The decision must be made before QPU execution and may not be overridden after seeing hardware results.

## Frozen hard stop

Regardless of Step-10D outcome, the next scientific stage is a separately frozen exploratory IBM-hardware transfer pilot.

No further simulator hyperparameter or architecture tuning is permitted before that hardware stage, except repair of a code-integrity/execution-invalidating defect that does not introduce a new scientific intervention.

If the simulator full gate remains unmet, the hardware stage must be described as **exploratory hardware-transfer validation under an incompletely satisfied simulator gate**.

The QPU pilot itself must be separately frozen before execution and may not be used for tuning.

## Repository lineage

Step 10D was branched from `main` after merged Step-10C result freeze PR #79, whose merge commit is `e71af0297abb41f99098459c438fb72beb728ff9`.

Protocol/config/runner/tests are frozen on branch `agent/step10d-final-simulator-lr-refinement` before any Step-10D training outcome.
