# Step 10D — final simulator-development LR refinement before IBM hardware

Status: **FROZEN BEFORE STEP-10D OUTCOME**

Schema: `triqto.v0_2.step10d_final_simulator_lr_refinement.v1`

## Purpose

Step 10D is the **final simulator-development intervention before IBM hardware**.

The project has a time-limited IBM QPU-access window. Step 10C was reviewer-clean but failed the frozen full dual-domain gate narrowly: warm-start bridge mechanism BA was `0.7939043` versus the frozen `0.80` requirement, while CI lower, minimum class recall, original-domain retention, execution integrity, and fresh-outer sequencing all passed. Step 10C also showed that the 40-epoch ceiling is no longer the main limitation and that every optimizer step was clipped at norm `1.0`.

Step 10D therefore tests exactly one late-refinement hypothesis and then stops simulator tuning. It is not an LR sweep, not an architecture experiment, not a new dataset experiment, and not another outer-validation claim.

## Frozen intervention

The Step-10C warm-start training trajectory is kept unchanged except for one piecewise-constant learning-rate schedule:

- epochs **1–20**: `3e-4`;
- epochs **21–40**: `1e-4`.

All of the following remain unchanged from the Step-10C warm-start arm:

- architecture: `late_concat`;
- trainable parameters: `453829`;
- Step-9A state-dict warm start;
- fresh AdamW optimizer at epoch 0;
- root batch size `32`;
- max epochs `40`;
- patience `4`;
- min delta `0.0005`;
- weight decay `1e-4`;
- gradient clip norm `1.0`;
- effect loss weight `1.0`;
- mechanism loss weight `1.0`;
- equal original/bridge optimizer-block count per epoch;
- fit and selection partitions from `product_0f7112597501f7ea5fbe123b`;
- seeds `1701, 1702, 1703`;
- per-seed original-domain retention floors;
- eligible-checkpoint ranking rule;
- crash-safe persistence and telemetry.

Scratch is **not** rerun in Step 10D. This stage is not intended to re-answer the initialization-comparison question. Step 10C already found no statistically clear primary winner. Step 10D is a time-bounded final refinement of the reusable warm-start candidate before expiring hardware access.

## Why these exact LR values

The values are deliberately conservative and predeclared; they are **not** claimed to be theoretically optimal.

### Why keep `3e-4` for epochs 1–20?

`3e-4` is already demonstrated to be numerically stable, to learn the bridge task, and to preserve the original task. Epoch 20 is also the exact former Step-10B ceiling. Keeping the first 20 epochs unchanged preserves the known-good early optimization regime and makes the intervention easy to interpret: only the late phase changes.

### Why switch to `1e-4` after epoch 20?

`1e-4` is exactly one third of `3e-4`. Step 10C showed a pattern consistent with late-stage refinement being useful: training loss continued to fall while bridge selection BA oscillated around its best region, and every optimizer step was clipped at norm `1.0`. A 3x reduction is large enough to create a materially finer late update regime without imposing a 10x-or-greater reduction that could effectively freeze a model that still showed development-selection headroom.

We intentionally do **not** test `2e-4`, `1.5e-4`, `5e-5`, `3e-5`, cosine decay, warm restarts, or any other schedule. Doing so would turn Step 10D into an LR hyperparameter search and delay the hardware stage.

## Data boundary

Step 10D is **selection-only development**.

It may access only:

- the original-domain fit and selection partitions from the frozen Step-10 mixture;
- the bridge-domain fit and selection partitions from the frozen Step-10 mixture;
- Step-9A warm-start checkpoints;
- the three frozen Step-10C warm-start checkpoints solely for re-evaluation on the same development selection cohorts.

The following are forbidden in Step 10D:

- reading or materializing the spent Step-10C fresh-outer NPZ artifacts;
- scoring the Step-10C outer cohort;
- generating a new outer cohort for Step 10D;
- QPU execution;
- QPU-driven model or hyperparameter selection.

The Step-10C outer cohort is spent. Step 10D makes no new outer-validation claim.

## Per-seed model selection

Each Step-10D seed uses the same Step-10C warm-start retention rule.

A checkpoint is eligible only if:

1. original-domain mechanism BA is at least the matching-seed Step-9A warm epoch-0 mechanism BA minus `0.02`;
2. original-domain effect BA is at least the matching-seed Step-9A warm epoch-0 effect BA minus `0.02`.

Eligible checkpoints are ranked by:

1. bridge mechanism balanced accuracy;
2. `min(original mechanism BA, bridge mechanism BA)`;
3. bridge effect balanced accuracy;
4. earlier epoch.

The existing `0.0005` minimum-delta rule remains in force. Human monitoring may not override the automatically selected checkpoint.

## Hardware-candidate decision before QPU execution

Step 10D predeclares only two candidate ensembles for the next hardware stage:

- **A:** frozen Step-10C warm-start ensemble;
- **B:** Step-10D warm-start ensemble.

Candidate A is re-evaluated only on the existing development selection cohorts. No Step-10C outer data are re-used.

Step 10D becomes the primary IBM-hardware candidate only if all of the following are true:

1. all three Step-10D selected checkpoints are retention-eligible;
2. the Step-10D ensemble satisfies the same `0.02` original-domain retention tolerance relative to the Step-9A warm epoch-0 ensemble;
3. Step-10D bridge-selection mechanism BA exceeds the re-evaluated Step-10C warm bridge-selection mechanism BA by **more than `0.0005`**.

Otherwise the frozen Step-10C warm-start ensemble remains the primary IBM-hardware candidate.

This decision is made **before any QPU result is observed**. Hardware results may not be used to switch simulator checkpoints retroactively.

## Crash safety

Step 10D retains the Step-10C crash-safe engineering contract:

- atomic `best.pt` on each eligible improvement;
- atomic `resume.pt` after every completed epoch;
- model, AdamW state, early-stopping state, history, and Python/NumPy/Torch RNG state persisted;
- exact run-identity and runtime-fingerprint checks before resume;
- atomic `progress.json`;
- wall-time, current LR, pre-clip norm, max pre-clip norm, fraction clipped, and post-clip norm telemetry.

The LR phase is derived from the completed epoch number, so an exact resume cannot silently continue with the wrong learning rate.

## Hard stop: no more simulator tuning before hardware

This is a predeclared project decision, not a suggestion:

> **After Step 10D completes, TriQTO proceeds to a separately frozen IBM-hardware pilot before any further simulator-development intervention.**

Before that hardware pilot, there will be:

- no LR sweep;
- no further LR schedule revision;
- no clip-threshold tuning;
- no loss-weight tuning;
- no max-epoch extension;
- no architecture change;
- no simulator dataset redesign;
- no additional initialization comparison.

The only exception is repair of a code-integrity or execution-invalidating defect. Such a repair may not be used as a vehicle for another scientific tuning intervention.

## Hardware interpretation boundary

The simulator gate remains whatever the frozen simulator evidence says. Step 10D does not retroactively change the Step-10C decision.

If the full simulator gate is still unmet when hardware begins, the IBM stage will be explicitly labeled:

> **exploratory hardware-transfer validation under an incompletely satisfied simulator gate**

It will not be described as confirmatory proof of the full TriQTO hypothesis.

The QPU pilot itself must be separately frozen before execution. No QPU tuning is permitted.

## Frozen next action

1. run the Step-10D warm-start three-seed development benchmark exactly once;
2. freeze the automatic simulator-candidate decision;
3. stop simulator tuning;
4. freeze the IBM-hardware pilot;
5. execute hardware while access remains available.
