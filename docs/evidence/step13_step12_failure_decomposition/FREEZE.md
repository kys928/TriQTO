# Step 13 failure-decomposition freeze

Status: **FROZEN POSTHOC DIAGNOSTIC STAGE BEFORE ANY NEW TRAINING**

Step 13 uses only the already-spent Step-12 physical-QPU bundle and frozen Step-10C / Step-9A checkpoints. It performs no QPU access and no parameter updates.

## Source

- Step-12 source bundle: `step12plan_24b2631b5cd94c58af951492.zip`
- SHA256: `9c9a4a726e0403a466960669917a394ef7df73c2f6fb12239dbdf799f818d7ff`
- Step-12 plan: `step12plan_24b2631b5cd94c58af951492`
- IBM Runtime job: `da6od4k6l22c73dmjscg`
- backend: `ibm_marrakesh`
- frozen Step-12 result: Step-10C 8/18, Step-9A report-only 9/18, predeclared gate failed

Step 12 is not rerun, replaced, or reinterpreted.

## Frozen decomposition result A — evidence geometry

The exact deployable local Pauli diagnostics reconstructed from Step-12 counts preserve the controlled ideal mechanism structure:

- mean true-template cosine: 0.9534
- median true-template cosine: 0.9707
- minimum true-template cosine: 0.7909
- ideal/QPU pairwise-distance Pearson: 0.9664
- ideal/QPU pairwise-distance Spearman: 0.9009
- posthoc nearest-ideal-template diagnostic: 18/18
- 1000-replicate multinomial bootstrap: mean oracle correctness probability 0.9982, minimum case 0.983

The template diagnostic is explicitly non-deployable and non-confirmatory; it is used only for failure decomposition.

Frozen interpretation: **simple hardware-evidence collapse is not supported as the primary Step-12 failure explanation. Out-of-distribution learned mapping/generalization is supported as the primary explanation, while the exact model subcomponent remains unresolved.**

## Frozen decomposition result B — required counterfactual replay

Before any new training, the exact frozen models must be replayed on the exact Step-12 graphs/layouts under four no-QPU diagnostic conditions:

1. acquired hardware diagnostics — must reproduce Step-12 predictions;
2. exact ideal statevector diagnostics;
3. ideal local + hardware pair/parity;
4. hardware local + ideal pair/parity.

Runner: `scripts/v0_2/run_step13_ideal_counterfactual_replay.py`.

This replay may diagnose whether idealization rescues the learned mechanism mapping and whether local versus correlation/parity evidence materially contributes. It may not train, choose new thresholds, change checkpoints, or rewrite Step 12.

## Training boundary

**No Step-13-driven retraining is authorized until the ideal-diagnostic counterfactual replay is executed and frozen.**

If ideal diagnostics do not materially rescue Step-10C, the next intervention should target simulator training/generalization across substantially more diverse graph contexts, with Step-10C warm-start reuse unless an architecture-specific failure is separately demonstrated.

If ideal diagnostics substantially rescue Step-10C, hardware-domain robustness of the diagnostic representation must be addressed before concluding that more motif diversity alone is sufficient.
