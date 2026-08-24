# Step 10D final simulator LR-refinement audit

Status: **POSTHOC AUDIT OF COMPLETED FROZEN BENCHMARK**

Benchmark: `benchmark_1455864a09de8804a7e7958a`

Predeclared role: final simulator-development intervention before IBM hardware.

## Integrity

The uploaded Step-10D ZIP has SHA256 `e62eb3de2c7e5ccd00b0087d095c4f60394d79ffbd601a1c2042ac7ee926901b` and passes ZIP CRC validation. It contains the run log, benchmark completion/decision/selection files, training history, and all three selected warm-start `.pt` checkpoints.

Every file hash declared in `benchmark_complete.json` matches the uploaded bytes, including all three checkpoints. The checkpoint payloads identify `late_concat`, seeds 1701/1702/1703, selected epochs 28/31/21, retention eligibility true, and no optimizer state in the exported selected checkpoints. Each state dict contains the expected model parameters and the frozen architecture remains 453,829 trainable parameters.

The history has 95 rows: epoch 0 plus 32 trained epochs for seed 1701, epoch 0 plus 35 for seed 1702, and epoch 0 plus 25 for seed 1703, totaling 92 trained epochs. Every trained epoch has 188 optimizer steps. All recorded training/selection metrics are finite and the run log contains no traceback, NaN, or infinity failure.

No crash-recovery resume was used. The spent Step-10C outer was not accessed, no new outer cohort was accessed, and no QPU execution occurred.

## Frozen intervention check

The realized LR schedule exactly matches the predeclared Step-10D contract for every history row:

- epochs 1–20: `3e-4`;
- epochs 21–40: `1e-4`.

No seed ran to 40 because the unchanged selection/early-stopping rule stopped all three trajectories earlier. Selected epochs are 1701=28, 1702=31, 1703=21. All selected checkpoints are retention-eligible.

Gradient clipping remained continuously active: every optimizer step in every trained epoch reports clipping fraction 1.0. This is recorded as context only; the frozen protocol prohibits another clipping experiment before hardware.

## Predeclared hardware-candidate decision

The Step-10C warm ensemble, re-evaluated on the same frozen development selection sets, has bridge mechanism balanced accuracy `0.7961509367`.

The Step-10D warm ensemble has bridge mechanism balanced accuracy `0.7925952870`.

Therefore:

`Step10D - Step10C = -0.0035556497`.

The predeclared rule required Step 10D to improve by more than `0.0005`, while also preserving seed and ensemble retention. Retention conditions pass, but the performance condition fails in the opposite direction.

**Frozen primary hardware candidate: `step10c_warm_start`.**

This decision is based only on frozen development-selection evidence and was made before any new QPU execution.

## Interpretation

The 3x late LR reduction did not improve the primary development mechanism metric. It modestly improves some original-domain/effect summaries but reduces the primary bridge mechanism BA relative to the already-frozen Step-10C warm ensemble. There is therefore no scientific basis to replace the Step-10C hardware candidate with Step 10D.

This is not a reason to start another simulator sweep. Step 10D was explicitly frozen as the final simulator-development intervention. The negative result is informative: the simple late-LR refinement hypothesis was tested once and rejected under its predeclared selection rule.

## Hard stop and next stage

No further pre-hardware LR search, clip tuning, loss reweighting, epoch extension, architecture revision, or fit/selection dataset redesign is permitted under the frozen Step-10D protocol.

The next stage is a separately frozen **exploratory IBM hardware transfer pilot** using the Step-10C warm ensemble as the primary candidate. Because the full simulator gate remains unmet, hardware evidence must remain explicitly exploratory and may not be used for QPU-driven tuning or retroactive simulator-model selection.
