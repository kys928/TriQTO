# Step 10C forensic audit

Status: **POSTHOC AUDIT OF COMPLETED FROZEN BENCHMARK**

Benchmark: `benchmark_f9478da45d68795655259054`

Official frozen outcome: `NO_INITIALIZATION_PASSES_FULL_DUAL_DOMAIN_GATE`.

## Integrity and execution boundary

The uploaded Step-10C audit ZIP has SHA256 `efbb016d7c4f6c0908217ce5cca8ba89e23fe785fa71e247dbbe843b09955ba4` and passes ZIP CRC validation. All included result artifacts whose hashes are declared by `benchmark_complete.json` match byte-for-byte: `decision.json`, `evaluation_boundary.json`, `model_selection.json`, `outer_domain_metrics.csv`, `paired_initialization_differences.csv`, `training_history.csv`, and `outer_predictions.npz`.

The six selected `.pt` checkpoint files are not present in the uploaded audit bundle, so their bytes cannot be independently rehashed here. Their SHA256 values remain frozen in `benchmark_complete.json`.

The history has 195 rows: six runs, epoch 0 for every run, and 189 trained epochs total. Every trained epoch used exactly 188 optimizer steps and 94 optimizer blocks per domain. All numeric training metrics and outer predictions are finite. No warning, traceback, NaN, inf, or exception appears in the run log.

The strict fresh-outer boundary held. All six checkpoint selections completed before fresh outer NPZ materialization. `evaluation_boundary.json` records six selected runs, no fresh-outer materialization during training, no outer use for epoch selection, and no outer use for threshold selection. The run did not evaluate the Step-10B outer cohort, did not execute a QPU, and did not use crash-recovery resume.

Warm/scratch truth arrays and bootstrap groups align exactly. Original outer validation uses 1,000 clean-root bootstrap groups; bridge outer validation uses 60 parent-group bootstrap groups.

## Official gate result

Warm-start bridge mechanism BA is `0.7939043`, with 95% CI lower `0.7809313` and minimum class recall `0.7493056`. Scratch bridge mechanism BA is `0.7868372`, CI lower `0.7735038`, and minimum recall `0.7645833`.

Both initializations therefore pass the CI-lower and minimum-recall subgates but fail the frozen `BA >= 0.80` requirement. Warm-start misses by `0.0060957`; scratch misses by `0.0131628`. The gate is not rounded or relaxed.

Both initializations pass original-domain retention. Relative to the untouched Step-9A ensemble on the same fresh original cohort, warm-start improves original mechanism BA by about `0.00646` while effect BA drops only `0.00118`; scratch improves original mechanism BA by about `0.00832` while effect BA drops only `0.00101`.

## Per-mechanism behavior

Fresh bridge effect-positive mechanism supports are RZ=1,279, RX=1,440, RY=1,920.

Warm-start recalls:
- RZ `0.814699`
- RX `0.749306`
- RY `0.817708`

Scratch recalls:
- RZ `0.788116`
- RX `0.764583`
- RY `0.807813`

Warm-start mainly gains RZ and RY while scratch gains RX. RX remains the weakest warm-start class. Independent reconstruction from `outer_predictions.npz` reproduces the frozen mechanism BA values and confusion matrices exactly.

## Initialization comparison

The paired bridge primary comparison is scratch minus warm-start mechanism BA = `-0.0071548`, 95% CI `[-0.0149801, 0.0004085]`. The interval narrowly includes zero, so neither initialization has statistically clear primary-metric superiority.

Secondary metrics are mixed. Scratch is significantly better on bridge integrated diagnosis BA by about `0.00804` with CI `[0.001796, 0.014417]`, and on bridge integrated macro-F1 by about `0.00789` with CI `[0.001042, 0.014442]`. Scratch also has a significant bridge effect macro-F1 advantage. These secondary findings do not replace the frozen primary decision rule.

A simple post-hoc 50/50 warm/scratch logit average does not rescue the primary gate; its bridge mechanism BA is about `0.79243`. Even an outer-tuned convex weight sweep peaks below `0.80`, so there is no evidence that a trivial warm/scratch ensemble is the missing solution. This observation is post-hoc only and may not be used as a fresh claim.

## Horizon result

Step 10C increased only the maximum training ceiling from 20 to 40 while keeping the scientific optimizer/training settings frozen.

Selected epochs are:
- warm-start: seed1701=24, seed1702=27, seed1703=18;
- scratch: seed1701=34, seed1702=38, seed1703=26.

Epochs actually run are:
- warm-start: 28, 31, 22;
- scratch: 38, 40, 30.

Five of six runs therefore stopped by the frozen patience-4 rule before epoch 40. Only scratch seed1702 reached the 40-epoch ceiling, and its selected checkpoint was epoch 38. This means the max-epoch ceiling is no longer the primary limiting factor.

On the shared Step-10 fit/selection trajectory, the longer horizon did create better selected checkpoints after the old Step-10B stopping points for five of six seeds. Relative to the old selected epochs, bridge selection BA gains are approximately:
- warm1701 +0.00754;
- warm1702 +0.02023;
- warm1703 +0.00000;
- scratch1701 +0.02599;
- scratch1702 +0.02810;
- scratch1703 +0.01317.

This is the valid evidence that additional optimization room mattered. The Step-10B and Step-10C outer cohorts are different, so their outer BA values must not be treated as a paired causal comparison of the horizon change.

## Loss and optimization behavior

Total training loss continues to fall through the end of every trajectory even when selection mechanism BA has plateaued or oscillated. This suggests that the current objective continues to be optimized after balanced mechanism classification stops improving materially.

The new clipping telemetry is decisive: `fraction_optimizer_steps_clipped = 1.0` for every trained epoch of all six runs. Mean post-clip gradient norm is effectively exactly 1.0 throughout. Mean pre-clip norms averaged about 8.56-9.86 for scratch and 11.64-13.06 for warm-start, with a maximum observed pre-clip norm of `248.8573`.

Therefore clip norm 1.0 is not acting as a rare emergency safeguard. It is continuously normalizing every optimizer step and is part of the effective optimization regime. This does not prove that changing the clip threshold will improve generalization, especially with AdamW's adaptive scaling, but it is now strong evidence that optimization behavior deserves a dedicated ablation before any architecture or dataset redesign.

## Interpretation

Step 10C is a valid near-miss, not a gate pass. The architecture and graph-information contract remain viable: deployment-domain performance is close to the predeclared gate and original-domain retention is intact. There is no evidence here requiring a new checkpoint architecture, new graph input contract, or new simulator family.

A further `40 -> 60` max-epoch-only experiment is not justified. Five of six runs already terminate by early stopping, and the remaining run selected before the ceiling. The next development stage should target optimization/objective alignment rather than simply increasing the epoch ceiling.

The fresh Step-10C outer cohort is now spent. Any follow-up regimen motivated by this audit must be selected using fit/selection data only and must use a newly frozen untouched outer cohort for a new reviewer-resistant outer claim.
