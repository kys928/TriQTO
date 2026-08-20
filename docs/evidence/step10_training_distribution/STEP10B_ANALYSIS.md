# Step 10B forensic audit

Status: **POSTHOC AUDIT OF COMPLETED FROZEN BENCHMARK**

Benchmark: `benchmark_0971e5bb2ec77c2a1bb550d8`

Official frozen outcome: `NO_INITIALIZATION_PASSES_FULL_DUAL_DOMAIN_GATE`.

## Integrity

The uploaded audit ZIP passed CRC validation. Every included result artifact whose hash is declared by `benchmark_complete.json` matched byte-for-byte: `decision.json`, `model_selection.json`, `outer_domain_metrics.csv`, `paired_initialization_differences.csv`, `training_history.csv`, and `outer_predictions.npz`. The audit ZIP SHA256 is `26aab8b957f3d06d583b347b4095a381c3efbae33d39af9ac0fe52359054d640`.

The history contains exactly six runs, each with epoch 0 plus epochs 1–20, for 120 trained epochs total. Every trained epoch used 188 optimizer steps and 94 optimizer blocks per domain. All recorded losses, gradient norms, logits, and predictions are finite. No warning, traceback, NaN, inf, or exception appears in the run log. Warm-start and scratch outer truth arrays and bootstrap-group arrays are exactly aligned. Original outer bootstrap uses 1,000 root groups of 13 examples each; bridge outer bootstrap uses 60 parent groups of 104 examples each.

The audit bundle does not contain the six `.pt` files, so their bytes cannot be independently rehashed from the ZIP. Their frozen SHA256 values are recorded in `benchmark_complete.json`.

## Gate result

Warm-start bridge mechanism BA was 0.7796769 with 95% CI lower 0.7612315 and minimum class recall 0.7425191. Scratch bridge mechanism BA was 0.7740111 with CI lower 0.7557038 and minimum recall 0.7432150. Both therefore pass the CI-lower and minimum-recall subgates but miss the predeclared BA >= 0.80 requirement. Warm-start misses by 0.0203231; scratch misses by 0.0259889.

Both initializations pass original-domain retention and all six selected checkpoints are retention-eligible. Relative to untouched Step-9A, warm-start improves original mechanism BA by about 0.00457 and effect BA by about 0.00253. Scratch improves original mechanism BA by about 0.00328 while effect BA drops about 0.00513, still well inside the 0.02 tolerance.

The paired bridge primary comparison, scratch minus warm-start mechanism BA, is -0.0056907 with 95% CI [-0.0124700, 0.0019181], so no statistically clear primary-metric superiority is demonstrated. Warm-start does show paired advantages on some secondary/integrated metrics, but those do not replace the frozen primary decision rule.

## Per-mechanism behavior

Warm-start bridge recalls: RZ 0.79453, RX 0.74252, RY 0.80198. Scratch: RZ 0.74766, RX 0.74322, RY 0.83116. Warm-start therefore mainly improves RZ while giving up some RY recall; RX is the weakest class for both and is almost identical between them.

Nominal bridge interventions are balanced in `outer_predictions.npz` at 1,920 examples each for RZ/RX/RY plus 480 clean examples. The effect-positive mechanism-diagnosis mask is not balanced because detectability differs: 1,280 RZ, 1,437 RX, and 1,919 RY examples are effect-positive. Balanced accuracy is therefore the correct primary summary rather than raw accuracy.

## Selection and horizon

The existing runner already performs best-checkpoint selection. It evaluates original and bridge selection metrics every epoch, applies the frozen original-retention eligibility floor, ranks eligible checkpoints primarily by bridge mechanism BA, stores the best eligible state in memory, reloads it after training, and only then evaluates outer validation. It does **not** simply save the last epoch.

Selected epochs were warm-start 1701=19, 1702=17, 1703=18; scratch 1701=20, 1702=20, 1703=20. Thus the epoch ceiling is clearly binding for scratch, but not literally binding for the selected warm-start checkpoints. Warm-start still remains near the ceiling and training losses continue falling, so a separately frozen longer-horizon follow-up is reasonable, but the audit does not justify assuming that more epochs alone will cross 0.80.

The current early-stopping design means a larger maximum epoch ceiling does not force every warm-start seed to run to that ceiling: eligible non-improvement increments patience, while ineligible epochs do not consume patience. With patience 4, a larger ceiling primarily creates room for additional improvements rather than requiring all extra epochs.

## Optimization diagnostics

Training is stable: total loss falls for all six runs, and no numerical failures occur. Mean pre-clip gradient norm is substantially above the clip threshold 1.0 throughout (roughly 4.8–11.2 for scratch and 7.2–16.5 for warm-start). This shows clipping is materially active, but the current audit records only the mean pre-clip norm, not the fraction of steps clipped or post-clip norm. Because retention is stable and losses fall, this alone is not sufficient evidence to change the clip threshold. A follow-up should log clipping fraction/max norm, but keep the frozen clip value unless a separate optimization study is declared.

## Engineering hardening

Atomic checkpointing is strongly recommended and is orthogonal to the scientific model change. The interrupted first Step-10B run demonstrated the operational weakness: the best eligible state lived only in RAM until the entire benchmark completed.

A hardened follow-up should atomically persist the best eligible checkpoint whenever the same existing selection rule improves (`write temp -> fsync/close -> os.replace`). It should also maintain an atomic per-epoch resume checkpoint containing model state, AdamW optimizer state, completed epoch, stale counter, best eligible/unconstrained keys and states, Python/NumPy/Torch RNG states, and the deterministic run identity/config hashes. Resume must verify exact config/data/checkpoint identity before proceeding. If resume is actually used, provenance must say so explicitly; it should not be described as a fresh optimizer continuation.

Saving the best eligible checkpoint and saving a resumable last-epoch state are different jobs and both are useful: `best` is for model selection/deployment; `resume` is for crash recovery.

## Follow-up discipline

No architecture, graph-information-contract, domain-balancing, learning-rate, or gradient-clipping change is justified by Step 10B alone. The cleanest next intervention is a separately frozen longer-horizon + crash-safe checkpointing study with the same optimization settings. Because this design is motivated by observed Step-10B results, the Step-10B outer cohorts are spent and cannot be presented as fresh confirmation in that follow-up. A new untouched outer cohort is required for a reviewer-resistant new claim.
