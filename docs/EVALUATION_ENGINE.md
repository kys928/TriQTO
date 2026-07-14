# Phase 15 Deterministic Evaluation Engine

Phase 15 consumes one completed Phase 12 training-view dataset and one completed Phase 14 training run. It selects a validated `best` or `final` checkpoint, restores the trained Phase 13 model, evaluates only the untouched `test` split, and publishes immutable reports.

## Source boundary

The evaluator validates:

- the complete Phase 12 managed inventory and snapshot;
- the complete Phase 14 managed inventory, summary, model config, training config, data spec, checkpoint manifest, and selected checkpoint;
- that Phase 14 never used the test split for optimization and never used `audit_only` rows for gradients;
- that the Phase 12 snapshot matches the exact source recorded by Phase 14;
- optional Phase 7/8/9/10 sources when baseline comparison is enabled.

The output root is rejected when it equals, contains, or is contained by any source root.

## Held-out evaluation

Only records physically labelled `test` are loaded. Phase 15 runs under `torch.no_grad()` with `model.eval()`. No optimizer, scheduler update, checkpoint mutation, early stopping, or validation-based decision occurs.

The evaluator supports the Phase 12 task universe:

- diagnosis;
- action ranking;
- Born prediction;
- optional Hilbert-to-Born prediction;
- joint multitask views;
- hardware-masked simulation views.

Tasks with no test rows may be absent unless the evaluation config requires non-empty coverage for the entire run.

## Metrics

Diagnosis reports classification accuracy and negative log-likelihood, optional strength absolute error, and affected-qubit accuracy.

Action ranking reports top-1 agreement with the Phase 9 selected target, reciprocal rank of the target, target rank and reward of the learned selection, and the fraction of learned selections that use privileged synthetic oracle candidates.

Born and Hilbert-to-Born distributions use complete per-graph metrics before averaging across items:

- KL divergence;
- Hellinger distance;
- Jensen-Shannon divergence;
- total variation distance;
- probability mean absolute error.

Outcome terms are never averaged globally across graphs, so variable support size does not reweight examples.

## Calibration

Diagnosis confidence is the maximum class probability. Action confidence is the selected candidate probability. Phase 15 reports expected calibration error, maximum calibration error, Brier score, mean confidence, empirical accuracy, and occupied calibration-bin count.

Calibration is descriptive. It is not post-hoc recalibrated on the held-out test set.

## Generalization tables

Metrics are aggregated as unweighted means across held-out items. Reports are grouped by:

- task;
- circuit family when present;
- qubit count;
- distortion identity when present.

Missing metadata is represented explicitly as `unknown`. Small subgroup counts remain visible and must not be interpreted as broad generalization evidence.

## Ablations

Phase 15 performs inference-time stream ablations without retraining:

- `full`;
- `no_topology`;
- `no_hilbert`.

The ablation removes the stream object and disables that stream in every per-head runtime mask. It does not enable any stream forbidden by the Phase 13 hard policy. `no_hilbert` on hardware-masked rows is expected to be a no-op because Hilbert input is already absent.

These are checkpoint-level sensitivity tests, not causal proof that a stream improved learning. Training-time ablations remain a future controlled experiment.

## Optional Phase 10 baseline comparison

When enabled, Phase 15 also validates the exact Phase 7/8/9/10 source chain. The learned action is joined to its Phase 9 rollout and evaluated with the same weighted exact Born objective used by Phase 10. It is then compared sample-by-sample with every enabled baseline.

Privilege is preserved explicitly:

- rule-only uses synthetic distortion metadata;
- loss-only consults the clean target during selection;
- SPSA and COBYLA optimize against the clean target;
- random correction is a sanity control;
- transpiler-only is backend-free and not hardware-aware.

A learned win against a privileged or non-deployable baseline is reported as a numerical comparison, not as deployment superiority.

## Immutable output

A fresh output root is built in sibling staging and atomically renamed only after typed readback and artifact validation.

```text
evaluation_config.json
evaluation_summary.json
evaluation_complete.json
manifests/evaluation_item_manifest.parquet
manifests/evaluation_aggregate_manifest.parquet
manifests/evaluation_baseline_manifest.parquet   # optional
artifacts/items/*.npz
reports/summary.json
reports/generalization.json
reports/calibration.json
reports/ablations.json
reports/baselines.json
```

Evaluation item artifacts are pickle-free NPZ files with strict JSON metadata stored as UTF-8 bytes. Logical hashes cover metadata, array names, shapes, dtypes, and exact bytes.

## Claim boundary

A completed Phase 15 run may claim measured performance on its exact held-out test universe. It may not claim real-hardware transfer, universal correction, quantum advantage, or performance outside the evaluated circuit families, qubit counts, distortions, and simulator assumptions.
