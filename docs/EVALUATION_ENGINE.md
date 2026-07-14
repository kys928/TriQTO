# Phase 15 deterministic evaluation engine

Phase 15 consumes one completed Phase 12 training-view dataset and one completed Phase 14 training run. It restores a validated `best` or `final` checkpoint, evaluates only Phase 12 `test` rows under `torch.no_grad()`, and publishes immutable evidence. It does not train, tune on test data, or execute hardware.

## Source and design validation

The evaluator validates the complete managed inventories and snapshots for Phases 12 and 14, the checkpoint/data-spec/config identity chain, and the rule that Phase 14 used neither `test` nor `audit_only` rows for gradients. Optional Phase 7/8/9/10 sources receive the same validation when baseline comparison is enabled. Output and source roots may not overlap.

Every run declares one of two evaluation designs:

- `iid_test` accepts only a Phase 12 `clean_circuit_hash` split. This is an untouched test partition from the same configured data universe. It is not OOD evidence.
- `ood_axis_holdout` accepts only a Phase 12 `axis_holdout` split with the exact same axis and values. Phase 15 scans train, validation, and test item metadata and fails unless the declared family, qubit-count, distortion-type, or backend values occur only in test and all development values are disjoint.

Executable paired Phase 12/15 templates exist for family, qubit-count, and distortion-type holdouts. Backend-axis contracts exist, but backend holdout remains non-executable until genuine backend IDs and evidence enter the data path.

## Evaluation universe and identifiability

The supported task universe is diagnosis, action ranking, Born prediction, optional Hilbert-to-Born prediction, joint multitask, and hardware-masked simulation.

Pure diagnosis and action-ranking rows marked `unidentifiable` are excluded from scoring even if supervision was explicitly enabled for an audit experiment. Joint and hardware-masked rows may remain available for their identifiable Born components, but diagnosis/action metrics are suppressed. The summary reports status/reason coverage and per-task exclusion counts, so masking cannot silently improve a headline metric.

## Metrics

Diagnosis reports class accuracy and negative log-likelihood, optional strength absolute error, and affected-qubit accuracy. Action ranking reports top-1 target agreement, target reciprocal rank, learned selection rank/reward, and privileged-oracle selection fraction.

Born and Hilbert-to-Born metrics are computed on each complete basis-conditioned distribution `p(y | M)` before averaging settings within an example and examples within an aggregate. Settings are never concatenated and renormalized as one distribution. Metrics include KL, Hellinger, Jensen-Shannon, total variation, and probability MAE.

Aggregates are unweighted means across items and are grouped by task, family, qubit count, distortion type, and backend ID. In an IID run these are descriptive subgroups, not OOD generalization tables. Small or missing groups remain explicit.

## Confidence and uncertainty diagnostics

Two different quantities are reported and never conflated:

- diagnosis/action softmax confidence receives ECE, MCE, Brier, mean-confidence, and empirical-accuracy summaries;
- when Phase 14 actually enabled uncertainty weighting, the learned per-example head exposes log variance and variance for each active task, paired with a nonnegative per-example predictive error. Phase 15 reports variance/error gaps, MAE, correlation, and binned summaries. Runs trained with uncertainty weighting disabled are marked and do not publish initialized-head calibration rows.

The uncertainty-head diagnostics evaluate the output that Phase 14 actually trains. They do not establish calibrated uncertainty without a credible trained checkpoint, adequate held-out coverage, and prespecified acceptance criteria. No post-hoc calibration is fitted on test data.

## Ablations and baselines

The `full`, `no_topology`, and `no_hilbert` modes are inference-time stream removals. They cannot enable a stream forbidden by the Phase 13 hard policy. These are checkpoint sensitivity tests, not training-time causal ablations.

When enabled, Phase 10 baseline comparison joins each learned action to the same exact Born rollout objective used by the baselines. IDs include run, sample, action-bearing task, and baseline, so multi-task comparisons cannot collide. Baseline privilege remains explicit: rule-only, loss-only, SPSA, and COBYLA use synthetic or clean-target information unavailable to a deployed model; random is a sanity control; transpiler-only is backend-free.

## Immutable output and claim boundary

A fresh root is built in sibling staging, typed-read, hash-checked, and atomically renamed. It contains item, aggregate, and optional baseline manifests; pickle-free NPZ item artifacts; the exact config and completion marker; and summary, subgroup, calibration, ablation, and baseline reports.

A completed IID run may claim performance on its exact untouched test partition. A completed axis-holdout run may claim performance on its exact, audited held-out axis values. Neither establishes broad OOD generalization, calibrated uncertainty, real-hardware transfer, universal correction, quantum advantage, or performance beyond the recorded data and simulator assumptions.
