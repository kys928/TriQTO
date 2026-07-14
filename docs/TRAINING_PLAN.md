# Training Plan

## Current implementation boundary

Phase 12 materializes deterministic task-specific views for diagnosis, action ranking, Born prediction, optional Hilbert-to-Born prediction, topology audit, joint multitask learning, and hardware-masked simulation.

Phase 13 implements the model architecture and strict tensor/output contracts. Phase 14 implements the data adapter, trainer, optimizer, schedules, safe checkpointing, per-example uncertainty likelihood, and actual learning. Phase 15 implements test-only evaluation, axis-holdout audits, uncertainty diagnostics, ablations, and baseline comparisons. No trained empirical result is committed.

## Split policy

The IID policy keeps all records derived from one `clean_circuit_id` in one deterministic train, validation, or test split. The OOD policy reserves configured family, qubit-count, distortion-type, or backend values for test and hashes all other groups into train/validation. Distortion/backend experiments use clean-circuit-plus-axis-value groups so intentional cross-axis comparisons do not leak one axis-specific sample group. Topology cohorts spanning several source splits remain `audit_only`.

## Implemented data-adapter responsibilities for Phase 14

The Phase 14 adapter:

- typed-read and verify the completed Phase 12 dataset;
- map variable Phase 12 graph arrays into `GraphTensorBatch` without fixed-qubit padding;
- map raw distortion names into the versioned Phase 13 coarse label vocabulary explicitly;
- construct parameter, phasor, Born, Hilbert, backend, topology, action, and outcome-query tensors;
- translate Phase 12 per-head masks into the six Phase 13 architecture heads;
- activate uncertainty wherever at least one supervised task is active;
- preserve source usage distinctions so target-provenance artifacts never become inputs;
- keep privileged-oracle masks out of observable candidate features;
- map topology feature names into a fixed versioned dense feature vector;
- reject, rather than silently truncate, oversized or malformed examples.

## Masking policy

Hilbert inputs are optional. Missing statevectors produce false Hilbert masks. Hardware-mode rows contain no Hilbert tensors and no Hilbert-dependent topology.

Born-prediction heads cannot consume Born input evidence. Joint views may contain Born evidence for diagnosis, so Phase 14 must enforce the per-head mask before fusion.

Inactive heads are explicitly marked inactive. Their latent states, fusion weights, predictions, and candidate/outcome probabilities remain zero.

## Curriculum direction

A defensible initial curriculum is:

1. validate tensorization and overfit a tiny debug set;
2. train diagnosis, Born prediction, and action ranking separately;
3. train optional Hilbert-deformation supervision in simulation mode;
4. enable joint multitask training with head-specific masks;
5. introduce hardware-masked simulation rows;
6. run topology as an audit/feature ablation with `lambda_top = 0`;
7. compare against Phase 10 baselines on held-out circuit families, qubit counts, distortions, and later backends.

This order is a plan, not a performance claim.

## Future loss

The documented future objective remains:

```text
L_total = L_task + lambda_geo L_geo + lambda_diag L_diag
          + lambda_action L_action + lambda_top L_top
```

During initial Phase 14 training:

```text
lambda_top = 0
```

Persistent homology is logged and available as a diagnostic feature from the beginning. A nonzero topology loss requires controlled ablations showing useful, non-leaking predictive signal and must be introduced through a later versioned change.

## Training integrity requirements

Phase 14 must record:

- exact Phase 12 dataset and model architecture identities;
- complete training configuration and random seeds;
- deterministic/non-deterministic backend settings;
- per-task and per-split sample counts;
- head and stream mask utilization;
- privileged-label usage;
- optimizer/scheduler state;
- checkpoint hashes;
- baseline comparisons;
- topology feature ablations;
- simulator-only versus hardware-masked results.

It must not label an initialized Phase 13 state as a trained checkpoint.

## Phase 14/15 implementation status

The deterministic training engine is now implemented. The CLI is `scripts/train.py`, with strict recipes in `configs/train/phase14_base.yaml` and `configs/train/phase14_small_debug.yaml`. It validates Phase 12 sources, derives training-only feature statistics, executes the staged curriculum, writes exact resumable NPZ checkpoints, and atomically publishes typed epoch/checkpoint manifests.

Phase 15 is implemented as an immutable evaluation engine. It distinguishes IID test reporting from audited axis-disjoint OOD designs, excludes unidentifiable diagnosis/action labels, scores basis settings independently, evaluates the uncertainty head directly, and task-qualifies baseline identities. Phase 16 remains responsible for real hardware execution and validation.
