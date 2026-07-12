# Phase 14 Deterministic Training Engine

Phase 14 turns the completed Phase 12 training-view dataset and the Phase 13 untrained architecture into a reproducible trained-model run. It performs optimization, validation-based model selection, exact checkpoint/resume, and immutable run publication. It does not perform held-out test evaluation, baseline comparison, reporting, or hardware execution; those remain Phases 15 and 16.

## Source boundary

The training engine accepts one completed Phase 12 root. It validates the completion marker, managed inventory, configuration, typed manifests, every NPZ item, content hashes, semantic joins, and split labels before optimization. Managed Phase 12 files are byte-hashed before and after training. A Phase 7 root is additionally required when selected views contain simulator-only statevector input references.

Only `train` examples contribute gradients. `validation` examples are used for loss monitoring, best-checkpoint selection, and optional early stopping. Validation is model selection, not held-out test evaluation. `test` is never loaded into the optimizer or model-selection loop. `audit_only` topology records never contribute gradients.

## Tensor adapter

The adapter converts Phase 12 arrays into the strict Phase 13 contracts without fixed-qubit padding:

- circuit graphs are concatenated with offset node, edge-event, gate, and incidence indices;
- parameter and phasor rows remain ragged through batch indices;
- Born distributions and queried outcome supports remain variable length;
- action candidates remain variable length, and multi-qubit edits expand into one operand row per referenced qubit;
- optional pure states are loaded from Phase 7 with `allow_pickle=False`, validated, and represented by real/imaginary amplitudes plus basis strings;
- topology feature names are qualified, sorted, and mapped into a versioned train-derived vocabulary;
- action and topology normalization statistics are derived from the train split only and persisted in `training_data_spec.json`.

Unknown distortion labels, action edit types, validation-only topology feature names, malformed pointers, unsafe paths, or oversized examples fail rather than being guessed, truncated, or silently ignored.

## Head masking and auxiliary Hilbert-to-Born pass

Phase 12 per-head masks remain authoritative. Runtime masks may remove streams but cannot override the Phase 13 hard policy.

Standard Born prediction uses graph/parameter/phasor inputs and cannot consume Born targets. Simulation-only Hilbert-to-Born supervision reuses the Born prediction head in a separate auxiliary forward pass whose only permitted scientific stream is Hilbert. This avoids weakening the standard Born-prediction leakage boundary. Standalone Hilbert-to-Born items borrow only the matching entity's Phase 12 Born-prediction graph anchor, and the graph stream remains masked out for the auxiliary head.

Hardware-masked simulation rows set hardware mode explicitly, contain no Hilbert values, and reject topology computed with Hilbert access. They remain simulator-derived examples and are not described as hardware evidence.

## Curriculum

A training configuration contains an ordered list of stages. The default recipe is:

1. diagnosis, action ranking, Born prediction, and optional Hilbert-to-Born foundation training;
2. joint multitask training with Phase 12 per-head masks;
3. hardware-masked simulation training.

Every epoch receives a deterministic item ordering derived from the configured seed. Batch packing obeys item-count, node, edge, gate, candidate, outcome, and Hilbert-amplitude ceilings. A single oversized example raises; no record is silently dropped or truncated.

## Objective

The implemented objective contains:

- diagnosis type classification, strength likelihood, and affected-qubit localization;
- selected-action listwise likelihood, rank-distribution matching, and reward regression;
- Born KL divergence and Hellinger distance computed over each graph's complete active support, then averaged across graphs;
- auxiliary Hilbert-to-Born KL and Hellinger distance;
- cross-example geometry consistency using Born Hellinger and optional normalized Fubini-Study targets;
- optional heteroscedastic uncertainty weighting;
- a topology term present in the contract but multiplied by exactly zero.

The topology boundary remains:

```text
topology = audit + optional input feature
lambda_top = 0
```

A nonzero topology loss requires a future versioned change supported by controlled ablations.

## Optimizer and scheduler

Phase 14 supports AdamW and SGD. Learning-rate schedules are constant or explicit warmup-cosine schedules. Gradient accumulation is normalized by the exact item count in each accumulation window, including the final partial window. Finite-norm clipping is supported. Non-finite losses or gradients fail immediately.

## Safe exact checkpoints

Checkpoints are compressed NPZ artifacts loaded with `allow_pickle=False`. A JSON tree describes model, optimizer, scheduler, and RNG state; every tensor or NumPy array is stored as a separate fixed-dtype NPZ array. The logical content hash covers metadata, names, shapes, dtypes, and exact bytes.

A resumable checkpoint contains:

- full model state;
- optimizer slots and parameter groups;
- scheduler state and step;
- Python, NumPy, Torch CPU, and optional CUDA RNG state;
- Phase 12 dataset, Phase 13 model, data-spec, recipe, operational, and run identities;
- epoch, stage, global step, best epoch, and best validation loss.

Initialized Phase 13 weights are not called trained checkpoints. Phase 14 checkpoints explicitly contain optimizer/scheduler/RNG state and are read back before publication.

## Immutable output

A run is built in a unique sibling staging directory, typed-read and hash-validated, and atomically renamed into a fresh output root. The output root is rejected when it equals, contains, or is contained by either the Phase 12 source or the optional Phase 7 source.

```text
training_config.json
model_config.json
training_data_spec.json
training_summary.json
training_complete.json
manifests/training_epoch_manifest.parquet
manifests/training_checkpoint_manifest.parquet
artifacts/checkpoints/*.npz
```

The completion marker records the exact managed inventory and source snapshot hashes. Output paths and timestamps do not enter scientific recipe identity.

## Claim boundary

A completed Phase 14 run may truthfully claim that a model was optimized on a particular validated training-view dataset. It may not claim held-out generalization, superiority to baselines, correction success, hardware transfer, universal correction, or quantum advantage until the later evaluation and hardware phases provide evidence.
