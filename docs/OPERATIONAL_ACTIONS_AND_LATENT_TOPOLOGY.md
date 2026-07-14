# Operational actions and checkpoint-bound latent topology

This document describes the offline engineering-validation path added after the core Phase 15 evaluator. It does not define physical-hardware evidence, a trained operational policy, topology benefit, or a new logical-correction reward.

## Operational action families

Operational actions are semantically separate from Phase 9 logical corrections:

- **Basis probe:** creates a separate diagnostic measured circuit for an explicit measurement setting and acquires new `p(y | M)` evidence. It does not modify or correct the primary circuit.
- **Layout selection:** compiles against an immutable fake-backend target and records physical-layout and coupling evidence. It is not a privileged inverse correction.
- **Routing/transpilation:** records backend-bound basis gates, coupling constraints, seed, optimization level, before/after costs, and swap evidence.
- **Semantics-verified depth reduction:** is accepted only when statevector fidelity is within the configured tolerance and at least one objective circuit-cost measure improves without a protected two-qubit-count regression.

Every operational result has a deterministic action ID, content hash, source circuit/backend identity, preconditions, availability mask, evidence tier, semantic validation metadata, objective comparison, and explicit hardware/privilege flags.

## Immutable operational artifacts

`triqto.actions.write_operational_action_dataset` publishes a fresh immutable directory containing:

```text
operational_action_summary.json
operational_action_complete.json
actions/<action_id>.json
phase12_compatible_action_arrays.npz
```

The Phase-12-compatible arrays include:

- action IDs and fixed feature names;
- action-family IDs and names;
- candidate availability masks;
- zero operational target masks;
- zero privileged-oracle masks;
- model-compatible edit arrays.

Unavailable candidates have zero features and own no edit rows. The adapter preserves these masks through deterministic batching. This is structural/model-contract integration, not evidence that Phase 14 learned an operational policy. Operational supervision remains empirically unvalidated.

## Trained checkpoint latent extraction

`triqto.training.extract_checkpoint_latents` requires:

- a completed Phase 14 run;
- a real positive-step checkpoint;
- a valid checkpoint content hash;
- matching model architecture and model configuration IDs;
- matching Phase 12 training-view identity;
- an explicit split, task, head, and latent representation;
- deterministic unique source view-item IDs.

Inference runs without gradients. The extractor rejects untrained or zero-step checkpoints, source/model mismatches, altered checkpoint bytes, missing split metadata, unsupported heads, duplicate IDs, non-finite coordinates, and output collisions.

Output is immutable and contains:

```text
latent_coordinates.npz
latent_metadata.json
latent_complete.json
```

## Checkpoint-bound latent persistent homology

`triqto.topology.run_checkpoint_bound_latent_topology` consumes only a validated latent-extraction artifact. Its identity binds to:

- checkpoint ID and checkpoint content hash;
- model architecture/configuration IDs;
- Phase 12 source identity;
- split, task/head, and representation;
- ordered point IDs and coordinate hash;
- topology configuration and normalization mode.

Absolute coordinate scale is preserved by default. Optional normalized analysis is explicitly labeled `shape_only` and receives a distinct identity. H0 and H1 remain enabled; H2 is optional. The artifact includes persistence diagrams, Betti curves, finite/essential counts, lifetime statistics, persistence entropy, and conservative collapse/loop/late-merge diagnostics.

The artifact always records:

```text
checkpoint_bound = true
trained_checkpoint = true
diagnostic_only = true
physical_hardware = false
topology_loss_weight = 0.0
```

No topology gradient is introduced and no topology-benefit or causal claim is made.

## Phase 15 reporting

The integrated evaluator preserves the existing Phase 15 trained-model report and adds separate operational families:

- basis probes: availability, execution rate, settings, evidence acquisition, failures;
- layout/routing: semantic-validation rate, depth/two-qubit/swap deltas, rejection reasons;
- depth reduction: accepted/rejected/no-op counts, fidelity, and cost deltas;
- latent topology: checkpoint identity, split/head, point count/dimension, normalization mode, and persistence summary.

These metrics are never pooled into one logical-correction reward.

## CPU smoke workflow

Run the complete engineering-validation path into a fresh directory outside repository-managed artifacts:

```bash
python scripts/run_cpu_smoke_workflow.py --output /tmp/triqto-operational-latent-smoke
```

The workflow executes Phase 7, Phase 8, Phase 9, operational artifacts, Phase 11, Phase 12, Phase 14 training, checkpoint restoration, latent extraction, checkpoint-bound latent topology, core Phase 15, and integrated Phase 15 reporting.

Individual stages can also be invoked with:

```bash
python scripts/extract_checkpoint_latents.py --help
python scripts/generate_checkpoint_latent_topology.py --help
python scripts/evaluate_operational_topology.py --help
```

All active configs are CPU-safe and fail closed. No IBM Runtime job is submitted. Generated datasets, checkpoints, latent coordinates, topology artifacts, and result cards are temporary/user-selected outputs and are not committed.
