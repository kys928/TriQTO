# Phase 9 Action and Correction Schema

Phase 9 is a deterministic candidate-generation and simulator-validation layer. It turns each completed Phase 7 clean/distorted sample and its completed Phase 8 graph pair into a bounded set of circuit edits, applies every edit to an independent circuit copy, evaluates every candidate with exact ideal-statevector Born probabilities, and ranks the resulting rollouts with a transparent reward.

Phase 9 does **not** contain a learned action policy. It produces validated candidate/rollout evidence that can later supervise and evaluate a learned policy. It makes no claim about noisy-simulator or hardware performance.

## Source contract

The engine requires two immutable inputs:

1. a completed Phase 7 raw dataset;
2. the completed Phase 8 graph dataset derived from that exact Phase 7 dataset.

The loader validates both completion markers, their sorted managed-file inventories, typed manifests, artifact references, graph/pair joins, graph conversion identity, Phase 7 scientific generation identity, and the Phase 7 source snapshot recorded by Phase 8. Every managed source file is byte-hashed before and after Phase 9 work. Statevector artifact files are never loaded from either source root.

## Bounded action vocabulary

Phase 9 v1 supports four circuit edits applied before final measurements:

- `append_rx(qubit, angle)`
- `append_ry(qubit, angle)`
- `append_rz(qubit, angle)`
- `append_rzz(qubit_a, qubit_b, angle)`

An empty edit tuple is the explicit no-op action. Angles are finite and wrapped to `(-π, π]`. Zero-angle edits are rejected in favor of the no-op representation. RZZ proposals are restricted to two-qubit logical interactions already present in the distorted circuit. Phase 9 does not invent physical coupling edges or all-to-all entangling actions.

Classical conditions, control flow, unbound parameters, and non-final measurements are unsupported in v1. They raise instead of being silently discarded.

## Candidate sources

Each sample may receive candidates from three deterministic sources:

- **No-op:** a conservative reference candidate.
- **Blind physics-prior grid:** bounded positive and negative RX, RY, RZ, and observed-edge RZZ edits. These proposals do not inspect the clean target or distortion label.
- **Synthetic oracle inverse:** an exact inverse derived from privileged Phase 7 distortion metadata when the synthetic unitary is known. This is supervised synthetic label generation only. It is not a hardware-facing inference rule, not a learned policy, and not evidence that the distortion can be identified from measurements.

Oracle inverses exist for Phase 7 RZ drift, RX/RY overrotation, RZZ drift, and the registered mixed unitary drift. Marker-only readout/layout records do not receive a fabricated circuit oracle. Duplicate edit payloads are merged while preserving all deterministic generation-source labels.

## Identity separation

`action_schema_id` versions the edit vocabulary, application semantics, angle normalization, risk heuristic, rollout representation, primary reward metrics, and ranking contract.

`action_engine_id` depends on:

- the Phase 7 scientific generation ID;
- the Phase 8 graph conversion ID;
- the action schema ID;
- scientific candidate-generation and reward choices.

Operational candidate/edit-count guardrails do not enter the scientific engine ID and cannot alter risk or rollout identity unless exceeded, in which case generation fails rather than truncates.

`action_id` depends on the sample, graph pair, distorted source circuit/run, ordered edit payload, and action schema. It is independent of output paths and whether the same edit was reached by the blind grid, the oracle, or both.

`candidate_circuit_id` depends on source circuit, action ID, and application version. `rollout_id` additionally depends on the clean target run and scientific reward/candidate configuration. Action and rollout content hashes protect the complete persisted artifact content while remaining independent of file locations.

## Safe circuit application

Every non-no-op edit is inserted before final measurements using the existing validated final-measurement handling contract. Measurement wiring is restored afterward. Source circuits are semantically fingerprinted before and after application, and mutation raises. Candidate circuits preserve logical qubit and classical-bit counts.

The resulting QPY candidate artifact is read back and checked with a semantic hash covering operation order, numeric parameters, qubit/classical operands, conditions, global phase, and circuit parameter state.

## Exact rollout validation

Every candidate is evaluated using `qiskit.quantum_info.Statevector` through the existing ideal simulation layer. The clean Phase 7 exact Born distribution is the target. The uncorrected distorted distribution is the baseline.

Primary metrics use a fixed order:

1. total variation distance;
2. Jensen–Shannon divergence;
3. Hellinger distance.

All are finite lower-is-better distances. The rollout stores baseline values, candidate values, improvements, the candidate exact Born outcome table, depth/gate deltas, and the complete transparent reward breakdown.

The reward is:

`weighted Born improvement - depth penalty - gate penalty - edit penalty - risk penalty`

No-op naturally receives zero edit/depth/gate/risk penalty. A candidate is marked non-worsening only when every primary metric is no worse than baseline within the configured tolerance. It dominates baseline only when all are non-worsening and at least one is strictly better. Ranking is deterministic: non-worsening candidates first, then reward descending, risk ascending, and action ID. Exactly one rank-one candidate is selected per sample.

This selection is an ideal-simulator data label, not a deployment decision. A no-op may correctly win when proposed edits do not improve the observable Born target enough to justify their cost.

## Output layout

Phase 9 writes a fresh immutable root:

```text
action_config.json
action_summary.json
action_complete.json
manifests/action_candidate_manifest.parquet
manifests/action_rollout_manifest.parquet
artifacts/actions/<action_id>.json
artifacts/circuits/<candidate_circuit_id>.qpy
artifacts/rollouts/<rollout_id>.npz
```

Action JSON is strict and rejects duplicate keys, NaN, and infinity. Rollout NPZ files use fixed Unicode/numeric arrays and JSON metadata encoded as `uint8`; they load with `allow_pickle=False`. Both manifests are typed-read. Every candidate, circuit, rollout, hash, rank, selected flag, reference, and join is revalidated before publication.

The output root must not exist and cannot be nested inside either source root. Phase 9 builds in a unique sibling staging directory, writes the completion marker only after full validation, and atomically renames the staging root. Failure removes only that staging directory.

## Scope boundary

Phase 9 includes deterministic bounded candidates, exact simulator rollouts, transparent rewards, and synthetic oracle supervision. It does not include:

- a learned correction policy or action-ranking neural network;
- a diagnosis model;
- noisy simulation, fake backends, IBM Runtime, or hardware calls;
- layout/routing/pulse changes;
- topology or persistent homology;
- baselines (Phase 10);
- training views, model architecture, or training;
- claims of universal correction, real-hardware safety, or quantum advantage.
