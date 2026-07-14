# Phase 12 Task-Specific Training Views

Phase 12 converts the validated Phase 7, Phase 8, Phase 9, and Phase 11 data lake into deterministic task-specific view items. It performs **no model training**. Its job is to decide what each future task may see, what counts as a target, which fields must be masked, and which related records must stay in the same split.

## Source chain

Phase 12 consumes:

- Phase 7 raw circuit, simulation, distortion, and metric records;
- Phase 8 variable-size graph and graph-pair artifacts;
- Phase 9 action candidates, candidate circuits, and exact rollout labels;
- Phase 11 persistent-homology features and alignment audits.

Phase 10 baselines are evaluation controls and are not training inputs.

Every managed source file is byte-snapshotted before and after view construction and publication. Phase 12 never modifies a source dataset.

## Leakage-safe split policy

The split unit is `clean_circuit_id`, not an individual distorted row.

All distortions, graph pairs, actions, rollouts, and sample-level task views derived from one clean circuit therefore inherit one deterministic split:

```text
SHA-256(clean_circuit_id + split_seed) -> train | validation | test
```

The default fractions are 80/10/10. Hash assignment is deterministic and does not rebalance small datasets by moving individual samples.

A topology cohort can contain points derived from several clean-circuit groups. When those source groups span more than one split, the topology item is retained as `audit_only` and is excluded from train/validation/test. Assigning it to one trainable split would expose information from another.

## Materialized item contract

Each item is a strict NPZ artifact with fixed NumPy dtypes and `allow_pickle=False`. It contains:

- input-group names and availability masks;
- target-group names and availability masks;
- source dataset, usage, and relative-reference arrays;
- task-specific structural inputs and targets;
- strict UTF-8 JSON metadata encoded as `uint8`.

References are separated by usage:

- `input` — a future loader may consume the referenced artifact as input;
- `target_provenance` — the artifact explains a materialized target and must not become input;
- `provenance` — traceability only;
- `audit` — diagnostic source only.

The view item hash covers all arrays, IDs, split information, masks, and metadata. Output paths and timestamps are excluded from scientific identity.

## Views

### Distortion diagnosis

Inputs:

- programmed clean circuit graph structure;
- distorted exact basis-conditioned `p(y | M)` evidence;
- backend group marked unavailable for current simulator data.

Targets:

- distortion type;
- optional distortion strength with an availability mask;
- affected-qubit mask.

Distortion labels exist only in `diagnosis_*` target arrays. Synthetically injected gates are not copied into graph inputs. Identifiability status, reason, and diagnosis supervision mask are explicit; unidentifiable class, strength, and affected-qubit targets are inactive by default.

### Action ranking

Inputs:

- distorted graph structure;
- safe candidate features: edit count, risk score, depth delta, gate delta, no-op flag;
- deterministic ragged edit definitions;
- candidate QPY circuit references.

Targets:

- exact Phase 9 rank;
- reward;
- selected/non-worsening/dominance masks.

Candidate ordering is by `action_id`, never by target rank. Phase 9 rollout artifacts are target provenance only. Candidate JSON files are provenance rather than inputs because they contain generation-source labels. Oracle-inverse candidates remain present with a separate privileged mask; that mask must not be treated as an observable deployment feature.

Clean-target metric context is unavailable as an action-ranking input because hardware execution will not provide a clean target.

### Born prediction

Inputs:

- graph structure;
- parameter values;
- sine/cosine phasor encodings.

Target:

- exact programmed-clean basis-conditioned Born distribution.

Phase 8 graph files contain exact Born evidence, so Phase 12 does not expose the raw graph artifact as a direct model input. It physically copies only structural and parameter arrays into the view item and excludes outcome probabilities and supplemental counts. The probability artifact is target provenance only.

### Hilbert-to-Born

Input:

- a Phase 7 programmed-clean statevector reference, only when persisted simulation state is available.

Target:

- the matching exact programmed-clean basis-conditioned Born distribution.

The statevector is not duplicated into the view artifact. This view is simulation-only and cannot be used in hardware mode. When the source dataset intentionally did not store statevectors, the view definition can be valid with zero items and the joint view masks the Hilbert head.

### Topology audit

Inputs:

- Phase 11 topology and alignment feature vectors.

Target:

- none in Phase 12.

`topology_audit_only` is explicitly unavailable as a supervised target. Topology is a reusable diagnostic feature, and `topology_loss_weight` remains exactly `0.0`.

### Joint multitask

One item combines diagnosis, action ranking, Born prediction, optional Hilbert-to-Born, and optional action-neighborhood topology evidence for the same sample.

A joint artifact may contain Born evidence for diagnosis and the same distribution as the Born-prediction target. It therefore carries a mandatory per-head input-mask matrix. The Born-prediction head masks the Born input; the action head masks rollout target provenance; the Hilbert head is active only when a statevector reference exists. Phase 13 must enforce these masks before shared or head-specific computation.

All Born input/target arrays include measurement-setting IDs, per-qubit `Z/X/Y` basis codes, and a row-to-setting index. Each setting distribution is normalized separately.

### Hardware-masked simulation

This is **not hardware data**. It is simulator-derived training material shaped to match missing hardware-only fields.

The view:

- removes all Phase 7 Hilbert input references;
- records `hilbert_available_mask = false`;
- exposes an explicit Hilbert-mask signal;
- keeps backend unavailable;
- excludes Phase 11 topology whenever that topology was computed with Hilbert access.

Topology may be included in a hardware-masked item only when the supplied Phase 11 dataset itself was generated with `include_hilbert=false`. Filtering Hilbert-named features afterward would be insufficient because the Phase 11 parameter pullback metric may already contain Hilbert information.

## Identities

`training_view_schema_id` versions:

- task names and ordering;
- input and target groups;
- split semantics;
- mask semantics;
- item and manifest formats;
- the fixed `topology_loss_weight = 0` boundary.

`training_view_dataset_id` depends on:

- Phase 7 scientific generation ID;
- Phase 8 graph conversion ID;
- Phase 9 action engine ID;
- Phase 11 topology audit ID;
- training-view schema ID;
- scientific task, split, mask, Hilbert, and topology choices.

Operational item/candidate/reference ceilings have a separate config ID. Exceeding one raises instead of truncating a view.

## Output layout

```text
training_view_config.json
training_view_summary.json
training_view_complete.json
manifests/training_view_manifest.parquet
manifests/training_item_manifest.parquet
artifacts/items/<view_item_id>.npz
```

The definition manifest records one row per enabled task, including empty optional views. The item manifest records every materialized item and its task, split, split group, masks, artifact reference, and content hash.

Before publication Phase 12 typed-reads both manifests, reloads every NPZ with `allow_pickle=False`, recomputes hashes and IDs, validates every source reference against its source root, verifies split-group isolation, and rechecks source snapshots. The final root must be new and is published by atomic sibling-directory rename.

## Scientific boundary

Phase 12 provides deterministic data selection, masks, targets, splits, and provenance. It does not provide:

- a neural model;
- training loops or gradients;
- activated topology loss;
- learned correction;
- model evaluation;
- hardware execution;
- quantum advantage.

The next phase is Phase 13 model architecture, which must consume these masks rather than bypass them.
