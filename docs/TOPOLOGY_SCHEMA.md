# Phase 11 Persistent-Homology and Topology Audit

Phase 11 implements **TriQTO-PH**, the persistent tri-manifold topology audit. It consumes the exact completed Phase 7 raw dataset, its Phase 8 graph dataset, and its Phase 9 action/rollout dataset. It produces deterministic point-cloud groups, precomputed manifold distances, Vietoris-Rips persistence diagrams, Betti curves, fixed-order topology features, and cross-manifold diagram-alignment features.

This phase is deliberately an **audit and feature layer**. The persisted topology-loss weight is exactly

```text
lambda_top = 0
```

No topology gradient, topology regularizer, learned topology encoder, or topology-optimization claim is introduced yet.

## Point-cloud groups

Phase 11 builds three deterministic group kinds when enough points exist.

### Action neighborhoods

One group is built per Phase 7 sample from every validated Phase 9 candidate action and exact rollout.

The shared point identity is the Phase 9 `action_id`. Each point has:

- a local correction-parameter representation built by summing angle magnitudes on deterministic action axes such as `append_rx:q0` and `append_rzz:q0-1`;
- a recomputed ideal pure state from the persisted candidate QPY circuit when Hilbert auditing is enabled;
- the exact Phase 9 candidate Born distribution.

The action-coordinate reduction is an audit representation, not a claim that arbitrary ordered edit sequences form a globally flat Euclidean parameter space.

### Family/qubit cohorts

Phase 7 distorted samples are grouped by:

```text
family + qubit count
```

The shared point identity is `sample_id`. Bound circuit parameters use a deterministic union of parameter names with an explicit availability mask. Missing parameters are not imputed. Exact distorted Born distributions and optional recomputed ideal pure states remain aligned to the same point IDs.

### Family/qubit/distortion cohorts

A more specific cohort additionally groups by the recorded distortion type:

```text
family + qubit count + distortion type
```

Groups below `min_points` are skipped deterministically and counted in the summary. Groups above operational point ceilings fail instead of being truncated or subsampled.

## Manifolds available in Phase 11

### Parameter manifold

The parameter distance is not plain Euclidean distance. Phase 11 uses a pullback-style pseudometric:

```text
d_theta(i,j) = sqrt(
    w_raw * d_periodic(i,j)^2
  + w_B   * d_Born(i,j)^2
  + w_H   * d_Hilbert(i,j)^2
) / sqrt(active weight sum)
```

`d_periodic` is a wrapped angular root-mean-square distance over shared coordinates. The downstream Born and optional Hilbert deformation components dominate by default. Cohort points with no shared raw parameter coordinate remain distinguishable through downstream behavior.

### Pure-state Hilbert manifold

When `include_hilbert=true`, candidate or distorted QPY circuits are simulated through the existing exact ideal-statevector layer. Distances use normalized Fubini-Study projective distance:

```text
d_FS(psi,phi) = arccos(|<psi|phi>|) / (pi/2)
```

The absolute overlap makes the metric invariant to global phase. Raw statevectors are never persisted in Phase 11 topology artifacts; only the validated distance matrix and derived topology are stored.

This is a pure-state simulation audit. Density-matrix/channel topology is not implemented in Phase 11.

### Born manifold

Born rows are aligned over the deterministic union of binary outcomes and remain exact normalized probability vectors. Configurable distance choices are:

- Hellinger distance;
- square-root Jensen-Shannon distance;
- normalized Fisher-Rao distance `2 arccos(sum sqrt(p_i q_i)) / pi`.

The default is Hellinger distance.

### Latent manifold

No learned TriQTO model exists before Phases 13–14. Phase 11 therefore records:

```text
latent_available = false
```

It does not fabricate latent vectors or latent topology.

## Distance normalization

By default each available manifold distance matrix is divided by its maximum finite entry before persistent homology. This maps the observed group diameter to one and makes filtration ranges comparable across manifolds. The original normalization scale is stored in metadata.

A zero-diameter cloud remains an exact zero distance matrix. Normalization never adds artificial separation.

## Persistent homology

Phase 11 uses `ripser` with a precomputed distance matrix and a Vietoris-Rips filtration.

H0 and H1 are mandatory:

- H0 audits connected components, rapid collapse, separation, and late mergers;
- H1 audits loops and cyclic/folding structure.

H2 is supported only when explicitly enabled in `homology_dimensions`. It is not active by default because its computational and sample requirements are materially higher.

Positive-infinity deaths are preserved in diagrams as essential features. Feature summaries use finite lifetimes and store essential counts separately.

## Model-ready topology features

For every manifold and homology dimension, Phase 11 stores:

- finite feature count;
- essential feature count;
- total persistence;
- persistence entropy;
- maximum and mean finite lifetime;
- top-k finite lifetimes;
- a fixed-grid Betti curve.

It additionally stores three audit heuristics:

- `collapse_score`: high when finite H0 components merge rapidly relative to the filtration range;
- `loop_score`: a bounded transformation of total H1 persistence per point;
- `late_merge_bridge_score`: the largest normalized finite H0 merge scale.

The bridge score is only a late-merge heuristic. It is **not proof** that a bridge is scientifically unwanted.

## Cross-manifold alignment

For every available manifold pair and homology dimension, Phase 11 computes:

- bottleneck distance with L-infinity ground metric;
- 1-Wasserstein distance with L-infinity ground metric and diagonal matching;
- essential-feature count gap;
- `exp(-bottleneck)` alignment similarity.

Finite diagram points are compared geometrically. Essential counts are audited separately because directly mixing infinite deaths into finite assignment costs is not meaningful.

An aggregate topology-preservation score is stored as:

```text
exp(-mean pairwise bottleneck distance)
```

This is a diagnostic comparison across the sampled point cloud. It does not establish that one manifold causally preserves another.

## Identity separation

`topology_schema_id` versions grouping, distance, persistent-homology, feature, alignment, artifact, and manifest contracts.

`topology_audit_id` depends on:

- the Phase 7 scientific generation ID;
- the Phase 8 graph conversion ID;
- the Phase 9 action engine ID;
- the topology schema ID;
- scientific grouping, metric, filtration, feature, and Hilbert-mask choices.

Operational ceilings such as maximum points, groups, and statevector amplitudes do not enter the scientific audit identity. They fail rather than alter or subsample a valid scientific result.

Each `topology_group_id` depends on the audit ID, group kind, group key, sorted point IDs, and grouping version. File paths and timestamps do not enter scientific identities.

## Output layout

Phase 11 writes a fresh immutable root:

```text
topology_config.json
topology_summary.json
topology_complete.json
manifests/topology_group_manifest.parquet
artifacts/groups/<topology_group_id>.npz
```

Each NPZ artifact contains fixed-dtype arrays only and loads with `allow_pickle=False`:

- point IDs;
- parameter coordinate names, values, and masks;
- aligned Born outcome vectors;
- parameter/Hilbert/Born distance matrices;
- persistence diagrams;
- Betti curves;
- manifold topology feature vectors;
- cross-manifold alignment feature vectors;
- strict UTF-8 JSON metadata encoded as `uint8`.

Raw statevectors are never stored in these artifacts.

The manifest is typed-read through `TopologyGroupRecordV1`. Every ID, reference, point count, manifold mask, feature dimension, diagram, distance matrix, probability row, content hash, and manifest/artifact join is revalidated before publication.

## Source and publication integrity

The completed Phase 7/8/9 chain is fully validated before use. Every managed source file is byte-snapshotted before and after topology construction and publication.

The final output root must not exist and cannot be nested inside a source root. Phase 11 writes into a unique sibling staging directory, creates the completion marker only after strict readback validation, and atomically renames the staging directory. Failure removes only that staging directory.

## Honest interpretation

Persistent homology is included because it may expose global structure missed by local scalar metrics. Phase 11 does not assume it is useful.

The correct current claim is:

```text
topology = audit + reusable feature, lambda_top = 0
```

Topology becomes a training signal only after later ablations show that these features predict correction success, failure mode, shortcut learning, or generalization breakdown. Phase 11 adds no model, training view, topology loss activation, hardware call, noisy backend, universal-correction claim, or quantum-advantage claim.
