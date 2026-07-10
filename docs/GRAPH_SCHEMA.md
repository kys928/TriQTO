# Phase 8 Graph Schema

Phase 8 is a deterministic representation step. It converts one completed Phase 7 raw dataset into a separate immutable graph dataset. It does not train a model, prove correction sufficiency, compute Hilbert geometry, implement topology, or establish quantum advantage.

## Source contract and immutability

The converter requires a valid Phase 7 `dataset_complete.json`, sorted unique `managed_files`, all five typed Phase 7 manifests, safe relative artifact references, exactly one Qiskit circuit per QPY file, and strict exact-probability/count JSON. Completion-marker scientific and operational IDs, sample count, and manifest count are cross-checked against persisted content.

Every managed source file is byte-hashed before and after conversion. Statevector files may be included in that byte snapshot but are never opened as NumPy arrays. The graph dataset is written elsewhere; the Phase 7 source root remains unchanged.

## Graph identity and sample provenance

A graph is a circuit/run-level object. Its deterministic `graph_id` depends on:

- source `circuit_id`
- source exact statevector `run_id`
- graph role (`clean` or `distorted`)
- versioned `graph_schema_id`

A graph does not own one authoritative `sample_id`. Reused graph provenance is stored as a sorted `source_sample_ids` list, excluded from graph identity and structural hashing. The authoritative sample relationship is `GraphPairRecord`, which joins one Phase 7 sample to its clean graph, distorted graph, distortion, and metric.

## Nodes

Each logical circuit qubit is exactly one node. `node_index` stores logical indices separately and is not a learned numeric feature. The fixed node feature order is:

1. measured flag
2. measurement count
3. reset count
4. single-qubit gate incidence count
5. two-qubit gate incidence count
6. total instruction incidence count
7. angular-parameter incidence count
8. sum of sine of angular parameters
9. sum of cosine of angular parameters
10. unique interaction-neighbor count
11. normalized first active logical layer
12. normalized last active logical layer
13. normalized active-layer span

Graphs are variable-size and are never padded to a dataset-wide qubit count.

## Gate events and logical layers

Every circuit instruction becomes one ordered gate event. Gate events preserve the raw operation name, deterministic vocabulary ID, arity, circuit order, logical layer, measurement/reset/barrier flags, parameter counts, and control/symmetry masks. CSR arrays preserve ordered qubit operands, classical operands, measurement wiring, and numeric gate parameters.

Logical layers use a deterministic per-qubit frontier algorithm: operations on disjoint qubits may share a layer, while each operation advances every qubit it touches. These layers are dependency layers, not backend durations or pulse times. Circuit order remains separately stored.

Phase 8 v1 rejects classical conditions and control-flow operations because it cannot preserve their semantics yet. Operations on more than two qubits remain gate-incidence events and do not create fabricated pairwise cliques.

## Directed interaction multigraph

Every two-qubit gate event creates two directed message edges. Both edges reference the same gate event through `edge_event_index`. Repeated gates remain repeated multiedges. CX-like control/target direction is explicit; symmetric interactions such as SWAP are marked symmetric. One-qubit gates, measurements, resets, and barriers create no interaction edges. Physical backend coupling edges are not invented from ideal Phase 7 data.

## Parameters and phasors

Global Phase 7 parameter bindings are sorted by name and stored as raw float64 values plus sine/cosine encodings. Gate parameters are stored in CSR order. Only slots in a fixed versioned angular-slot map receive gate-level sine/cosine values; non-angular numeric parameters keep raw values and use zero phasor placeholders. Unbound, nonnumeric, boolean, and non-finite parameters are rejected.

Circuit global phase is retained only as provenance. It is excluded from node, edge, gate, global learned features, graph identity, and the structural content hash.

## Born evidence and supplemental counts

Exact Born probabilities are required and stored as a sorted variable-length outcome table:

- `outcome_bitstrings`: fixed-width Unicode
- `exact_probabilities`: float64

Bitstrings must be unique binary strings of width `n_qubits`. Values must be finite int/float values and cannot be booleans or numeric strings. Meaningfully negative values are rejected. Tiny negative numerical noise inside the declared tolerance may be clipped to zero and is recorded; no renormalization, thresholding, or truncation occurs. Tiny positive mass is preserved.

Ideal-shot counts are separate supplemental arrays. They are linked through the ideal-shot `SimulationRecord.metadata["source_run_id"]`, strictly validated against shot totals, and never replace exact probabilities. Disabling supplemental counts does not change graph IDs, pair IDs, graph schema IDs, exact evidence, or structural content hashes.

## Born metric pairs

Pair artifacts preserve sorted Born metric names, finite float64 placeholders, and a boolean positive-infinity mask. Positive infinity uses the Phase 7 explicit marker contract. NaN, negative infinity, orphan markers, unknown markers, and finite values with non-finite markers are rejected. Applicability warnings come from `MetricRecord.metadata`, not distortion metadata.

## Structural hashes and provenance

The graph structural content hash is SHA-256 over canonical scientific schema metadata and sorted core arrays, including exact Born evidence. It excludes IDs, sample provenance, file paths, statevector references, count references, count availability, timestamps, and temporary/output locations. Consequently marker-only clean/distorted graphs may have different graph IDs but equal structural hashes when their represented content is equal.

Pair artifacts have their own deterministic content hash over metric arrays, pair identities, labels, and applicability warning. Hashes are recomputed during NPZ readback and compared with typed manifest records.

## Persistence

A Phase 8 output root contains:

```text
graph_config.json
graph_summary.json
graph_complete.json
manifests/graph_manifest.parquet
manifests/graph_pair_manifest.parquet
artifacts/graphs/<graph_id>.npz
artifacts/pairs/<graph_pair_id>.npz
```

The output root must not exist. All data is written into a unique sibling staging directory. Graph and pair NPZ files are loaded with `allow_pickle=False`, fully validated, and hash-checked. Both manifests are typed-read and their joins are validated. A sorted managed-file inventory is checked before `graph_complete.json` is written. The staging directory is then atomically renamed to the final root. A failed write removes only its staging directory and never creates a partial final root.

## Scope exclusions

Phase 8 creates no training split, correction target, action label, learned embedding, graph neural network, topology feature, noisy simulation, physical coupling graph, hardware call, Hilbert feature vector, or evaluation claim. The next planned phase is Phase 9: action and correction engine.
