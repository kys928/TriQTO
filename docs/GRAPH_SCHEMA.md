# Phase 8 Graph Schema

Phase 8 is a deterministic representation step that converts completed Phase 7 raw datasets into framework-neutral NumPy graph artifacts. It does not train a model, prove correction sufficiency, establish quantum advantage, compute Hilbert geometry, implement topology, or create correction actions.

## Variable-size graph design

Each logical circuit qubit is exactly one node. Two-qubit gate events produce a directed event-level multigraph with one forward and one reverse message edge; repeated interactions remain repeated multiedges. One-qubit gates, measurements, resets, and barriers remain ordered gate events but do not create interaction edges. Operations on more than two qubits remain incidence-table events; Phase 8 v1 does not fabricate pairwise cliques or physical backend coupling graphs.

## Node, edge, gate, and global features

Node feature order is fixed: measured flag, measurement count, reset count, single-qubit incidence count, two-qubit incidence count, total incidence count, angular-parameter incidence count, summed sine/cosine of angular parameters, unique interaction-neighbor count, and normalized first/last/span active logical layers. Raw qubit indices are stored separately in `node_index` and are not learned node features.

Edge features store normalized gate order/layer, forward or reverse direction, operand positions, known control/target semantics, symmetric-gate flag, and parameter counts. CX-like gates preserve control direction; SWAP/CZ are marked symmetric.

Gate events preserve raw operation names, deterministic vocabulary IDs, arity, order, logical layer, flags for measurement/reset/barrier/arity class, parameter counts, control-semantics masks, and symmetric-interaction masks. CSR arrays preserve variable qubit operands, classical-bit operands, measurement wiring, and numeric gate parameters without padding.

Global learned feature vectors are empty in v1. Circuit global phase is retained only in metadata because global phase is not physically meaningful as a learned input; relative and gate-local phase parameters are represented via gate parameters and phasor encodings.

## Temporal layers

Logical layers are deterministic circuit layers, not physical pulse schedules or backend-duration times. Disjoint gates may share a layer; order remains separately preserved.

## Parameters and Born evidence

Phase 7 parameter bindings are sorted by name and stored with raw values plus sine/cosine phasors. Only fixed, versioned angular operation slots receive gate-level sine/cosine encodings; non-angular numeric parameters keep raw values and zero phasor placeholders.

Exact Born probabilities are required as a variable-length sorted outcome table (`outcome_bitstrings`, `exact_probabilities`). Tiny positive probabilities are preserved without thresholding or truncation. Optional count artifacts are provenance/supplemental evidence only and never replace exact probabilities or alter structural graph IDs/content hashes. Statevectors are never loaded for graph features; only source references are retained, and `hilbert_available_mask` remains false.

## IDs, manifests, and artifacts

`graph_schema_id` hashes the schema version, feature ordering, gate vocabulary version, angle-slot mapping version, and directed-multiedge representation version. `graph_id` depends on source circuit ID, role, and schema ID. `graph_pair_id` depends on sample ID and clean/distorted graph IDs. `graph_conversion_id` depends on the source scientific generation ID and graph schema ID. Output paths, timestamps, guardrails, and supplemental counts do not define structural identities.

Logical content hashes are SHA-256 hashes over sorted array names, dtypes, shapes, contiguous bytes, and canonical schema metadata, not raw NPZ bytes. Graph manifests and graph-pair manifests are Parquet files with safe relative artifact references.

## Persistence

Graph datasets are written to new immutable output roots only. A unique sibling staging directory is fully written and validated before an atomic rename publishes the final root. No Phase 7 source dataset is modified.

## Scope exclusions

Phase 8 creates no training split, action label, correction target, graph neural network, PyTorch Geometric/DGL dependency, topology feature, noisy simulation, hardware call, Hilbert metric, or evaluation report. The next recommended phase is Phase 9: action and correction engine.
