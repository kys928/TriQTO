# TriQTO — Tri-Manifold Quantum Topological Optimization

TriQTO (pronounced “Trikto”) is a research codebase for studying quantum-native, hardware-aware quantum-circuit optimization. The project is organized around

```text
Parameter manifold → Hilbert-state manifold → Born-probability manifold
θ → |ψ(θ, x)⟩ → pθ(y|x) = |⟨y|ψ(θ, x)⟩|²
```

TriQTO does **not** claim quantum advantage, universal quantum correction, or solved topology optimization. The repository is a staged implementation scaffold for future validation.

## Repository principle

The final architecture exists from the beginning, while expensive validation is populated progressively:

```text
TriQTO Data Lake → task-specific training views → model heads → evaluation/hardware validation
```

The data lake stores circuit, backend, simulation, distortion, metric, action, topology, and training-view records. Training jobs should select only the fields required by a task rather than forcing one monolithic dataset.

## Variable-size graph design

TriQTO treats circuits and hardware lattices as variable-size graphs. Qubits are nodes, interaction events or physical couplings are edges, gates carry logical layer/order information, and measurements are observable output evidence. This avoids fixed 4-qubit or 8-qubit vector assumptions.

## Simulation and hardware modes

Simulation records may include Hilbert-state references such as statevectors or density matrices. Hardware records cannot expose Hilbert states, so Hilbert inputs must use masks and optional references. This prevents Hilbert-field leakage during hardware-masked training.

## Phasors, geometry, and topology

The architecture reserves first-class modules for sine/cosine angle encodings, magnitude/phase features, relative phase, interference-sensitive signals, geometry metrics across the three manifolds, and persistent homology. Topology is initially diagnostic; topology loss remains inactive until its signals are validated.

## Baselines

Physics priors are scaffolding and validators, not unquestioned final authority. Phase 10 implements deterministic random, privileged rule-only, clean-target loss-only, SPSA, COBYLA, and backend-free transpiler controls. Their access privileges and limits are persisted explicitly.

## Implementation phases

1. Repo skeleton and contracts.
2. Core IDs, enums, schema dataclasses, manifest writer/reader.
3. Circuit family generation.
4. Simulation layer.
5. Distortion engine.
6. Metric engine.
7. Data generation pipeline.
8. Graph conversion.
9. Action and correction engine.
10. Baselines.
11. Topology module.
12. Training views.
13. Model architecture.
14. Training engine.
15. Evaluation and reports.
16. Hardware validation.

## Phase 8 graph conversion

Phase 8 converts completed Phase 7 datasets into deterministic, framework-neutral NumPy graph artifacts. One logical qubit becomes one node; each two-qubit gate event becomes two directed multiedges; ordered gate events, operand incidence, classical wiring, parameters, logical layers, and exact Born evidence are preserved without dataset-global padding.

Graph identity is circuit/run-level, not sample-level. Sample ownership lives in graph-pair records, so a clean graph can be reused across several distortion samples without inheriting an arbitrary first `sample_id`. Exact probability evidence participates in graph identity through the source exact run. Supplemental ideal-shot counts link through the Phase 7 shot record’s `source_run_id`, remain separate from exact probabilities, and do not alter structural graph IDs or structural content hashes.

Phase 8 validates the completed Phase 7 marker and manifests, hashes all managed source files before and after conversion, never loads statevector arrays, writes graph and pair NPZ files with `allow_pickle=False`, typed-reads both graph manifests, validates all joins and hashes, then atomically publishes a fresh immutable output root. Global phase is provenance only and Hilbert-derived feature masks remain unavailable.

Phase 8 introduces no graph neural network, training split, topology feature, correction action, noisy backend, hardware call, or quantum-advantage claim. See [`docs/GRAPH_SCHEMA.md`](docs/GRAPH_SCHEMA.md).

## Phase 9 action and correction engine

Phase 9 converts completed Phase 7/8 sources into deterministic bounded action candidates and exact ideal-statevector validation rollouts. Candidate edits currently include no-op, RX/RY/RZ rotations, and observed-interaction RZZ edits. Every candidate is applied to an independent circuit copy, compared with the clean Phase 7 Born target, assigned a transparent reward, and deterministically ranked.

The engine includes privileged synthetic oracle inverses only as supervised labels for known Phase 7 unitary distortions. It is not a learned policy and does not infer those inverses from hardware observations. Marker-only distortions receive no fabricated circuit oracle, and no-op can win. Phase 9 performs no noisy simulation, hardware calls, topology, baselines, training-view construction, or model training. See [`docs/ACTION_SCHEMA.md`](docs/ACTION_SCHEMA.md).

## Phase 10 baseline suite

Phase 10 consumes the exact completed Phase 7/8/9 chain and evaluates six deterministic controls under the same exact Born objective: random correction, privileged synthetic rule-only inversion, clean-target loss-only action selection, SPSA, COBYLA, and backend-free transpilation. Results are immutable typed artifacts with explicit access-privilege metadata and byte-level source immutability checks.

The baseline suite does not yet compare a trained TriQTO policy because the model and training phases have not been implemented. The transpiler control is semantic and backend-free, not hardware-aware. See [`docs/BASELINE_SCHEMA.md`](docs/BASELINE_SCHEMA.md).
