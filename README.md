# TriQTO — Tri-Manifold Quantum Topological Optimization

TriQTO (pronounced “Trikto”) is a research codebase for studying quantum-native, hardware-aware quantum circuit optimization. The project is organized around the chain

```text
Parameter manifold → Hilbert-state manifold → Born-probability manifold
θ → |ψ(θ, x)⟩ → pθ(y|x) = |⟨y|ψ(θ, x)⟩|²
```

TriQTO does **not** claim quantum advantage, universal quantum correction, or solved topology optimization. This repository is a staged implementation scaffold for future validation.

## Repository principle

The final architecture exists from the beginning, while expensive validation is populated progressively:

```text
TriQTO Data Lake → task-specific training views → model heads → evaluation/hardware validation
```

The data lake is designed to store circuit, backend, simulation, distortion, metric, action, topology, and training-view records. Training jobs should pull only the fields required by a task rather than forcing one monolithic dataset.

## Variable-size graph design

TriQTO treats circuits and hardware lattices as variable-size graphs: qubits are nodes, entangling gates or physical couplings are edges, gates are layer/time features, and measurements are observable output evidence. This avoids fixed 4-qubit or 8-qubit vector assumptions.

## Simulation and hardware modes

Simulation records may include Hilbert-state references such as statevectors or density matrices. Hardware records cannot expose Hilbert states, so Hilbert inputs must be represented by masks and optional references. This prevents Hilbert-field leakage during hardware-masked training.

## Phasors, geometry, and topology

The architecture reserves first-class modules for sin/cos angle encodings, magnitude/phase features, relative phase, interference-sensitive signals, geometry metrics across the three manifolds, and persistent homology. Topology is included from the start as a diagnostic and data-lake component; topology loss remains inactive until signals are validated.

## Baselines

Baselines are required because physics priors should act as scaffolding, validators, and comparisons rather than unquestioned final authority. Random, rule-only, loss-only, SPSA, COBYLA, and transpiler-only baselines are scaffolded.

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
