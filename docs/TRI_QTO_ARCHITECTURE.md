# TriQTO Architecture

## 1. Motivation
TriQTO studies quantum circuit optimization using quantum-native structure: parameters, Hilbert-state evolution, Born probabilities, phase, phasors, entanglement, lattice connectivity, geometry, and topology.

## 2. Core hypothesis
Optimization signals may become more interpretable and transferable when modeled across the parameter, Hilbert-state, and Born-probability manifolds instead of only through scalar loss traces.

## 3. Three-manifold chain
The central map is `θ → |ψ(θ, x)⟩ → pθ(y|x) = |⟨y|ψ(θ, x)⟩|²`.

## 4. Parameter manifold
Parameter records describe trainable angles, constraints, sin/cos encodings, and local perturbations.

## 5. Hilbert-state manifold
Simulation can expose statevector or density-matrix references. These are optional and masked in hardware mode.

## 6. Born-probability manifold
Measurement probabilities and counts are the common observable interface across simulation and hardware.

## 7. Variable-size circuit/lattice graph representation
Qubits are nodes, entangling gates or couplings are edges, gates are layer/time features, and measurements are output evidence.

## 8. Phasor-aware representation
Features will include sin/cos angles, magnitude/phase pairs, relative phase, and interference-sensitive summaries.

## 9. Dual-mode simulation/hardware encoder
The encoder must distinguish simulation-available Hilbert inputs from hardware-masked inputs without leakage.

## 10. Geometry metric stack
Metrics are scaffolded for parameter, Hilbert, and Born manifolds, including QGT/QFI placeholders.

## 11. Persistent homology module
Persistent homology is a topology module from the beginning. H0 and H1 are planned first; H2 is optional.

## 12. Cross-manifold topology alignment
Alignment modules will compare topology across parameter, Hilbert, and Born point clouds.

## 13. Distortion diagnosis
Distortion records describe phase, amplitude, entangling, readout, depolarizing, damping, thermal, layout, and mixed noise.

## 14. Learned action/correction policy
Actions may operate at node, edge, or circuit level and are validated before reward estimation.

## 15. Baselines
Random correction, rule-only correction, loss-only optimization, SPSA, COBYLA, and transpiler-only baselines are required.

## 16. Training stages
Training proceeds through task-specific views: diagnosis, action ranking, Born prediction, Hilbert-to-Born, topology audit, multitask, and hardware-masked training.

## 17. Hardware validation
IBM Runtime validation is deferred until simulation, fake backend, data lake, and masking contracts work.

## 18. Limitations
This scaffold does not implement heavy algorithms, train models, call hardware, or claim performance.

## 19. Implementation phases
See `docs/CODEX_IMPLEMENTATION_ORDER.md` for the exact phase order.
