# TriQTO Architecture

## 1. Motivation
TriQTO studies quantum circuit optimization using quantum-native structure: parameters, Hilbert-state evolution, Born probabilities, phase, phasors, entanglement, lattice connectivity, geometry, and topology.

## 2. Core hypothesis
Optimization signals may become more interpretable and transferable when modeled across the parameter, Hilbert-state, and Born-probability manifolds instead of only through scalar loss traces.

## 3. Three-manifold chain
The central map is `θ → |ψ(θ, x)⟩ → pθ(y|x) = |⟨y|ψ(θ, x)⟩|²`.

## 4. Parameter manifold
Parameter records describe trainable angles, constraints, sine/cosine encodings, and local perturbations. The Phase 13 model keeps raw parameter and periodic phasor streams separate so their contribution can be audited.

## 5. Hilbert-state manifold
Simulation can expose pure-state amplitudes. Phase 13 canonicalizes global phase before encoding and treats Hilbert information as optional privileged input. Density matrices and general channels remain future work.

## 6. Born-probability manifold
Measurement probabilities and counts are the common observable interface across simulation and hardware. Phase 13 uses variable-support basis-string tables rather than a fixed `2^n` output layer.

## 7. Variable-size circuit/lattice graph representation
Qubits are nodes, entangling gates or couplings are edges, gates are layer/time features, and measurements are output evidence. Phase 13 batches concatenated graphs through ownership indices instead of fixing a global qubit count.

## 8. Phasor-aware representation
Parameters use explicit sine/cosine encodings. Graph messages are mixed through learned in-phase and quadrature channels using bounded sine/cosine phase fields. This preserves a phasor language without pretending the classical network is executing a quantum operation.

## 9. Dual-mode simulation/hardware encoder
Simulation mode may use Hilbert features. Hardware mode rejects Hilbert tensors and Hilbert-dependent topology before forward computation. A mode embedding conditions mask-aware fusion while the information-set difference remains explicit.

## 10. Geometry metric stack
Parameter pullback geometry, Hilbert fidelity/Fubini–Study geometry, and Born Hellinger/Jensen–Shannon/Fisher–Rao geometry remain data and training objectives rather than implicit Euclidean assumptions. Phase 13 includes mask-aware geometry-consistency loss primitives but no trainer.

## 11. Persistent homology module
Phase 11 computes deterministic Vietoris-Rips persistent homology over aligned action neighborhoods and circuit cohorts. H0 and H1 are active; H2 is optional. Outputs include diagrams, Betti curves, persistence entropy, top lifetimes, total persistence, and bounded audit heuristics.

## 12. Cross-manifold topology alignment
Phase 11 compares parameter, optional pure-state Hilbert, and Born persistence diagrams. Phase 13 can encode the resulting feature vectors, while its topology prediction head cannot directly copy topology input and has no supervised target in this phase.

## 13. Distortion diagnosis
The diagnosis head predicts a coarse versioned distortion family, optional strength, and affected-qubit logits. Mapping raw Phase 12 distortion names into the coarse label vocabulary is an explicit future data-adapter responsibility.

## 14. Learned action/correction policy
The action head scores a variable number of candidates and normalizes them independently per graph. Candidate ordering is independent of target rank. Privileged oracle status is target/provenance metadata, not an observable input feature.

## 15. Baselines
Random correction, rule-only correction, loss-only optimization, SPSA, COBYLA, and transpiler-only baselines remain required controls for later evaluation.

## 16. Training stages
Phase 12 materializes leakage-safe views. Phase 13 implements the forward architecture and loss contracts. Phase 14 will add data adapters, optimization, schedules, checkpoints, and actual learning while enforcing Phase 12 per-head masks.

## 17. Hardware validation
IBM Runtime validation remains deferred until training and evaluation are complete. Hardware-masked simulation is explicitly not hardware evidence.

## 18. Limitations
The Phase 13 network is untrained. Its outputs carry no empirical performance meaning. It has not demonstrated correction success, topology benefit, simulator-to-hardware transfer, generalization, or quantum advantage.

## 19. Implementation phases
See `docs/CODEX_IMPLEMENTATION_ORDER.md` for the exact phase order.

## Phase 9 deterministic action precursor

Phase 9 implements a bounded action substrate: deterministic candidates, privileged synthetic inverse labels, safe circuit application, exact Born rollout evaluation, transparent rewards, and immutable records. Physics priors provide scaffolding and supervision; they do not replace a learned model.

## Phase 10 baseline controls

Phase 10 evaluates deterministic controls against the same exact Born target and records every method's access privileges. These controls define floors and simulator-oracle ceilings for later model evaluation.

## Phase 11 TriQTO-PH audit

TriQTO-PH materializes parameter/Hilbert/Born topology evidence across aligned point clouds. The topology contract remains:

```text
topology = audit + feature
lambda_top = 0
```

## Phase 12 leakage-safe view layer

Phase 12 groups all records derived from one clean circuit into one split, isolates targets from inputs, carries per-head masks, and removes Hilbert-dependent features from hardware-masked simulation.

## Phase 13 phase-coupled graph model

The implemented model contains seven streams:

```text
circuit graph | parameter | phasor | optional Hilbert | Born | backend | topology
```

Each output head receives a separately masked fusion. A hard policy prevents runtime code from enabling known shortcut streams. The core circuit encoder uses directed multiedge message passing with learned phase quadratures:

```text
m_ij = m_cos · cos(phi_ij) + m_sin · sin(phi_ij)
```

This replaces transformer-style Q/K/V attention as the central interaction mechanism. Graph pooling supports variable qubit counts, Born prediction supports variable queried outcome support, and action ranking supports variable candidate counts.

The model exposes fused latent states, which can later form the learned latent point cloud `P_Z`. Persistent homology over trained latent trajectories remains a Phase 14–15 activity because an untrained architecture does not yet define meaningful latent topology.

The architecture manifest is deliberately honest:

```text
trained = false
optimizer_state_present = false
training_checkpoint = false
topology_loss_weight = 0.0
```

## Phase 14 deterministic optimization

Phase 14 connects Phase 12 views to the Phase 13 model through a strict variable-size adapter. Graphs, candidates, outcomes, parameters, and optional Hilbert amplitudes remain ragged. Action/topology normalization is fitted on train data only. The test split and `audit_only` records are excluded from gradients and model selection.

The training curriculum progresses from foundational task views to joint multitask views and hardware-masked simulation. A separate Hilbert-to-Born auxiliary pass reuses the Born head without allowing standard Born prediction to consume privileged Hilbert or Born-target streams. Model, optimizer, scheduler, and RNG state are checkpointed without pickle and can be restored exactly.

Persistent topology remains an audit and optional feature stream. Its objective coefficient is structurally present but fixed at zero throughout Phase 14.
