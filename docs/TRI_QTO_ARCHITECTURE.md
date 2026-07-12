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
Features include parameter values and sine/cosine angle encodings, with future model layers reserved for richer relative-phase and interference-aware processing.

## 9. Dual-mode simulation/hardware encoder
Phase 12 now defines the input masks the future encoder must obey. Simulation views may reference Hilbert state. Hardware-masked simulation contains no Hilbert values or references and carries an explicit false Hilbert-availability signal.

## 10. Geometry metric stack
Metrics are scaffolded for parameter, Hilbert, and Born manifolds, including QGT/QFI placeholders.

## 11. Persistent homology module
Phase 11 computes deterministic Vietoris-Rips persistent homology over aligned action neighborhoods and circuit cohorts. H0 and H1 are active; H2 is available only when explicitly configured. Outputs include diagrams, Betti curves, persistence entropy, top lifetimes, total persistence, and bounded audit heuristics.

## 12. Cross-manifold topology alignment
Phase 11 compares parameter, optional pure-state Hilbert, and Born persistence diagrams using bottleneck and 1-Wasserstein distances. Parameter distance is a downstream pullback-style pseudometric, Hilbert distance is projective Fubini-Study distance, and Born distance is configurable among Hellinger, square-root Jensen-Shannon, and normalized Fisher-Rao distance.

## 13. Distortion diagnosis
Distortion records describe phase, amplitude, entangling, readout, depolarizing, damping, thermal, layout, and mixed noise. Phase 12 isolates distortion type, strength, and affected-qubit labels from diagnosis inputs.

## 14. Learned action/correction policy
Actions may operate at node, edge, or circuit level and are validated before reward estimation. Phase 12 orders candidates independently of target rank and keeps rollout scores as targets rather than observable inputs.

## 15. Baselines
Random correction, rule-only correction, loss-only optimization, SPSA, COBYLA, and transpiler-only baselines are required.

## 16. Training stages
Phase 12 materializes diagnosis, action-ranking, Born-prediction, optional Hilbert-to-Born, topology-audit, joint-multitask, and hardware-masked simulation views. Actual model architecture and optimization remain Phases 13 and 14.

## 17. Hardware validation
IBM Runtime validation is deferred until simulation, fake backend, data lake, masking, model, training, and evaluation contracts work. Hardware-masked simulation is explicitly not hardware evidence.

## 18. Limitations
No learned TriQTO model, model training, hardware execution, or performance claim exists yet. Topology remains audited evidence and reusable features with `lambda_top = 0`; Phase 12 makes no claim that topology is predictive or beneficial before later ablations.

## 19. Implementation phases
See `docs/CODEX_IMPLEMENTATION_ORDER.md` for the exact phase order.

## Phase 9 deterministic action precursor

The learned action policy remains a later model/training concern. Phase 9 implements the validated action substrate that such a policy will need: a versioned bounded edit vocabulary, deterministic physics-prior candidates, privileged synthetic inverse labels for known simulator distortions, safe circuit application, exact Born rollout evaluation, transparent rewards, and immutable action/rollout records. Physics priors provide candidate scaffolding and supervision; they do not override a model that does not yet exist.

## Phase 10 baseline controls

Phase 10 makes baseline comparison executable before any learned TriQTO model exists. Random correction, privileged rule-only inversion, clean-target loss-only action selection, SPSA, COBYLA, and backend-free transpilation are evaluated against the same exact Born target and metric order. Each result records what privileged information its method used.

These controls are not substitutes for the future learned policy. They establish the floor and simulator-oracle ceilings that later model evaluation must beat or approach. Hardware-aware transpilation, noisy execution, and device-calibrated optimization remain deferred until the hardware-validation layer; Phase 10 does not fabricate backend structure.

## Phase 11 TriQTO-PH audit

TriQTO-PH materializes topology evidence across aligned parameter/Hilbert/Born point clouds from Phase 9 candidate neighborhoods and Phase 7 family cohorts. Parameter geometry includes downstream deformation, Hilbert topology is optional and pure-state simulation-only, Born topology remains available without Hilbert access, and latent topology remains unavailable until a learned model exists.

The topology contract remains:

```text
topology = audit + feature
lambda_top = 0
```

## Phase 12 leakage-safe view layer

Phase 12 is the boundary between the data lake and future models. It groups every distortion and action derived from one clean circuit into the same deterministic split. Topology cohorts spanning several split groups remain `audit_only` rather than contaminating train, validation, or test.

Born-prediction views physically omit Born evidence from structural graph inputs. Action-ranking views isolate clean-target rollout results as targets. Optional Hilbert views use statevector references only when available. Joint views carry per-head masks because one task may legitimately use evidence that would leak another task's target.

Hardware-masked simulation removes Hilbert references and also removes topology computed with Hilbert access. It is a missing-data training mode, not a claim of real-hardware validation.
