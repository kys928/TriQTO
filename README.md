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

The baseline suite does not yet compare a trained TriQTO policy because the training phase has not been implemented. The transpiler control is semantic and backend-free, not hardware-aware. See [`docs/BASELINE_SCHEMA.md`](docs/BASELINE_SCHEMA.md).

## Phase 11 persistent-homology audit

Phase 11 implements deterministic Vietoris-Rips persistent homology over aligned action-neighborhood and circuit-cohort point clouds. It computes H0 and H1 by default, optional H2, Betti curves, persistence entropy, top lifetimes, collapse/loop/late-merge heuristics, and bottleneck/Wasserstein alignment across parameter, pure-state Hilbert, and Born manifolds.

Parameter topology uses a downstream pullback-style pseudometric rather than plain Euclidean distance. Hilbert topology is optional and maskable, uses Fubini–Study projective distance, and never persists raw statevectors. Topology remains **audit + reusable feature** with `lambda_top = 0`; no topology optimization or training signal is claimed. See [`docs/TOPOLOGY_SCHEMA.md`](docs/TOPOLOGY_SCHEMA.md).

## Phase 12 task-specific training views

Phase 12 turns the validated Phase 7/8/9/11 chain into deterministic diagnosis, action-ranking, Born-prediction, optional Hilbert-to-Born, topology-audit, joint-multitask, and hardware-masked simulation views. Related distortions and actions are split together by clean circuit, while topology cohorts spanning several splits remain `audit_only`.

The view layer physically blocks Born-target leakage from Born-prediction graph inputs, separates action rollout labels from candidate inputs, carries explicit privileged-oracle masks, supports optional Hilbert references, and removes Hilbert-dependent topology from hardware-masked simulation. It performs no model training and keeps `lambda_top = 0`. See [`docs/TRAINING_VIEW_SCHEMA.md`](docs/TRAINING_VIEW_SCHEMA.md).

## Phase 13 model architecture

Phase 13 implements the untrained PyTorch TriQTO architecture. It combines a variable-size circuit graph encoder, explicit parameter and phasor streams, optional global-phase-invariant Hilbert encoding, variable-support Born encoding, optional backend/topology streams, dual simulation/hardware mode, and head-specific mask-aware fusion.

The graph core uses learned sine/cosine phase quadratures over directed lattice messages rather than transformer Q/K/V attention. Hard stream policies and Phase 12 runtime masks prevent Born-target copying, direct Hilbert copying in the Hilbert-deformation head, direct topology copying in the topology audit head, and Hilbert leakage into hardware-mode rows. Inactive heads and unavailable streams are forced to zero.

The architecture exposes diagnosis, variable-candidate action ranking, variable-support Born prediction, Hilbert-deformation, uncertainty, and topology-audit heads. It is deterministically initialized and identity-versioned, but explicitly records `trained=false`, no optimizer state, no training checkpoint, and `lambda_top=0`. See [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md).

## Phase 14 deterministic training engine

Phase 14 trains the Phase 13 graph model from completed Phase 12 views while preserving clean-circuit splits and per-head leakage masks. It provides train-only normalization, deterministic budget-aware batching, staged single-task/joint/hardware-masked curricula, AdamW or SGD, constant or warmup-cosine schedules, gradient accumulation and clipping, validation-based best-checkpoint selection, and exact resume.

Checkpoints are pickle-free NPZ artifacts containing model, optimizer, scheduler, and Python/NumPy/Torch RNG state with logical content hashes. Training and checkpoint manifests are typed-read before atomic publication. Test records and `audit_only` topology records never enter optimization, and `lambda_top` remains exactly zero. Phase 14 makes no held-out, hardware, universal-correction, or quantum-advantage claim. See [`docs/TRAINING_ENGINE.md`](docs/TRAINING_ENGINE.md).

## Current evidence level and claim boundaries

TriQTO is currently an offline deterministic research scaffold. The executable default path supports CPU-safe ideal-simulator tests and deterministic artifact/provenance checks. It does **not** establish quantum advantage, hardware validation, OOD generalization, calibrated uncertainty, or causal topology impact. Hardware Runtime work remains credential-gated future work and is not executed by default.

## Reproducible CPU installation

Supported default matrix: Python 3.11, Qiskit 2.1.2, Qiskit Aer 0.17.1, IBM Runtime client 0.40.1, Torch 2.8.0, NumPy 2.3.2, SciPy 1.16.1, PyArrow 21.0.0, Ripser 0.6.12, and Gudhi 3.11.0. The default dependency path is CPU-safe and excludes `qiskit-aer-gpu`.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints/cpu.txt
python -m pip install -e .
python scripts/verify_dependency_pins.py
PYTHONPATH=src pytest -q
```

Optional GPU dependencies are isolated in `requirements-gpu.txt` plus `constraints/gpu.txt`; do not use them for default CI or CPU-only validation.

## Config migration note

Broad future configs that mention noisy Aer, fake-backend/transpilation, RunPod, hardware validation, or unsupported actions are explicitly marked `unsupported: true` with a reason. Active configs are validated by `triqto.config.validators` and must not imply unimplemented execution modes. Old artifacts/configs that relied on `monster_generation.yaml`, `runpod_generation.yaml`, `hardware_validation.yaml`, or `configs/eval/heldout_*.yaml` as executable should be treated as planning inputs until the corresponding mode has implementation and offline tests.

See `docs/CAPABILITY_MATRIX.md` for the maintained capability matrix.

### Current evidence boundary update

The current branch includes offline fake-backend metadata propagation into Phase 7/12/14 artifacts, an executable deterministic fake-backend-axis holdout smoke/audit path, standalone operational actions, and checkpoint-bound latent-topology diagnostics. These are not physical-hardware results: no IBM Runtime job is submitted by default or in tests, fake-backend evidence remains simulator/fixture evidence, latent topology requires a real checkpoint identity supplied by a reproducible run, and no OOD or calibration claim is made without corresponding empirical artifacts.

## Capability-category status (2026-07-14)

The maintained category matrix is in [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md) and uses these exact categories: integrated into the primary pipeline, standalone executable API, credential-gated, empirically unvalidated, and planning-only/unsupported. In this repository state, fake-backend fixture evidence reaches Phase 7/12/14 model-training inputs with masks; noisy/density evidence remains a standalone API and does not enter the main data lake by default; no physical IBM hardware job has been submitted; no committed trained research checkpoint, calibrated-uncertainty result, backend-holdout performance report, or trained-representation latent-topology result exists; Phase 15 is present only as tested identity/generalization utility pieces rather than a completed empirical report publisher; and topology loss remains exactly zero.
