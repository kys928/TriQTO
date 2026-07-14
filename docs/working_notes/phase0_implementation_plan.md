# TriQTO staged implementation plan

- [x] Phase 0 — Reproducibility, CI, and claim boundaries. Completed in the Phase 0 dependency/config-boundary commit on this branch.
- [x] Phase 1 — Measurement context and identifiability, initial integrated pass. Added first-class basis-conditioned ideal probabilities `p(y|M)`, per-qubit Z/X/Y basis codes through simulation/generation/graph artifacts/model Born contracts, identifiability status/reasons, supervision masks, strict rejection, and adversarial tests for marker/phase-blind targets.
- [ ] Phase 2 — Partial standalone noisy/density execution added. Seeded Aer noisy shots and density-matrix helpers are tested; fake-backend/transpilation and Runtime boundary remain pending.
- [x] Phase 3 — Initial public metrics and global-phase continuity. Added pure/density metrics, finite-difference pure-state QGT/QFI, and replaced argmax phase canonicalization with a continuous soft-phasor anchor. Geometry/topology causal claims remain unsupported.
- [ ] Phase 4 — Operational actions.
- [x] Phase 5 — Initial OOD holdout evaluation and Phase 15 baseline identity repair. Added standalone deterministic axis holdout/audit utilities, executable holdout configs, IID split labeling, and comparison IDs including task/view/ablation/execution mode. Full Phase 15 evaluator remains a later integration boundary.
- [ ] Phase 6 — Partial per-example uncertainty objective and direct diagnostics added. Calibration remains empirically unvalidated without a trained calibrated checkpoint.
- [ ] Phase 7 — Latent topology and final reporting honesty.

This plan is intentionally conservative: later phases remain unsupported until the code and tests in this branch establish them.
