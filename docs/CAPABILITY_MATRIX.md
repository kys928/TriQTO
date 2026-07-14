# TriQTO capability matrix

TriQTO currently provides an offline, deterministic research scaffold. It must not be described as quantum-advantage evidence, physical-hardware validation, broad OOD generalization, calibrated uncertainty, or topology-validated optimization.

Status categories used below are exact repository claim boundaries:

- **integrated into the primary pipeline**: executable through the current Phase 7/8/12/14 path and covered by tests.
- **standalone executable API**: implemented and tested as a callable/CLI boundary, but not yet part of the primary trained-result path.
- **credential-gated**: requires explicit external credentials and confirmation; not run by default tests.
- **empirically unvalidated**: code may execute, but no repository-trained checkpoint/result supports a paper-level empirical claim.
- **planning-only/unsupported**: config or design placeholder; active execution must fail closed with a machine-readable reason.

| Capability | Category | Current truth boundary |
| --- | --- | --- |
| Ideal statevector / Born simulation | integrated into the primary pipeline | Simulator-only Phase 7/8/12 evidence; hardware artifacts must reject Hilbert/statevector/exact-probability fields. |
| Sampled ideal shots | integrated into the primary pipeline | Offline ideal sampler evidence with deterministic seeds; not physical hardware. |
| Measurement settings and identifiability masks | integrated into the primary pipeline | Basis-conditioned `p(y | M)` records and target masks are executable for simulator data; unidentifiable targets remain masked. |
| Phase 8 graph artifacts | integrated into the primary pipeline | Graph/pair artifacts are deterministic and separate structural graph identity from sample ownership. |
| Phase 12 task-specific training views | integrated into the primary pipeline | Diagnosis, action-ranking, Born-prediction, optional Hilbert-to-Born, topology-audit, joint, and hardware-masked views enforce leakage masks. |
| Phase 14 deterministic training engine | integrated into the primary pipeline | Trains from Phase 12 views with train-only normalization and zero topology loss; test/audit-only rows do not fit normalization or select checkpoints. |
| Fake-backend metadata stream | integrated into the primary pipeline | Stable local fake-backend evidence reaches Phase 7/12/14 model inputs with availability masks and train-only normalization. This is fixture/simulator evidence, not hardware validation. |
| Backend-ID holdout utilities and audits | integrated into the primary pipeline | A strict fake-backend generation config, Phase 12 config, and Phase 15 audit config exercise an exact backend-ID axis holdout with clean-circuit assignment and train/validation-vs-test disjointness checks. Any result is limited to the recorded fake-backend fixture axis, not physical-hardware generalization. |
| Phase 15 comparison identity utilities | standalone executable API | Baseline comparison IDs include task/view/ablation/execution/evidence discriminators; full empirical Phase 15 report publication remains unmerged. |
| Phase 15 trained evaluation/reporting | planning-only/unsupported | No complete repository Phase 15 trained-result artifact is committed; IID must not be relabeled OOD. |
| Noisy Aer shots / density simulation | standalone executable API | Seeded helpers are tested for small circuits, but noisy/density evidence does not enter the main Phase 7 data lake by default. |
| Public Hilbert/QFI/QGT metrics | standalone executable API | Metric helpers are tested; physical hardware records must reject Hilbert-derived metrics. |
| Global-phase continuity | integrated into the primary pipeline | Hilbert encoders avoid largest-amplitude argmax anchoring; global phase is provenance, not a supervised shortcut. |
| IBM Runtime submission/collection boundary | credential-gated | Requires explicit confirmation and credentials; tests use doubles only and no real hardware call has been made. |
| Physical hardware result claims | empirically unvalidated | No physical hardware was used by this repository state. |
| Operational basis/layout/routing/depth actions | standalone executable API | Actions record preconditions, availability, before/after evidence, rejection reasons, and semantic-depth checks; compilation actions are not privileged inverse corrections. |
| Phase 9 logical correction actions | integrated into the primary pipeline | Synthetic inverse labels are limited to identifiable simulator distortions and remain distinct from operational compilation actions. |
| Latent persistent homology diagnostics | standalone executable API | Requires a nonblank checkpoint identity, point IDs, coordinate hash, split/head/config binding, and deterministic artifacts. |
| Topology loss | integrated into the primary pipeline | Enforced exactly `0.0`; topology remains diagnostic/audit-only. |
| Per-example uncertainty diagnostics | standalone executable API | Masked losses/diagnostics execute, but uncertainty calibration has not been demonstrated by a trained checkpoint/result. |
| Clean CPU install/import path | integrated into the primary pipeline | Supported profile is Python 3.11 CPU with pinned requirements/constraints; Python 3.14 observations are not validation of the supported profile. |
| Broad monster/RunPod/hardware configs | planning-only/unsupported | Marked `unsupported: true` with reasons; they must not imply active executable workflows. |

## Explicit current-state answers

- **Trained checkpoint/result exists:** no committed research checkpoint or paper-level result is present; tests may create temporary smoke checkpoints only.
- **Noisy/density evidence enters the main data lake:** no; it remains a standalone executable API.
- **Fake-backend evidence reaches model training:** yes, as offline fixture evidence with masks through the Phase 7/12/14 path.
- **Backend holdout has been executed:** yes for an executable deterministic fake-backend-axis smoke/audit path; no paper-level held-out performance claim is committed.
- **Physical hardware was used:** no.
- **Latent topology was run on trained representations:** no committed trained-representation topology result exists.
- **Uncertainty calibration was demonstrated:** no.
- **Phase 15 is merged:** only identity/generalization utility pieces are present; full empirical Phase 15 reporting is not merged as a completed result path.
- **Placeholders remain:** credential-gated hardware execution, broad monster/RunPod/hardware configs, and full empirical Phase 15 publication remain unsupported/planning-only boundaries.
- **Topology loss remains zero:** yes; nonzero topology loss is rejected.

## Warning audit

The remaining test warnings are project-external `stevedore`/Qiskit IBM Runtime plugin deprecation warnings emitted while Qiskit discovers installed transpiler plugins. They do not indicate credential use or hardware submission. Project-owned deprecation/unsafe-behavior warnings are not globally suppressed.
