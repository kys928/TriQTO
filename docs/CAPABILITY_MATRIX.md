# TriQTO capability matrix

TriQTO currently provides an offline, deterministic research scaffold. It must not be described as quantum-advantage evidence, physical-hardware validation, broad OOD generalization, calibrated uncertainty, or topology-validated optimization.

Status categories used below are exact repository claim boundaries:

- **integrated into the primary pipeline**: executable through the current Phase 7/8/12/14/15 path or its validated Phase 15.5/15.6 extensions and covered by tests.
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
| Phase 12 task-specific training views | integrated into the primary pipeline | Diagnosis, action-ranking, Born-prediction, optional Hilbert-to-Born, topology-audit, joint, and hardware-masked views enforce leakage masks. Operational actions use separately versioned adapters/targets because they do not share logical-correction supervision. |
| Phase 14 deterministic training engine | integrated into the primary pipeline | Trains the Phase 13 model from Phase 12 views with train-only normalization and zero topology loss; test/audit-only rows do not fit normalization or select checkpoints. |
| Fake-backend metadata stream | integrated into the primary pipeline | Stable local fake-backend evidence reaches Phase 7/12/14 model inputs and Phase 15.5 compilation targets. This is fixture/simulator evidence, not hardware validation. |
| Backend-ID holdout utilities and audits | integrated into the primary pipeline | A strict fake-backend generation config, Phase 12 config, and Phase 15 audit config exercise an exact backend-ID axis holdout with clean-circuit assignment and train/validation-vs-test disjointness checks. Any result is limited to the recorded fake-backend fixture axis, not physical-hardware generalization. |
| Phase 15 trained evaluation/reporting | integrated into the primary pipeline | A CPU smoke evaluator restores a Phase 14 checkpoint, scores untouched Phase 12 test rows, publishes immutable compact manifests/cards, and can re-audit exact fake-backend holdout disjointness. This is engineering validation only. |
| Operational basis/layout/routing/depth actions | integrated into the primary pipeline | Typed immutable operational artifacts, strict configs, candidate masks, smoke execution, and family-specific reporting are executable. Basis probes acquire evidence; layout/routing/depth actions are compilation or semantics-preserving operations; none is treated as a privileged logical inverse. |
| Noisy Aer shots and density summaries | integrated into the primary pipeline | Phase 15.5 generates seeded basis-conditioned noisy-shot evidence and optional density summaries as an immutable extension bound to completed Phase 7/12/14 sources. It does not alter the exact Phase 7 scientific-generation identity and is not physical hardware. |
| Operational action policy training | integrated into the primary pipeline | Phase 15.5 trains a separate family-conditioned policy with matched simulator targets, train-only normalization, validation checkpoint selection, untouched grouped test evaluation, and pickle-free content-bound artifacts. Privileged clean-pair target evidence is excluded from policy inputs. |
| Operational-policy empirical benefit | empirically unvalidated | The small CPU smoke benchmark compares trained, random, no-op, family-heuristic, and oracle-upper-bound choices with grouped bootstrap intervals. It is engineering validation, not evidence of research-scale superiority, broad OOD generalization, or hardware transfer. |
| Phase 15.6 pod/campaign orchestration | integrated into the primary pipeline | Strict planning, preflight, immutable config snapshots, external-workspace enforcement, staged data construction, multi-seed training/evaluation, locks, completed-stage reuse, and cross-seed aggregation are executable. The repository does not launch pods or run the expensive campaign automatically. |
| Phase 15.6 research pilot result | empirically unvalidated | A 13,440-sample, 2–8-qubit, three-seed pilot is configured, but no research dataset, checkpoint, or cross-seed result has been generated or committed by the repository. |
| Phase 9 logical correction actions | integrated into the primary pipeline | Synthetic inverse labels are limited to identifiable simulator distortions and remain distinct from operational compilation actions. |
| Checkpoint-derived latent extraction | integrated into the primary pipeline | The CPU smoke workflow restores a real positive-step Phase 14 checkpoint, verifies checkpoint/model/Phase-12 identities, reads an explicit split without gradients, preserves ordered view-item IDs, and publishes immutable latent-coordinate artifacts. |
| Checkpoint-bound latent persistent homology | integrated into the primary pipeline | Persistent homology consumes only validated latent-extraction artifacts and binds identity to checkpoint bytes, model identities, Phase 12 source, split/head/representation, ordered points, coordinate hash, and topology config. Absolute scale is default; optional shape-only normalization has a distinct identity. |
| Topology benefit / causal value | empirically unvalidated | Checkpoint-bound latent topology is diagnostic only. No performance benefit, causal value, calibration, hardware transfer, or optimization success has been demonstrated. |
| Public Hilbert/QFI/QGT metrics | standalone executable API | Metric helpers are tested; physical hardware records must reject Hilbert-derived metrics. |
| Global-phase continuity | integrated into the primary pipeline | Hilbert encoders avoid largest-amplitude argmax anchoring; global phase is provenance, not a supervised shortcut. |
| IBM Runtime submission/collection boundary | credential-gated | Requires explicit confirmation and credentials; tests use doubles only and no real hardware call has been made. |
| Physical hardware result claims | empirically unvalidated | No physical hardware was used by this repository state. |
| Topology loss | integrated into the primary pipeline | Enforced exactly `0.0`; topology remains diagnostic/audit-only. |
| Per-example uncertainty diagnostics | standalone executable API | Masked losses/diagnostics execute, but uncertainty calibration has not been demonstrated by a trained checkpoint/result. |
| Clean CPU install/import path | integrated into the primary pipeline | Supported profile is Python 3.11 CPU with pinned requirements/constraints; Python 3.14 observations are not validation of the supported profile. |
| Legacy broad monster/hardware configs | planning-only/unsupported | Legacy broad configs remain marked unsupported. The specific Phase 15.6 pod campaign path is separately versioned, executable, and does not enable hardware. |

## Explicit current-state answers

- **Trained checkpoint/result exists:** temporary CPU smoke checkpoints/results are created by tests or user-selected workflow output directories; no research checkpoint, weights, or result artifacts are committed.
- **Noisy/density evidence enters the exact Phase 7 data lake:** no. Phase 15.5 creates a separate immutable extension so clean Phase 7 identities are not silently changed.
- **Noisy/density evidence enters an executable trained path:** yes, through the offline Phase 15.5 operational-policy workflow.
- **Fake-backend evidence reaches model training:** yes, as offline fixture evidence with masks through the Phase 7/12/14 path; Phase 15.5 also uses fixture-bound compilation evidence.
- **Backend holdout has been executed:** yes as an executable deterministic fake-backend-axis smoke/audit evaluation path only; no paper-level held-out performance claim is committed.
- **Operational actions are integrated:** yes for immutable generation, masking/batching, noisy simulator target construction, separate policy training, grouped test benchmarking, and family-specific reporting.
- **Operational-policy benefit was demonstrated:** only as a small deterministic smoke/engineering result. No research-scale superiority, broad OOD, calibration, or hardware-transfer claim is made.
- **Phase 15.6 environment is executable:** yes for pod bootstrap, preflight, campaign preparation, staged execution, multi-seed runs, and aggregation.
- **The Phase 15.6 research pilot was run:** no. The repository contains the campaign definition and runner only; the user must run the expensive stages on external compute.
- **Physical hardware was used:** no.
- **Latent topology was run on trained representations:** yes in the temporary deterministic CPU smoke workflow using a restored positive-step checkpoint and an explicit Phase 12 split. No trained-representation topology artifact is committed and no topology-benefit claim is made.
- **Uncertainty calibration was demonstrated:** no.
- **Phase 15.5 is executable:** yes as an offline noisy-simulation operational-policy and grouped benchmark extension; it is not Phase 16 hardware validation.
- **Placeholders remain:** credential-gated hardware execution, legacy broad monster/hardware configs, calibrated uncertainty, expanded held-out-axis campaigns, and full paper-level publication remain unsupported or empirically unvalidated boundaries.
- **Topology loss remains zero:** yes; nonzero topology loss is rejected.

## Warning audit

The remaining test warnings, when present, are project-external `stevedore`/Qiskit IBM Runtime plugin deprecation warnings emitted while Qiskit discovers installed transpiler plugins. They do not indicate credential use or hardware submission. Project-owned deprecation/unsafe-behavior warnings are not globally suppressed.
