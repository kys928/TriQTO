# Failure Modes

TriQTO is intentionally vulnerable to several scientific and engineering failure modes. The architecture should expose them rather than hide them.

| Failure mode | Current mitigation or required test |
|---|---|
| Simulator-only overfitting | Hold out circuit families, qubit counts, distortions, and later backends; separate simulation and hardware-masked reporting. |
| Hilbert-field leakage into hardware mode | Phase 12 scrubs Hilbert references; Phase 13 rejects Hilbert tensors and Hilbert-dependent topology on hardware rows. |
| Born-target shortcut learning | Born-prediction views physically exclude Born inputs; the Phase 13 hard policy also forbids the Born stream for that head. |
| Topology-copy shortcut | The topology prediction head cannot consume topology input directly and has no supervised topology target in Phase 13. |
| Privileged oracle leakage | Candidate generation-source labels and oracle masks remain target/provenance data, not observable candidate features. |
| Fixed-qubit design failure | Concatenated graph batches, segment pooling, variable candidate sets, and variable Born support avoid a global qubit dimension. |
| Batch-composition dependence | Basis-position normalization uses each row's active qubit count rather than the maximum width of the batch. |
| Global-phase shortcut | Hilbert states are canonicalized by a deterministic reference amplitude before encoding. |
| Measurement shortcut instead of hidden structure | Use Born-blind phase cases, Hilbert-deformation tasks, held-out measurement bases, and ablations. |
| Topology features becoming decorative | Keep `lambda_top = 0`, log utilization, and require ablation evidence before activation. |
| Topology contaminating held-out splits | Cross-split topology cohorts remain `audit_only`. |
| Action policy becoming rule-only | Compare learned rankings with random, rule-only, loss-only, SPSA, COBYLA, and transpiler controls. |
| Candidate target leakage | Order candidates by stable ID, not target rank; keep rollout rewards/ranks out of inputs. |
| Inactive task heads producing accidental predictions | Phase 13 forces inactive latent states, fusion weights, and outputs to zero. |
| Missing-data masks becoming hidden labels | Audit fusion weights and performance by availability pattern; use controlled mask-dropout and held-out mask patterns in Phase 14. |
| Poor generalization to held-out circuit families | Dedicated held-out-family evaluation. |
| Poor generalization to larger qubit counts | Train across sizes and evaluate on unseen qubit counts without changing model dimensions. |
| Poor generalization to held-out backends | Backend-held-out evaluation once real/fake backend data exists. |
| Shot noise overwhelming Born metrics | Train/evaluate exact and shot-derived evidence separately; calibrate uncertainty. |
| Fake backend mismatch with real hardware | Treat fake-backend results as simulation and require Phase 16 real-device validation. |
| Too-expensive statevector storage | Hilbert references remain optional; hardware mode does not depend on them. |
| Exponential full-support Born prediction | Use sparse or queried support for larger circuits and report support coverage. |
| Topology computation becoming too slow | Phase 11 guardrails fail instead of silently subsampling; cache/version point-cloud audits. |
| Dense topology feature-map drift | Phase 14 must version the feature-name-to-index mapping and reject unknown/missing required features. |
| Uncertainty head becoming uncalibrated | Evaluate calibration error, coverage, and selective-risk behavior in Phase 15. |
| Architecture identity confused with initialization | `model_architecture_id` excludes name/seed; `model_config_id` and exact initialized-state signature identify initialization. |
| Initialized weights mislabeled as trained | Phase 13 manifest records `trained=false`, no optimizer state, and no training checkpoint. |
| Baselines outperforming TriQTO | Report it directly; do not redefine success after seeing results. |
| Quantum-advantage overclaim | The current system is a classical model around quantum-circuit data and makes no quantum-advantage claim. |
