# Phase 10 Baseline Suite

Phase 10 provides deterministic comparison controls for the TriQTO research program. It consumes the exact completed Phase 7 raw dataset, its completed Phase 8 graph dataset, and the completed Phase 9 action/rollout dataset derived from those sources. It does not train or evaluate a learned TriQTO model yet.

## Purpose

A future TriQTO policy must outperform meaningful alternatives rather than merely improve over an uncorrected circuit. Phase 10 therefore establishes six explicit controls:

1. `random_correction`
2. `rule_only`
3. `loss_only`
4. `spsa`
5. `cobyla`
6. `transpiler_only`

Every method is evaluated with the same exact clean-target Born objective in the fixed metric order:

1. total variation distance;
2. Jensen–Shannon divergence;
3. Hellinger distance.

The persisted scalar objective is a configured nonnegative weighted sum of those three lower-is-better values.

## Source contract

The Phase 10 loader validates the complete Phase 7 → Phase 8 → Phase 9 chain. It checks completion markers, managed-file inventories, scientific and operational IDs, typed manifests, graph/action joins, candidate QPY hashes, rollout NPZ hashes, and the Phase 7/8 snapshot hashes recorded by Phase 9. Every managed source file is byte-snapshotted before and after baseline execution and publication.

Phase 7 statevector artifacts are not loaded from disk. New exact evaluations use the existing in-memory circuit QPY artifacts and `qiskit.quantum_info.Statevector` through the ideal simulation layer.

## Baseline definitions

### Random correction

The random baseline selects one eligible Phase 9 action using a deterministic SHA-256 function of the configured seed and `sample_id`. By default it excludes no-op and every candidate carrying `oracle_inverse` provenance. Selection does not inspect clean-target metrics or distortion labels. If no eligible candidate exists, it falls back explicitly to no-op.

This is a sanity baseline. A future learned policy should beat it reliably.

### Rule-only correction

The rule-only baseline selects the smallest Phase 9 candidate carrying `oracle_inverse` provenance. Those inverse candidates were generated from privileged synthetic Phase 7 distortion metadata. Marker-only distortions receive no fabricated inverse and therefore fall back to no-op.

This is an intentionally privileged simulator control, not a deployable hardware diagnosis rule. It asks whether a future learned system can approach a known synthetic inverse when one exists.

### Loss-only selection

The loss-only baseline chooses the eligible Phase 9 candidate with minimum weighted clean-target Born loss. It ignores Phase 9 risk, depth, gate, and edit penalties. By default it excludes candidates carrying `oracle_inverse` provenance.

Because it consults the clean target during selection, it is an oracle action-selection upper control, not a hardware-available policy.

### SPSA

The SPSA baseline optimizes a fixed append-rotation parameterization using exact clean-target Born loss. Coordinates are RX, RY, and RZ on each logical qubit plus RZZ on two-qubit interactions observed in the distorted logical circuit. All coordinates are bounded by `max_abs_angle`. The perturbation stream is deterministic per sample.

The implementation returns the best point actually evaluated rather than trusting the final iterate blindly. It is still a clean-target simulator optimizer, not a learned policy and not a hardware result.

### COBYLA

The COBYLA baseline uses the same bounded logical parameterization and exact objective through `scipy.optimize.minimize(method="COBYLA")`. Bound constraints are explicit. The persisted result is the best evaluated point, with SciPy status and evaluation counts retained as metadata.

### Transpiler-only

The transpiler control runs `qiskit.transpile` on the distorted circuit without a backend, coupling map, correction action, or clean-target information. The clean target is used only afterward for evaluation. This checks ideal semantic preservation and compiler simplification.

It is deliberately labelled `hardware_aware = false`. Physical-layout and calibrated-backend transpiler comparisons remain deferred until the hardware-validation layer exists. Phase 10 does not invent backend connectivity or noise.

## Optimizer parameterization

For an `n`-qubit circuit the continuous baselines create coordinates in deterministic order:

- RX, RY, RZ for qubit 0;
- RX, RY, RZ for qubit 1;
- and so on;
- one RZZ coordinate for every sorted observed logical two-qubit edge.

Nonzero rotations are appended before final measurements using the existing safe measurement-removal/restoration contract. Unbound parameters, classically conditioned operations, control flow, and mid-circuit measurements remain unsupported and raise rather than being silently changed.

`max_optimizer_dimensions` and `max_objective_evaluations` are operational guardrails. They do not alter scientific identity when they are not exceeded, and exceeding them fails rather than truncating a run.

## Identity separation

`baseline_schema_id` versions the method names, selection semantics, optimizer parameterization, SPSA/COBYLA contracts, transpiler control, primary metric order, and result artifact schema.

`baseline_suite_id` depends on:

- Phase 7 scientific generation ID;
- Phase 8 graph conversion ID;
- Phase 9 action engine ID;
- baseline schema ID;
- enabled methods and scientific method hyperparameters.

The operational config ID additionally includes guardrails. A `baseline_result_id` depends on the sample, baseline name, suite ID, and result artifact version. Output paths and timestamps do not enter scientific identities.

## Output layout

Phase 10 writes one fresh immutable root:

```text
baseline_config.json
baseline_summary.json
baseline_complete.json
manifests/baseline_result_manifest.parquet
artifacts/results/<baseline_result_id>.npz
```

Each NPZ artifact loads with `allow_pickle=False` and contains exact arrays for metric names, before/after metric values, improvement values, output bitstrings, output probabilities, and any optimizer parameter vector. Strict UTF-8 JSON metadata is stored as a `uint8` array. Typed manifest readback validates IDs, references, hashes, sample coverage, graph-pair joins, selected Phase 9 action joins, objective arithmetic, probability normalization, and source identity.

The output root must not already exist and cannot be nested inside any source root. Publication occurs through a unique sibling staging directory followed by an atomic rename. Failure removes only the staging directory.

## Interpretation limits

Phase 10 does not yet compare a trained TriQTO policy because no learned model exists before Phases 13–14. The suite establishes reusable controls and immutable evidence for later evaluation.

It makes no claim of:

- learned correction;
- hardware-aware optimization;
- noisy-backend performance;
- generalization;
- topology benefit;
- universal correction;
- quantum advantage.

Rule-only, loss-only, SPSA, and COBYLA deliberately use forms of simulator privilege. Their metadata exposes that privilege so later reports cannot present them as deployment-equivalent methods.
