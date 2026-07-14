# TriQTO Data Lake Schema

The data lake supports the whole research program from day one while allowing task-specific training views to select only needed fields.

## Record families
- Circuit records
- Measurement-setting records
- Backend records
- Simulation records
- Distortion records
- Metric records
- Action candidate records
- Topology records
- Training view records

## Manifest-centered organization
Future manifests are expected at `data/manifests/`:
- `circuit_manifest.parquet`
- `simulation_manifest.parquet`
- `measurement_setting_manifest.parquet`
- `distortion_manifest.parquet`
- `metric_manifest.parquet`
- `action_manifest.parquet`
- `topology_manifest.parquet`
- `backend_manifest.parquet`
- `split_manifest.parquet`

Large tensors should be referenced by path or URI rather than embedded in manifests. Candidate formats include Zarr, HDF5, NumPy arrays, and Parquet tables.

## Phase 7 raw data generation pipeline

Phase 7 implements a deterministic raw research-data lifecycle:

`CircuitGenerationSpec -> circuit family generator -> deterministic parameter binding -> clean ideal statevector simulation -> explicit measurement settings M -> controlled distortion -> distorted p(y | M) -> setting-conditioned Born metric comparison -> identifiability assessment -> deterministic IDs -> artifacts -> linked manifests`.

The generated data is synthetic, simulator-derived raw research data. It is not real hardware data. Unitary distortions are controlled circuit-level interventions rather than calibrated hardware noise. Readout bit flips are implemented as an observable classical measurement channel. Layout markers remain audit-only and unidentifiable because backend/layout evidence is unavailable. A setting-specific `born_zero_shift` means no selected measurement distribution changed beyond tolerance; it does not prove Hilbert-state equality. Statevectors and exact probabilities are simulation-only artifacts. No correction ability has been learned, no training has occurred, and no quantum advantage is claimed.

### Sample manifest and joins

`manifests/sample_manifest.parquet` stores `DatasetSampleRecord` rows. Each row joins:

- `clean_circuit_id` and `distorted_circuit_id` in `circuit_manifest.parquet`
- `clean_run_id` and `distorted_run_id` in `simulation_manifest.parquet`
- ordered `measurement_setting_ids`, clean measurement run IDs, and distorted measurement run IDs in `measurement_setting_manifest.parquet` and `simulation_manifest.parquet`
- `distortion_id` in `distortion_manifest.parquet`
- `metric_id` in `metric_manifest.parquet`

Each sample also stores identifiability status/reason, a default diagnosis supervision mask, and an observable-evidence fingerprint that excludes labels and provenance.

This join layer is a raw-sample manifest only. It is not a `TrainingViewRecord`, does not contain train/validation/test allocations, and does not create final training views. Training-view allocation remains Phase 12.

### Artifact layout

Phase 7 writers use relative POSIX references from the output root:

```text
generation_config.json
dataset_summary.json
manifests/sample_manifest.parquet
manifests/circuit_manifest.parquet
manifests/simulation_manifest.parquet
manifests/measurement_setting_manifest.parquet
manifests/distortion_manifest.parquet
manifests/metric_manifest.parquet
artifacts/circuits/<circuit_id>.qpy
artifacts/statevectors/<run_id>.npy
artifacts/probabilities/<run_id>.json
artifacts/counts/<run_id>.json
```

Manifests store references to artifacts, not large statevectors inline. Exact Born probabilities come from ideal statevector simulation and are compared separately for every declared measurement setting. Optional ideal-shot counts are setting-conditioned, supplemental, and never replace exact probabilities.

### Non-finite Born metric serialization

Strict JSON does not allow `NaN` or infinity. Phase 7 metric serialization keeps finite values as floats. Positive infinity, for example KL divergence when distorted support has zero mass where clean support is positive, is encoded as:

```json
{"kl_divergence": null, "kl_divergence__nonfinite": "positive_infinity"}
```

This preserves the non-finite value explicitly and reversibly without capping it.

### Scope boundary

Phase 7 does not implement graph conversion, graph neural networks, correction actions, candidate repairs, baselines, topology, persistent homology, training-view splitting, model architecture, model training, noisy simulation, fake backend simulation, IBM Runtime, real hardware calls, Hilbert metrics, fidelity, Fubini-Study distance, trace distance, QFI, QGT, evaluation reports beyond a generation summary, or quantum-advantage claims. Those systems are future phases; the next recommended phase is Phase 8 graph conversion.

### Phase 7 identity separation patch

Phase 7 now distinguishes scientific identity from operational configuration identity:

- `scientific_generation_id` describes the exact simulator-derived scientific sample universe: schema version, base seed, circuit specs, distortion specs, parameter range, and Born metric schema/version.
- `config_id` describes the complete operational configuration, including human dataset label and artifact/shot/test-guard options.
- Clean circuit-generation seeds, parameter-binding seeds, clean circuit IDs, exact statevector run IDs, distorted exact run IDs, metric IDs, and sample IDs are derived from scientific payloads only. They do not depend on output paths, dataset names, optional statevector storage, optional shot sampling, or `max_samples` when it is not exceeded.
- Adding a new distortion does not change existing clean circuit payloads, clean seeds, parameter bindings, clean circuit IDs, or existing clean/distorted comparison sample IDs.

`born_zero_shift` and `born_observable_shift_absent` are labels computed with the configured `born_zero_atol` tolerance (`1e-12` by default): total variation values less than or equal to this tolerance are labeled Born-zero-shift. The exact metric value is preserved and is not rounded or replaced. The tolerance is stored in sample metadata and the dataset summary.

Dataset writing returns populated artifact path categories (`circuits`, `probabilities`, `statevectors`, and `counts`) and verifies all manifest references with explicit exceptions instead of Python assertions. Circuit persistence lazily requires Qiskit QPY support; if QPY is unavailable, in-memory generation remains usable but `write_dataset()` raises a clear runtime error. Ideal shot records use the `triqto.ideal_probability_sampler` backend label and identify their source exact statevector run.

### Phase 7 final persistence and integrity rules

Metric manifests preserve application-level empty deferred metric maps. Because some Parquet backends cannot write columns whose every value is an empty struct, Phase 7 may physically store deferred metric-map columns as null and records metadata with `empty_metric_map_storage_encoding = "parquet_null_normalized_to_empty_dict"`. Typed readback through `ManifestReader.read_typed_records(..., MetricRecord)` normalizes those fields back to `{}` so application code never receives `None` for metric-family maps.

Artifact references are required by record type and simulation mode. Circuit records require `metadata["artifact_ref"]`. `ideal_statevector` simulation records require `probabilities_ref`; they require `statevector_ref` exactly when `store_statevectors` is true and must omit it when statevector storage is disabled. `ideal_shot` simulation records require `counts_ref`, `metadata["source_run_id"]`, and `metadata["sampling_source"] == "sampled_from_exact_born_probabilities"`; they must omit probability and statevector references. Unknown simulation modes are rejected by Phase 7 verification.

Artifact references must be non-empty normalized relative POSIX paths resolving inside the dataset output root. Absolute paths, empty references, `.` references, path traversal such as `../outside.json`, nested traversal such as `artifacts/../../outside.json`, symlink/root escapes, and directory targets are rejected.

Born metric serialization accepts finite numeric values and explicitly encodes positive infinity as `null` plus a `metric__nonfinite = "positive_infinity"` marker. NaN, negative infinity, malformed non-finite markers, orphan markers, and unmarked null metric values are rejected instead of being silently coerced.

Generation configuration parsing is strict: bool-as-number values, numeric strings, null numeric fields, NaN, Infinity, and -Infinity are rejected. `store_statevectors` must be an actual boolean. Count JSON artifacts are exact deterministic ideal-probability-sampler outputs and are read back against the in-memory count dictionaries, not only checked by shot totals. QPY circuit artifacts are expected to roundtrip with semantic circuit fingerprints including global phase, parameters, operation parameters, qubit operands, classical operands, and measurement wiring.

`write_dataset()` writes into a sibling staging directory and validates artifacts/manifests before publishing. A `dataset_complete.json` marker is written only after successful validation. On failure, the staging directory created by that call is removed; when overwriting, only known Phase 7 dataset outputs are replaced and unrelated user files under the output directory are preserved. A dataset without `dataset_complete.json` must be treated as incomplete.

### Phase 7 transactional publication ownership

Phase 7 owns exact managed files, not entire `artifacts/` or `manifests/` directories. Those directories may contain unrelated user files. The completion marker records a sorted `managed_files` inventory containing the exact files owned by the committed Phase 7 dataset, including manifests, JSON summaries, QPY circuit artifacts, probability JSON artifacts, optional statevector NPY artifacts, optional count JSON artifacts, and `dataset_complete.json` itself.

Overwrite replaces or removes only paths listed in the previous managed-file inventory plus exact known legacy Phase 7 files. It does not recursively delete `output_root`, `artifacts/`, `manifests/`, or nested directories that may contain unrelated content. Managed paths are validated with the same root-escape checks used for artifact references; absolute paths and traversal are rejected.

Publication is transactional at the file level. A complete dataset is first built and typed-read back in a unique staging directory. Persisted typed manifests are validated for duplicate IDs, missing joins, semantic joins, required references, and path safety before any completion marker exists. During overwrite, previous managed files are moved to a unique backup directory. New managed files are published in deterministic order, final-root persisted manifests/references are validated again, and `dataset_complete.json` is written last via a temporary-file atomic replacement. If publication fails before that final commit point, newly published managed files are removed, previous managed files are restored from backup, staging/backup directories are removed, and unrelated files are left untouched.

Typed `MetricRecord` readback treats null `born_metrics` or null/malformed `metadata` as corruption and raises. Only deferred metric maps may use the documented null-to-empty storage encoding, and only when `metadata["empty_metric_map_storage_encoding"] == "parquet_null_normalized_to_empty_dict"`. Duplicate manifest IDs and missing joins are integrity errors with record IDs in the error messages.

## Phase 8 graph conversion artifacts

Phase 8 consumes one completed Phase 7 root and writes a separate immutable graph root:

```text
graph_config.json
graph_summary.json
graph_complete.json
manifests/graph_manifest.parquet
manifests/graph_pair_manifest.parquet
artifacts/graphs/<graph_id>.npz
artifacts/pairs/<graph_pair_id>.npz
```

`GraphRecord` is circuit/run-level and contains no authoritative sample owner. Its identity includes the source circuit, exact statevector run, clean/distorted role, and versioned graph schema. `GraphPairRecord` is sample-level and joins one Phase 7 sample to clean/distorted graph IDs, distortion ID, and metric ID. Reused clean graphs carry only sorted sample provenance metadata; sample provenance is excluded from graph identity and structural hashing.

Graph NPZ artifacts contain variable-size qubit nodes, directed two-qubit event multiedges, ordered gate events, CSR operand/parameter incidence arrays, deterministic logical dependency layers, exact Born outcome tables, and optional separate supplemental count tables. Arrays use fixed dtypes, contain no Python objects, and load with `allow_pickle=False`. Operations on more than two qubits remain incidence events and do not produce fabricated pairwise cliques. Physical backend coupling edges are not invented from ideal simulation data.

Exact Born evidence is strict and authoritative. Bitstrings are sorted unique binary strings with circuit width; probabilities reject booleans, strings, NaN, and infinity. Tiny positive mass is preserved without thresholding or truncation. Ideal-shot counts are linked through the shot `SimulationRecord.metadata["source_run_id"]`, checked against their exact run and shot total, and remain supplemental. Excluding supplemental counts does not alter graph IDs, pair IDs, exact Born arrays, or structural graph hashes.

Structural graph hashes include the versioned representation, fixed feature ordering, circuit structure, parameters, and exact Born evidence. They exclude graph IDs, sample provenance, source/output paths, statevector references, supplemental counts, timestamps, and operational guardrails. Pair artifacts have separate deterministic content hashes covering pair identities, labels, applicability warnings, and Born metric arrays. Applicability warnings come from `MetricRecord.metadata`.

The Phase 7 source completion marker, scientific generation ID, operational config ID, sample count, manifest count, managed inventory, typed joins, and artifact references are cross-checked. Every managed source file is byte-hashed before and after conversion. Statevector arrays are never loaded for graph features; their references remain provenance only and `hilbert_available_mask` remains false.

Both Phase 8 manifests are typed-read before publication. Graph and pair IDs are unique, graph-pair joins are explicit, artifact dimensions agree with manifests, and all logical hashes are recomputed. The actual `GraphConversionConfig` used for conversion is persisted in `graph_config.json`.

Phase 8 output roots are immutable: the final root must not exist. The complete dataset is created and validated in a unique sibling staging directory, `graph_complete.json` records the source generation ID, graph conversion/config/schema IDs, source snapshot hash, counts, and sorted managed-file inventory, and the staging directory is atomically renamed. Failure removes only that staging directory and cannot leave a partial final root.

Phase 8 remains representation only. It adds no graph neural network, correction action, policy, baseline, topology, persistent homology, training split, model training, noisy backend, hardware call, Hilbert metric, or quantum-advantage claim.

## Phase 9 action candidate and rollout artifacts

Phase 9 consumes the completed Phase 7 raw dataset together with its exact completed Phase 8 graph dataset. Both managed inventories are byte-snapshotted before and after action generation. Their completion identities, source-snapshot relationship, typed manifests, graph artifacts, pair artifacts, and semantic joins are cross-validated.

The immutable Phase 9 output contains:

```text
action_config.json
action_summary.json
action_complete.json
manifests/action_candidate_manifest.parquet
manifests/action_rollout_manifest.parquet
artifacts/actions/<action_id>.json
artifacts/circuits/<candidate_circuit_id>.qpy
artifacts/rollouts/<rollout_id>.npz
```

`ActionCandidateRecordV1` links a sample and graph pair to one bounded edit definition and its validated candidate QPY circuit. `ActionRolloutRecord` links that candidate to exact ideal-statevector Born evidence, transparent reward, deterministic rank, and selected/non-worsening labels. Exactly one rank-one rollout is selected per source sample.

Action identity is based on source sample/graph/circuit/run and ordered edit content, not file paths or generation-source provenance. The complete action artifact hash additionally protects its deterministic generation labels, risk, and metadata. Rollout identity includes the clean target run and scientific action/reward configuration. Operational candidate/edit guardrails are excluded from scientific identity and fail rather than truncate when exceeded.

Synthetic oracle inverses use privileged Phase 7 distortion metadata and are stored only as supervised synthetic labels. They are not a learned policy, a hardware diagnosis rule, or evidence of real-device correction. Phase 9 v1 supports no-op and bounded RX/RY/RZ/RZZ circuit edits, exact ideal simulation, and Born-space scoring only. No baseline comparison, noisy backend, hardware execution, topology, training split, model, or quantum-advantage claim is introduced.

Empty dictionaries at any manifest nesting depth are encoded with a reserved Parquet sentinel and decoded back to exact application-level `{}` values. The sentinel is an internal storage workaround for PyArrow's inability to write a struct with zero child fields; user data containing the reserved sentinel key is rejected.
