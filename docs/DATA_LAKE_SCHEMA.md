# TriQTO Data Lake Schema

The data lake supports the whole research program from day one while allowing task-specific training views to select only needed fields.

## Record families
- Circuit records
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
- `distortion_manifest.parquet`
- `metric_manifest.parquet`
- `action_manifest.parquet`
- `topology_manifest.parquet`
- `backend_manifest.parquet`
- `split_manifest.parquet`

Large tensors should be referenced by path or URI rather than embedded in manifests. Candidate formats include Zarr, HDF5, NumPy arrays, and Parquet tables.

## Phase 7 raw data generation pipeline

Phase 7 implements a deterministic raw research-data lifecycle:

`CircuitGenerationSpec -> circuit family generator -> deterministic parameter binding -> clean ideal statevector simulation -> controlled distortion -> distorted ideal statevector simulation -> Born metric comparison -> deterministic IDs -> artifacts -> linked manifests`.

The generated data is synthetic, simulator-derived raw research data. It is not real hardware data. Unitary distortions are controlled circuit-level interventions rather than calibrated hardware noise. Marker-only readout/layout records do not simulate physical effects and must not fabricate Born-probability changes. A `born_zero_shift` label means no computational-basis Born shift was observed; it does not prove Hilbert-state equality. Statevectors are simulation-only artifacts. No correction ability has been learned, no training has occurred, and no quantum advantage is claimed.

### Sample manifest and joins

`manifests/sample_manifest.parquet` stores `DatasetSampleRecord` rows. Each row joins:

- `clean_circuit_id` and `distorted_circuit_id` in `circuit_manifest.parquet`
- `clean_run_id` and `distorted_run_id` in `simulation_manifest.parquet`
- `distortion_id` in `distortion_manifest.parquet`
- `metric_id` in `metric_manifest.parquet`

This join layer is a raw-sample manifest only. It is not a `TrainingViewRecord`, does not contain train/validation/test allocations, and does not create final training views. Training-view allocation remains Phase 12.

### Artifact layout

Phase 7 writers use relative POSIX references from the output root:

```text
generation_config.json
dataset_summary.json
manifests/sample_manifest.parquet
manifests/circuit_manifest.parquet
manifests/simulation_manifest.parquet
manifests/distortion_manifest.parquet
manifests/metric_manifest.parquet
artifacts/circuits/<circuit_id>.qpy
artifacts/statevectors/<run_id>.npy
artifacts/probabilities/<run_id>.json
artifacts/counts/<run_id>.json
```

Manifests store references to artifacts, not large statevectors inline. Exact Born probabilities come from ideal statevector simulation and are the target for Phase 7 Born metrics. Optional ideal-shot counts are supplemental and never replace exact probabilities.

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

Phase 8 consumes a completed Phase 7 root and writes a separate immutable graph root containing `graph_config.json`, `graph_summary.json`, `graph_complete.json`, `manifests/graph_manifest.parquet`, `manifests/graph_pair_manifest.parquet`, `artifacts/graphs/<graph_id>.npz`, and `artifacts/pairs/<graph_pair_id>.npz`. Graph references are relative POSIX paths. NPZ payloads use non-object arrays loadable with `allow_pickle=False`.

Graph records describe variable-size qubit-node circuit graphs and graph-pair records describe clean/distorted sample links. They are representation records only: no train/validation/test allocation, action label, topology feature, learned embedding, Hilbert feature vector, physical coupling graph, or model output is stored.
