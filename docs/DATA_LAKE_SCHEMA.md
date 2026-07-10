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
