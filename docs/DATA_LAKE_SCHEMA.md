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
