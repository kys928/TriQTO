# Phase 11 → Phase 12 topology attachment

This stage attaches validated Phase 11 topology feature vectors to an already
completed Phase 12 model-ready product. It publishes a new immutable output and
never rewrites Phase 11, Phase 12, or the source model-ready directory.

## Why this stage exists

Phase 11 persists fixed-order topology, alignment, parameter-manifold, and
Born-manifold feature vectors. The Phase 15.6 five-view publication omitted the
standalone `topology_audit` task, so its sample-level joint and hardware-masked
items received no Phase 11 topology arrays. This stage restores the missing
join without regenerating the expensive Phase 7→12 chain.

## Split policy

Every Phase 11 group is resolved to Phase 12 entities and their immutable split
assignments.

- An action-neighborhood group maps to the `sample_id` in its metadata.
- Cohort point IDs map directly to Phase 12 entity/sample IDs.
- A group with unresolved members is audit-only.
- A group spanning train, validation, and/or test is audit-only.
- Only same-split groups are eligible for attachment.

The dense per-entity stream uses the unique same-split action-neighborhood group.
Same-split cohort groups are retained in audit manifests rather than averaged
into a transductive model input.

## Leakage policy

Action-neighborhood topology is computed from exact candidate rollouts and Born
evidence. It therefore cannot be exposed to the action-ranking or Born-prediction
heads. The attachment stage enables it only for the joint diagnosis stream and
the unsupervised topology-audit projection. `lambda_top` remains exactly `0.0`,
and no `y_topology*` arrays are created.

Hardware-masked attachment is off by default. It can be explicitly enabled only
when Phase 11 declares `include_hilbert=false`; the action and Born heads remain
masked even then.

## Train-only scaling

Feature-name order is treated as a versioned contract. Unknown ordering or width
drift fails the run. Robust median/IQR scalers are fit on unique training
entities only. Count-like nonnegative features use `log1p` before robust
scaling. Nonfinite values are replaced by zero only after explicit finite and
positive/negative-infinity masks are materialized.

## Model arrays

Attached items receive:

- `x_topology_features`
- `x_topology_feature_mask`
- `x_topology_alignment_features`
- `x_topology_alignment_feature_mask`
- `x_topology_parameter_features`
- `x_topology_parameter_feature_mask`
- `x_topology_born_features`
- `x_topology_born_feature_mask`
- corresponding feature-name and infinity-mask arrays
- `x_topology_manifold_available_mask`

## Run

The runner uses environment variables rather than argparse:

```bash
TRIQTO_PHASE11_ROOT=/path/to/phase11 \
TRIQTO_PHASE12_ROOT=/path/to/phase12 \
TRIQTO_MODEL_READY_ROOT=/path/to/phase12_model_ready/run \
TRIQTO_TOPOLOGY_OUTPUT_ROOT=/path/to/phase12_model_ready_topology \
python scripts/attach_phase11_topology.py
```

The published root contains updated model artifacts and manifest plus:

- `manifests/topology_scalers.json`
- `manifests/topology_group_audit.parquet`
- `manifests/topology_entity_manifest.parquet`
- `reports/topology_attachment_report.json`
- `reports/topology_attachment_summary.md`
- `topology_attachment_complete.json`
