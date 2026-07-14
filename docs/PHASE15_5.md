# Phase 15.5 — noisy operational policy and empirical smoke benchmarking

Phase 15.5 is an offline bridge between the exact/fake-backend engineering pipeline and any future credential-gated hardware pilot. It does not submit IBM Runtime jobs and does not convert simulator evidence into a hardware-validation claim.

## Executable path

The workflow binds to immutable completed Phase 7, Phase 12, and Phase 14 sources plus a positive-step trained Phase 14 checkpoint. It then:

1. restores the checkpoint and extracts diagnosis-head latent context without gradients;
2. generates seeded noisy-Aer shot evidence under explicit X, Y, and Z measurement settings;
3. optionally records density-matrix probability and purity summaries;
4. builds separate operational candidate groups for basis probes, layout selection, routing/transpilation, and semantics-preserving depth optimization;
5. creates family-specific simulator supervision while excluding clean-pair target evidence from policy inputs;
6. trains a separate deterministic operational policy on Phase 12 train groups and selects its checkpoint using validation groups only;
7. evaluates untouched Phase 12 test groups against random, no-op, family-heuristic, and oracle-upper-bound controls;
8. publishes grouped bootstrap confidence intervals, per-family summaries, failure cases, content-bound policy artifacts, and a completion marker.

Run the standalone workflow with completed sources:

```bash
python scripts/run_phase15_5.py \
  --phase7-root /path/to/phase7 \
  --training-view-root /path/to/phase12 \
  --training-root /path/to/phase14 \
  --checkpoint /path/to/final-checkpoint.npz \
  --output /tmp/triqto-phase15-5 \
  --config configs/eval/phase15_5_smoke.json
```

The complete engineering smoke workflow also executes Phase 15.5:

```bash
python scripts/run_cpu_smoke_workflow.py --output /tmp/triqto-smoke
```

## Supervision and leakage boundary

Operational families do not share the Phase 9 logical-correction target.

- **Basis probe:** utility is based on basis-conditioned diagnostic separation minus an explicit probe cost. It is evidence acquisition, not a correction.
- **Layout and routing:** utility uses backend-bound compilation evidence and cost deltas. These are semantics-preserving compilation choices, not logical inverses.
- **Depth optimization:** utility is nonzero only when semantic validation and the configured objective support acceptance.
- **No-op:** each family receives an explicit no-op candidate so that intervention is not forced.

Privileged clean/noisy pairs are permitted only for offline simulator target construction. Their probabilities and target metrics are not policy input features. Phase 12 test groups never enter optimization or checkpoint selection.

## Artifacts

A successful run atomically publishes:

- `phase15_5_config.json`
- `noisy_evidence.jsonl`
- `operational_supervision.jsonl`
- `policy_dataset.npz`
- `policy_dataset_schema.json`
- `training_history.json`
- `operational_policy_checkpoint.npz`
- `operational_policy_checkpoint.json`
- `benchmark_report.json`
- `phase15_5_complete.json`

The checkpoint is pickle-free, content-hashed, source-bound, explicitly marked offline, and rejects nonzero topology loss. The result loader verifies the managed inventory, file hashes, checkpoint identity, claim boundaries, and report/completion joins.

## Claim boundary

Phase 15.5 demonstrates that the repository can execute a deterministic noisy-simulation operational-policy experiment. It does not demonstrate:

- physical-hardware performance or calibration transfer;
- broad out-of-distribution generalization;
- calibrated uncertainty;
- universal correction or fault tolerance;
- quantum advantage;
- topology benefit or causal topology value;
- research-quality empirical superiority.

The smoke benchmark is intentionally small. Reported trained-policy differences from controls are engineering-test outputs, not paper-level evidence. `topology_loss_weight` remains exactly `0.0`.
