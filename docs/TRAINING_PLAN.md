# Training Plan

Phase 12 now materializes deterministic task-specific views for:

- distortion diagnosis;
- action ranking;
- Born prediction;
- optional Hilbert-to-Born prediction;
- topology audit;
- joint multitask learning;
- hardware-masked simulation training.

The view layer does not train a model. It defines split groups, source references, materialized structural arrays, targets, availability masks, per-head masks, and leakage exclusions.

## Split policy

All records derived from the same `clean_circuit_id` stay in one deterministic train, validation, or test split. Individual distortions or candidate actions are never randomly split away from their clean-circuit lineage. Topology cohorts spanning several source splits remain `audit_only`.

## Masking policy

Hilbert inputs are optional. A missing statevector produces no Hilbert-to-Born item and a false Hilbert mask in joint views. Hardware-masked simulation removes every Hilbert input reference and excludes topology that was computed using Hilbert access.

Born-prediction items materialize graph structure and parameter/phasor arrays without exact Born input evidence. Joint items carry per-head masks so the diagnosis head may use Born evidence while the Born-prediction head cannot.

Action-ranking candidate ordering is independent of target rank. Rollout metrics and ranks are targets, not candidate inputs. Privileged oracle candidates carry a separate mask.

## Future loss

The documented future loss remains:

`L_total = L_task + λ_geo L_geo + λ_diag L_diag + λ_action L_action + λ_top L_top`

During Phases 12–14 initial training:

```text
λ_top = 0
```

Persistent homology is computed, logged, and available as a diagnostic feature from the beginning. Topology loss can become active only after controlled ablations demonstrate useful, non-leaking predictive signal.

Model architecture is Phase 13. Optimizers, gradients, epochs, checkpoints, and actual learning remain Phase 14.
