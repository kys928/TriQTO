# Model-ready debug training

This stage is the first optimizer/checkpoint smoke run over the immutable
Phase 12 model-ready product. It is intentionally small and is not a reported
scientific result.

## Scope

The first debug recipe trains the two-stage action path on the dedicated
`action_ranking` view only:

1. graph-level `should_act` weighted binary cross-entropy;
2. candidate listwise ranking and reward regression only where
   `y_ranking_loss_mask=true`.

Using only the dedicated action view prevents the same entity from being counted
again through the joint and hardware-masked views. The debug subset is balanced
between act/no-action examples so both branches execute; its accuracy and loss
must not be treated as representative campaign metrics.

The runner uses one variable-size graph forward at a time and accumulates
gradients across each logical batch. This is deliberately slower than a future
vectorized collator, but it validates the scientific path with minimal batching
complexity.

## Preserved boundaries

- the source manifest and NPZ files are read-only and hash checked;
- only train and validation rows are selected;
- the test split is never loaded into optimization;
- action ranking and Born prediction remain forbidden from topology;
- topology loss remains exactly `lambda_top=0`;
- checkpoints are pickle-free NPZ artifacts containing model, optimizer,
  scheduler, and RNG state;
- outputs are published atomically to a content-addressed run directory.

## Environment-driven run

```bash
export TRIQTO_MODEL_READY_ROOT="/path/to/phase12_topology_<id>"
export TRIQTO_MODEL_READY_DEBUG_OUTPUT_ROOT="/path/to/debug-runs"

TRIQTO_MODEL_CONFIG="configs/model/phase15_6_base.json" \
TRIQTO_TRAINING_CONFIG="configs/train/phase15_6_model_ready_debug.yaml" \
TRIQTO_DEBUG_TASK="action_ranking" \
TRIQTO_DEBUG_TRAIN_ITEMS="16" \
TRIQTO_DEBUG_VALIDATION_ITEMS="8" \
PYTHONPATH="$PWD/src" \
python scripts/train_model_ready_debug.py
```

The output contains:

- `model_ready_debug_complete.json`;
- `manifests/selection.json`;
- model/training/data-spec identity snapshots;
- per-epoch train and validation metrics;
- epoch, best, and final safe checkpoints;
- `reports/summary.json`.

A successful debug run proves that data loading, forward computation, weighted
action gating, conditional ranking, backward computation, optimizer stepping,
validation, and checkpoint serialization work together. It does not yet prove
model quality or enable the full multi-task Phase 15.6 campaign.
