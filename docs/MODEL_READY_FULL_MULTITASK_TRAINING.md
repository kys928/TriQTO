# Vectorized model-ready Phase 15.6 training

This runner is the first full multi-task optimizer for the immutable model-ready
`x_*`/`y_*` product. It follows the successful sequential action-policy debug
run, but replaces one-graph forwards with real variable-size graph batches.

## Curriculum

The default full curriculum contains three stages:

1. `diagnosis`, `action_ranking`, and `born_prediction` foundations;
2. `joint_multitask`, where topology is available only to safe heads;
3. `hardware_masked`, with Hilbert and attached simulator topology unavailable.

Batches are task-homogeneous and interleaved deterministically inside a stage.
This keeps loss accounting exact while still updating the shared model across
multiple tasks. Train rows are shuffled deterministically. Validation rows use
the natural class distribution. Test rows are not loaded.

## Scientific boundaries

- only `x_*` arrays enter the model;
- only `y_*` arrays supply supervision;
- the action and Born heads cannot observe topology;
- the diagnosis head may observe topology in `joint_multitask`;
- `lambda_top` remains exactly zero;
- no Hilbert input is reconstructed or invented;
- the source manifest is hash checked before and after training;
- ranking success is not claimed merely because the trainer completes.

## Metrics

The runner writes per-epoch, per-task train and validation metrics including:

- should-act accuracy, balanced accuracy, precision, recall, F1, AUROC, and
  confusion counts;
- ranking top-1/top-3, MRR, NDCG, selected-candidate percentile, and a
  candidate-count-adjusted random top-1 baseline;
- reward MSE, zero-prediction MSE, training-mean MSE, and selected-versus-no-op
  target reward;
- diagnosis class accuracy/confusion, strength MAE, and affected-qubit F1;
- Born KL and Hellinger losses;
- pre/post clipping norms and fraction of optimizer steps clipped;
- stream/head mask utilization.

## Checkpoint and resume policy

The run root is created with an incomplete marker. Every completed epoch writes a
pickle-free checkpoint containing model, optimizer, scheduler, and RNG state.
A failed or disconnected process can be resumed by supplying an epoch checkpoint
from the same run root. The completion marker is written only after the final
checkpoint and source-integrity verification succeed.

## Smoke run

Use `configs/train/phase15_6_model_ready_multitask_smoke.yaml` with small per-task
caps before the full campaign. The smoke run proves vectorized collation and all
heads; its metrics are not a scientific result.

```bash
TRIQTO_MODEL_READY_ROOT=/path/to/phase12_topology_<id> \
TRIQTO_MODEL_READY_FULL_OUTPUT_ROOT=/path/to/multitask-smoke \
TRIQTO_TRAINING_CONFIG=configs/train/phase15_6_model_ready_multitask_smoke.yaml \
TRIQTO_FULL_TRAIN_LIMIT_PER_TASK=32 \
TRIQTO_FULL_VALIDATION_LIMIT_PER_TASK=16 \
PYTHONPATH=$PWD/src \
python scripts/train_model_ready_full.py
```

For the full campaign, select `phase15_6_model_ready_full.yaml` and set both
per-task limits to `0`. Do not launch the full campaign until the smoke run and
checkpoint restoration pass on the real 67,200-row product.
