#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace/triqto"
SOURCE="/workspace/triqto-data/phase15_6_pilot_v2/data/phase12_model_ready_topology_queries/phase12_queries_1eeca8e089c88b41cfb756ea"
OUTPUT="/workspace/triqto-data/phase15_6_pilot_v2/runs/model_ready_multitask_full"
LOG_DIR="/workspace/triqto-data/phase15_6_pilot_v2/logs"
LOG="$LOG_DIR/model_ready_multitask_full.log"
SESSION="triqto-multitask-full"

if [[ "${TRIQTO_IN_TMUX:-0}" != "1" ]]; then
    cd "$REPO"
    mkdir -p "$OUTPUT" "$LOG_DIR"

    tmux kill-session -t "$SESSION" 2>/dev/null || true

    if [[ -f "$LOG" ]]; then
        mv "$LOG" "${LOG%.log}_previous_$(date -u +%Y%m%dT%H%M%SZ).log"
    fi

    tmux new-session -d -s "$SESSION" \
        env TRIQTO_IN_TMUX=1 bash "$0"

    echo "Started tmux session: $SESSION"
    echo "Log: $LOG"
    echo
    echo "Monitor with:"
    echo "  tail -n 0 -F '$LOG'"
    exit 0
fi

cd "$REPO"
set -o pipefail

echo "Starting TriQTO full multitask campaign" | tee "$LOG"
echo "Commit: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Started: $(date --iso-8601=seconds)" | tee -a "$LOG"
echo "CUBLAS_WORKSPACE_CONFIG=:4096:8" | tee -a "$LOG"
nvidia-smi \
    --query-gpu=name,memory.total,memory.free \
    --format=csv,noheader | tee -a "$LOG"
echo | tee -a "$LOG"

set +e
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
TRIQTO_MODEL_READY_ROOT="$SOURCE" \
TRIQTO_MODEL_READY_FULL_OUTPUT_ROOT="$OUTPUT" \
TRIQTO_MODEL_CONFIG="$REPO/configs/model/phase15_6_base.json" \
TRIQTO_TRAINING_CONFIG="$REPO/configs/train/phase15_6_model_ready_full.yaml" \
TRIQTO_FULL_DEVICE="cuda" \
TRIQTO_FULL_TRAIN_LIMIT_PER_TASK="0" \
TRIQTO_FULL_VALIDATION_LIMIT_PER_TASK="0" \
TRIQTO_FULL_PROGRESS_EVERY_BATCHES="100" \
PYTHONPATH="$REPO/src" \
python scripts/train_model_ready_full.py 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

echo | tee -a "$LOG"
echo "Finished: $(date --iso-8601=seconds)" | tee -a "$LOG"
echo "Full campaign exit code: $status" | tee -a "$LOG"

exec bash
