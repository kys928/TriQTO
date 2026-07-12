"""Phase 14 manifest conversion helpers."""
from __future__ import annotations

from triqto.storage.training_schema import (
    TrainingCheckpointRecordV1,
    TrainingEpochRecordV1,
)

from .models import CheckpointSummary, EpochMetrics


def epoch_record(
    metrics: EpochMetrics,
    *,
    training_run_id: str,
    training_recipe_id: str,
    training_view_dataset_id: str,
    model_architecture_id: str,
) -> TrainingEpochRecordV1:
    record = TrainingEpochRecordV1(
        training_run_id=training_run_id,
        training_recipe_id=training_recipe_id,
        training_view_dataset_id=training_view_dataset_id,
        model_architecture_id=model_architecture_id,
        epoch=metrics.epoch,
        stage_index=metrics.stage_index,
        stage_name=metrics.stage_name,
        active_tasks=list(metrics.active_tasks),
        global_step=metrics.global_step,
        train_item_count=metrics.train_item_count,
        validation_item_count=metrics.validation_item_count,
        train_batch_count=metrics.train_batch_count,
        validation_batch_count=metrics.validation_batch_count,
        learning_rate=metrics.learning_rate,
        gradient_norm=metrics.gradient_norm,
        train_total_loss=metrics.train_total_loss,
        validation_total_loss=metrics.validation_total_loss,
        train_losses=dict(metrics.train_losses),
        validation_losses=dict(metrics.validation_losses),
        mask_utilization=dict(metrics.mask_utilization),
        privileged_candidate_fraction=metrics.privileged_candidate_fraction,
        topology_loss_weight=metrics.topology_loss_weight,
    )
    record.validate()
    return record


def checkpoint_record(summary: CheckpointSummary, *, training_run_id: str) -> TrainingCheckpointRecordV1:
    record = TrainingCheckpointRecordV1(
        checkpoint_id=summary.checkpoint_id,
        training_run_id=training_run_id,
        kind=summary.kind,
        epoch_completed=summary.epoch_completed,
        global_step=summary.global_step,
        artifact_ref=summary.artifact_ref,
        content_hash=summary.content_hash,
        model_state_signature=summary.model_state_signature,
        validation_loss=summary.validation_loss,
        optimizer_state_present=summary.optimizer_state_present,
        scheduler_state_present=summary.scheduler_state_present,
        rng_state_present=summary.rng_state_present,
        metadata={},
    )
    record.validate()
    return record


__all__ = ["checkpoint_record", "epoch_record"]
