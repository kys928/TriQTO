"""Deterministic Phase 14 optimization, checkpointing, and atomic publication."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Sequence
import uuid

import numpy as np
import torch
from torch import Tensor, nn

from triqto.graph.utils import resolve_safe_file, strict_json_load, write_strict_json
from triqto.model import (
    TriQTOModel,
    TriQTOModelConfig,
    load_model_config,
    model_architecture_id,
    model_config_id,
    model_config_to_dict,
)
from triqto.storage.manifest import ManifestReader, ManifestWriter
from triqto.storage.training_schema import (
    TrainingCheckpointRecordV1,
    TrainingEpochRecordV1,
)

from .callbacks import EarlyStoppingState
from .checkpoints import load_training_checkpoint, save_training_checkpoint
from .config import TrainingConfig, training_config_to_dict
from .curriculum import EpochPlan, build_epoch_plan
from .datamodule import (
    build_training_data_spec,
    collate_training_examples,
    deterministic_budget_batches,
    load_training_examples,
)
from .identities import (
    training_operational_config_id,
    training_recipe_id,
    training_run_id,
    training_schema_id,
)
from .logging import checkpoint_record, epoch_record
from .losses import compute_supervised_losses
from .models import (
    CheckpointSummary,
    EpochMetrics,
    TrainingRunResult,
)
from .optimizer import build_optimizer, clip_gradient_norm
from .scheduler import DeterministicLRScheduler
from .source import (
    load_completed_training_view_dataset,
    load_phase7_managed_snapshot,
    snapshot_managed_files,
    verify_training_view_snapshot,
)


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TrainingConfig requested CUDA but CUDA is unavailable")
    return torch.device(name)


def _move_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            setattr(value, field.name, _move_tree(getattr(value, field.name), device))
        return value
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def _average_components(accumulator: dict[str, float], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        raise ValueError("Loss averaging denominator must be positive")
    return {name: value / denominator for name, value in sorted(accumulator.items())}


def _mask_statistics(output: Any, item_count: int) -> dict[str, float]:
    result: dict[str, float] = {}
    stream = output.stream_available_mask.to(torch.float32).mean(dim=0)
    from triqto.model.constants import HEAD_ORDER, STREAM_ORDER

    for index, name in enumerate(STREAM_ORDER):
        result[f"stream_available:{name}"] = float(stream[index].detach().cpu())
    effective = output.effective_head_stream_mask.to(torch.float32).mean(dim=0)
    for head_index, head in enumerate(HEAD_ORDER):
        for stream_index, stream_name in enumerate(STREAM_ORDER):
            result[f"head_stream:{head}:{stream_name}"] = float(
                effective[head_index, stream_index].detach().cpu()
            )
    return result


def _run_epoch(
    *,
    model: TriQTOModel,
    batches: Sequence[Sequence[Any]],
    config: TrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: DeterministicLRScheduler | None,
    global_step: int,
) -> tuple[dict[str, float], int, float, dict[str, float], float]:
    training = optimizer is not None
    model.train(training)
    component_sums: dict[str, float] = defaultdict(float)
    mask_sums: dict[str, float] = defaultdict(float)
    item_total = 0
    gradient_norm_total = 0.0
    optimizer_steps = 0
    privileged_candidates = 0
    candidate_targets = 0
    window_item_totals: dict[int, int] = {}
    if training:
        optimizer.zero_grad(set_to_none=True)
        accumulation = config.gradient_accumulation_steps
        for window_start in range(0, len(batches), accumulation):
            window_end = min(window_start + accumulation, len(batches))
            window_total = sum(len(batch) for batch in batches[window_start:window_end])
            if window_total <= 0:
                raise ValueError("Gradient-accumulation window has no items")
            for index in range(window_start, window_end):
                window_item_totals[index] = window_total

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, examples in enumerate(batches):
            supervised = collate_training_examples(examples)
            item_count = supervised.graph_count
            item_total += item_count
            candidate_targets += int(supervised.targets.action.candidate_target_mask.sum())
            privileged_candidates += int(
                (
                    supervised.targets.action.candidate_target_mask
                    & supervised.targets.action.privileged_oracle_mask
                ).sum()
            )
            supervised = _move_tree(supervised, device)
            output = model(supervised.model_batch)
            auxiliary = (
                model(supervised.auxiliary_hilbert_to_born_batch)
                if supervised.auxiliary_hilbert_to_born_batch is not None
                else None
            )
            losses = compute_supervised_losses(
                output,
                supervised,
                config.loss,
                auxiliary_hilbert_output=auxiliary,
            )
            for name, value in losses.items():
                component_sums[name] += float(value.detach().cpu()) * item_count
            for name, value in _mask_statistics(output, item_count).items():
                mask_sums[name] += value * item_count
            if training:
                window_total = window_item_totals[batch_index]
                item_weight = item_count / window_total
                (losses["total"] * item_weight).backward()
                should_step = (
                    (batch_index + 1) % config.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(batches)
                )
                if should_step:
                    norm = clip_gradient_norm(model, config.max_gradient_norm)
                    gradient_norm_total += norm
                    optimizer_steps += 1
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    assert scheduler is not None
                    scheduler.step()
                    global_step += 1

    if item_total == 0:
        raise ValueError("Epoch has no items")
    averaged = _average_components(component_sums, item_total)
    masks = _average_components(mask_sums, item_total)
    gradient_norm = gradient_norm_total / optimizer_steps if optimizer_steps else 0.0
    privileged_fraction = (
        privileged_candidates / candidate_targets if candidate_targets else 0.0
    )
    return averaged, global_step, gradient_norm, masks, privileged_fraction


def _checkpoint_summary(
    payload: dict[str, Any],
    *,
    kind: str,
    epoch: int,
    global_step: int,
    artifact_ref: str,
    validation_loss: float,
) -> CheckpointSummary:
    return CheckpointSummary(
        checkpoint_id=payload["checkpoint_id"],
        kind=kind,
        epoch_completed=epoch,
        global_step=global_step,
        artifact_ref=artifact_ref,
        content_hash=payload["content_hash"],
        model_state_signature=payload["model_state_signature"],
        validation_loss=validation_loss,
        optimizer_state_present=True,
        scheduler_state_present=True,
        rng_state_present=True,
    )


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}



def _resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _reject_output_source_overlap(
    output: Path,
    training_view: Path,
    phase7: Path | None,
) -> None:
    sources = (("Phase 12 training-view source", training_view),)
    if phase7 is not None:
        sources += (("Phase 7 statevector source", phase7),)
    for label, source in sources:
        if _paths_overlap(output, source):
            raise ValueError(
                f"Training output root {output} overlaps {label} {source}"
            )


def run_training(
    *,
    training_view_root: str | Path,
    output_root: str | Path,
    training_config: TrainingConfig,
    model_config: TriQTOModelConfig,
    phase7_root: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> TrainingRunResult:
    """Train from Phase 12 views and atomically publish a complete Phase 14 run."""
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be TrainingConfig")
    if not isinstance(model_config, TriQTOModelConfig):
        raise TypeError("model_config must be TriQTOModelConfig")
    output = _resolved_path(output_root)
    training_view = _resolved_path(training_view_root)
    phase7 = _resolved_path(phase7_root) if phase7_root is not None else None
    _reject_output_source_overlap(output, training_view, phase7)
    if output.exists():
        raise FileExistsError(f"Training output root already exists: {output}")
    source = load_completed_training_view_dataset(training_view)
    phase7_snapshot = load_phase7_managed_snapshot(phase7) if phase7 is not None else None
    verify_training_view_snapshot(source)
    data_spec = build_training_data_spec(source, model_config, training_config)

    schema_id = training_schema_id()
    architecture_id = model_architecture_id(model_config)
    model_config_identifier = model_config_id(model_config)
    recipe_id = training_recipe_id(
        source.training_view_dataset_id,
        architecture_id,
        model_config_identifier,
        training_config,
        data_spec.content_hash,
    )
    operational_id = training_operational_config_id(training_config)
    run_id = training_run_id(recipe_id, operational_id)

    plans = build_epoch_plan(training_config)
    example_cache: dict[tuple[tuple[str, ...], str], list[Any]] = {}
    batch_cache: dict[tuple[int, str], list[list[Any]]] = {}
    total_optimizer_steps = 0
    for plan in plans:
        for split in ("train", "validation"):
            key = (plan.tasks, split)
            if key not in example_cache:
                examples = load_training_examples(
                    source,
                    tasks=plan.tasks,
                    split=split,
                    spec=data_spec,
                    phase7_root=phase7,
                )
                if len(examples) > training_config.max_items:
                    raise RuntimeError(
                        f"Stage {plan.stage_name} {split} has {len(examples)} items, "
                        f"exceeding max_items={training_config.max_items}"
                    )
                example_cache[key] = examples
            examples = example_cache[key]
            if not examples:
                raise ValueError(
                    f"Stage {plan.stage_name!r} has no {split} items for tasks {plan.tasks}"
                )
            batches = deterministic_budget_batches(
                examples,
                training_config,
                epoch_seed=training_config.seed + plan.epoch,
                shuffle=split == "train",
            )
            batch_cache[(plan.epoch, split)] = batches
        total_optimizer_steps += math.ceil(
            len(batch_cache[(plan.epoch, "train")])
            / training_config.gradient_accumulation_steps
        )
    if total_optimizer_steps <= 0:
        raise ValueError("Training plan contains no optimizer steps")

    _seed_everything(training_config.seed, training_config.deterministic_algorithms)
    device = _resolve_device(training_config.device)
    model = TriQTOModel(model_config).to(device)
    optimizer = build_optimizer(model, training_config.optimizer)
    scheduler = DeterministicLRScheduler(
        optimizer,
        training_config.scheduler,
        total_steps=total_optimizer_steps,
    )
    stopping = EarlyStoppingState(training_config.early_stopping_patience)
    global_step = 0
    start_epoch = 0
    resumed_from: str | None = None
    if resume_checkpoint is not None:
        metadata = load_training_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
            expected_training_run_id=run_id,
        )
        if metadata["training_config"] != training_config_to_dict(training_config):
            raise ValueError("Resume checkpoint training config mismatch")
        if metadata["model_config"] != model_config_to_dict(model_config):
            raise ValueError("Resume checkpoint model config mismatch")
        if metadata["data_spec"] != data_spec.to_dict():
            raise ValueError("Resume checkpoint data spec mismatch")
        start_epoch = int(metadata["epoch_completed"]) + 1
        global_step = int(metadata["global_step"])
        stopping.best_loss = float(metadata["best_validation_loss"])
        stopping.best_epoch = int(metadata["best_epoch"])
        resumed_from = str(metadata["checkpoint_id"])
        if start_epoch >= len(plans):
            raise ValueError("Resume checkpoint already completed the configured curriculum")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected existing training staging root: {staging}")
    checkpoint_dir = staging / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    epoch_metrics: list[EpochMetrics] = []
    checkpoints: list[CheckpointSummary] = []
    stopped_early = False
    final_epoch = start_epoch - 1

    try:
        for plan in plans[start_epoch:]:
            train_losses, global_step, gradient_norm, masks, privileged_fraction = _run_epoch(
                model=model,
                batches=batch_cache[(plan.epoch, "train")],
                config=training_config,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
            )
            validation_losses, _, _, _, _ = _run_epoch(
                model=model,
                batches=batch_cache[(plan.epoch, "validation")],
                config=training_config,
                device=device,
                optimizer=None,
                scheduler=None,
                global_step=global_step,
            )
            validation_total = validation_losses["total"]
            improved, should_stop = stopping.update(validation_total, plan.epoch)
            final_epoch = plan.epoch
            metrics = EpochMetrics(
                epoch=plan.epoch,
                stage_index=plan.stage_index,
                stage_name=plan.stage_name,
                active_tasks=plan.tasks,
                global_step=global_step,
                train_item_count=sum(len(batch) for batch in batch_cache[(plan.epoch, "train")]),
                validation_item_count=sum(len(batch) for batch in batch_cache[(plan.epoch, "validation")]),
                train_batch_count=len(batch_cache[(plan.epoch, "train")]),
                validation_batch_count=len(batch_cache[(plan.epoch, "validation")]),
                learning_rate=scheduler.learning_rate,
                gradient_norm=gradient_norm,
                train_total_loss=train_losses["total"],
                validation_total_loss=validation_total,
                train_losses=train_losses,
                validation_losses=validation_losses,
                mask_utilization=masks,
                privileged_candidate_fraction=privileged_fraction,
                topology_loss_weight=0.0,
            )
            epoch_metrics.append(metrics)

            kinds: list[str] = []
            if (plan.epoch + 1) % training_config.checkpoint_every_epochs == 0:
                kinds.append("epoch")
            if improved and training_config.keep_best_checkpoint:
                kinds.append("best")
            for kind in kinds:
                artifact_ref = f"artifacts/checkpoints/{kind}-epoch-{plan.epoch:04d}.npz"
                payload = save_training_checkpoint(
                    staging / artifact_ref,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_config=training_config,
                    model_config=model_config,
                    data_spec=data_spec,
                    training_schema_id=schema_id,
                    training_recipe_id=recipe_id,
                    operational_config_id=operational_id,
                    training_run_id=run_id,
                    epoch_completed=plan.epoch,
                    stage_index=plan.stage_index,
                    global_step=global_step,
                    best_validation_loss=stopping.best_loss,
                    best_epoch=stopping.best_epoch,
                    kind=kind,
                )
                loaded = load_training_checkpoint(
                    staging / artifact_ref,
                    expected_training_run_id=run_id,
                )
                if loaded["content_hash"] != payload["content_hash"]:
                    raise ValueError("Checkpoint immediate readback hash mismatch")
                checkpoints.append(
                    _checkpoint_summary(
                        payload,
                        kind=kind,
                        epoch=plan.epoch,
                        global_step=global_step,
                        artifact_ref=artifact_ref,
                        validation_loss=validation_total,
                    )
                )
            if should_stop:
                stopped_early = True
                break

        if final_epoch < 0:
            raise RuntimeError("Training completed no epoch")
        final_ref = f"artifacts/checkpoints/final-epoch-{final_epoch:04d}.npz"
        final_payload = save_training_checkpoint(
            staging / final_ref,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
            model_config=model_config,
            data_spec=data_spec,
            training_schema_id=schema_id,
            training_recipe_id=recipe_id,
            operational_config_id=operational_id,
            training_run_id=run_id,
            epoch_completed=final_epoch,
            stage_index=plans[final_epoch].stage_index,
            global_step=global_step,
            best_validation_loss=stopping.best_loss,
            best_epoch=stopping.best_epoch,
            kind="final",
        )
        load_training_checkpoint(final_ref if Path(final_ref).is_absolute() else staging / final_ref, expected_training_run_id=run_id)
        checkpoints.append(
            _checkpoint_summary(
                final_payload,
                kind="final",
                epoch=final_epoch,
                global_step=global_step,
                artifact_ref=final_ref,
                validation_loss=epoch_metrics[-1].validation_total_loss,
            )
        )

        write_strict_json(staging / "training_config.json", training_config_to_dict(training_config))
        write_strict_json(staging / "model_config.json", model_config_to_dict(model_config))
        write_strict_json(staging / "training_data_spec.json", data_spec.to_dict())
        epoch_records = [
            epoch_record(
                metrics,
                training_run_id=run_id,
                training_recipe_id=recipe_id,
                training_view_dataset_id=source.training_view_dataset_id,
                model_architecture_id=architecture_id,
            )
            for metrics in epoch_metrics
        ]
        checkpoint_records = [checkpoint_record(value, training_run_id=run_id) for value in checkpoints]
        writer = ManifestWriter(staging / "manifests")
        writer.write_records("training_epoch_manifest", epoch_records, overwrite=False)
        writer.write_records("training_checkpoint_manifest", checkpoint_records, overwrite=False)

        summary = {
            "training_schema_id": schema_id,
            "training_recipe_id": recipe_id,
            "operational_config_id": operational_id,
            "training_run_id": run_id,
            "training_view_dataset_id": source.training_view_dataset_id,
            "model_architecture_id": architecture_id,
            "model_config_id": model_config_identifier,
            "data_spec_hash": data_spec.content_hash,
            "training_executed": True,
            "model_trained": True,
            "optimizer_state_present": True,
            "scheduler_state_present": True,
            "rng_state_present": True,
            "epoch_count": len(epoch_metrics),
            "checkpoint_count": len(checkpoints),
            "best_epoch": stopping.best_epoch,
            "best_validation_loss": stopping.best_loss,
            "final_epoch": final_epoch,
            "global_step": global_step,
            "stopped_early": stopped_early,
            "resumed_from_checkpoint_id": resumed_from,
            "train_split_used_for_optimization": True,
            "validation_split_used_for_model_selection": True,
            "test_split_used_for_optimization": False,
            "test_split_evaluated": False,
            "audit_only_used_for_gradient": False,
            "topology_loss_weight": 0.0,
            "hardware_data_present": False,
            "hardware_execution_performed": False,
            "heldout_evaluation_performed": False,
            "universal_correction_claim": False,
            "quantum_advantage_claim": False,
        }
        write_strict_json(staging / "training_summary.json", summary)

        reader = ManifestReader(staging / "manifests")
        loaded_epochs = reader.read_typed_records("training_epoch_manifest", TrainingEpochRecordV1)
        loaded_checkpoints = reader.read_typed_records(
            "training_checkpoint_manifest", TrainingCheckpointRecordV1
        )
        if len(loaded_epochs) != len(epoch_records) or len(loaded_checkpoints) != len(checkpoint_records):
            raise ValueError("Typed Phase 14 manifest readback count mismatch")
        for record in loaded_epochs:
            record.validate()
        checkpoint_ids: set[str] = set()
        for record in loaded_checkpoints:
            record.validate()
            if record.checkpoint_id in checkpoint_ids:
                raise ValueError(f"Duplicate checkpoint ID {record.checkpoint_id}")
            checkpoint_ids.add(record.checkpoint_id)
            metadata = load_training_checkpoint(
                resolve_safe_file(staging, record.artifact_ref, "checkpoint artifact_ref"),
                expected_training_run_id=run_id,
            )
            if metadata["checkpoint_id"] != record.checkpoint_id or metadata["content_hash"] != record.content_hash:
                raise ValueError("Checkpoint manifest/artifact join mismatch")

        managed_before_marker = sorted(_relative_files(staging))
        managed_files = sorted([*managed_before_marker, "training_complete.json"])
        completion = {
            "complete": True,
            "training_schema_id": schema_id,
            "training_recipe_id": recipe_id,
            "operational_config_id": operational_id,
            "training_run_id": run_id,
            "training_view_dataset_id": source.training_view_dataset_id,
            "model_architecture_id": architecture_id,
            "model_config_id": model_config_identifier,
            "data_spec_hash": data_spec.content_hash,
            "epoch_count": len(epoch_records),
            "checkpoint_count": len(checkpoint_records),
            "final_epoch": final_epoch,
            "global_step": global_step,
            "phase12_snapshot_hash": source.snapshot.aggregate_sha256,
            "phase7_snapshot_hash": phase7_snapshot.aggregate_sha256 if phase7_snapshot else None,
            "topology_loss_weight": 0.0,
            "test_split_used_for_optimization": False,
            "audit_only_used_for_gradient": False,
            "managed_files": managed_files,
        }
        write_strict_json(staging / "training_complete.json", completion)
        if _relative_files(staging) != set(managed_files):
            raise ValueError("Phase 14 final managed inventory mismatch")
        if strict_json_load(staging / "training_complete.json") != completion:
            raise ValueError("Phase 14 completion marker readback mismatch")
        verify_training_view_snapshot(source)
        if phase7_snapshot is not None:
            actual_phase7 = snapshot_managed_files(
                phase7,
                tuple(entry.reference for entry in phase7_snapshot.entries),
            )
            if actual_phase7 != phase7_snapshot:
                raise RuntimeError("Managed Phase 7 files changed during Phase 14")
        if output.exists():
            raise FileExistsError(f"Training output root appeared during publication: {output}")
        os.replace(staging, output)
        return TrainingRunResult(
            training_view_root=source.root,
            phase7_root=phase7,
            config=training_config,
            data_spec=data_spec,
            training_schema_id=schema_id,
            training_recipe_id=recipe_id,
            operational_config_id=operational_id,
            training_run_id=run_id,
            model_architecture_id=architecture_id,
            model_config_id=model_config_identifier,
            training_view_dataset_id=source.training_view_dataset_id,
            source_snapshot=source.snapshot,
            phase7_snapshot=phase7_snapshot,
            epoch_metrics=epoch_metrics,
            checkpoints=checkpoints,
            best_epoch=stopping.best_epoch,
            best_validation_loss=stopping.best_loss,
            final_epoch=final_epoch,
            global_step=global_step,
            stopped_early=stopped_early,
            resumed_from_checkpoint_id=resumed_from,
            summary=summary,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = ["run_training"]
