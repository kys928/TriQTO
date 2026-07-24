"""Vectorized, resumable full multi-task trainer for immutable model-ready data."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch import Tensor

from triqto.core.ids import make_deterministic_id
from triqto.model import (
    TriQTOModel,
    TriQTOModelConfig,
    model_architecture_id,
    model_config_id,
    model_config_to_dict,
)
from triqto.training.callbacks import EarlyStoppingState
from triqto.training.checkpoints import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from triqto.training.config import TrainingConfig, training_config_to_dict
from triqto.training.curriculum import EpochPlan, build_epoch_plan
from triqto.training.identities import (
    training_operational_config_id,
    training_recipe_id,
    training_schema_id,
)
from triqto.training.optimizer import (
    build_optimizer,
    clip_gradient_norm,
    finite_gradient_norm,
)
from triqto.training.scheduler import DeterministicLRScheduler

from .debug_trainer import build_model_ready_data_spec
from .full_trainer_types import FullTrainingResult
from .multitask_adapter import build_model_ready_multitask_example
from .multitask_collate import (
    collate_model_ready_multitask_examples,
    validate_model_ready_batch_budget,
)
from .multitask_losses import compute_model_ready_multitask_losses
from .multitask_metrics import ModelReadyMetricAccumulator
from .source import (
    load_model_ready_artifact,
    load_model_ready_dataset,
    sha256_file,
)
from .types import CANONICAL_TOPOLOGY_INPUT_DIM, ModelReadyDataset

MODEL_READY_FULL_SCHEMA = "triqto.training.model_ready_full.v1"
MODEL_READY_FULL_RUNNER_VERSION = "triqto.training.model_ready_full_runner.v1"
_FULL_TASKS = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "joint_multitask",
    "hardware_masked",
)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
        raise RuntimeError("training requested CUDA, but CUDA is unavailable")
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


def _hash_order(seed: int, split: str, task: str, item_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{split}:{task}:{item_id}".encode("utf-8")
    ).hexdigest()


def _natural_capped_records(
    rows: Sequence[dict[str, Any]],
    *,
    task: str,
    split: str,
    limit: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: _hash_order(
            seed, split, task, str(row["view_item_id"])
        ),
    )
    if limit <= 0 or limit >= len(ordered):
        return tuple(ordered)
    if task not in {"action_ranking", "joint_multitask", "hardware_masked"}:
        return tuple(ordered[:limit])
    positives = [row for row in ordered if bool(row.get("should_act"))]
    negatives = [row for row in ordered if not bool(row.get("should_act"))]
    if not positives or not negatives:
        return tuple(ordered[:limit])
    positive_target = int(round(limit * len(positives) / len(ordered)))
    positive_target = min(max(positive_target, 1), len(positives))
    negative_target = min(max(limit - positive_target, 1), len(negatives))
    while positive_target + negative_target < limit:
        if positive_target < len(positives):
            positive_target += 1
        elif negative_target < len(negatives):
            negative_target += 1
        else:
            break
    selected = positives[:positive_target] + negatives[:negative_target]
    selected.sort(
        key=lambda row: _hash_order(
            seed + 7919, split, task, str(row["view_item_id"])
        )
    )
    return tuple(selected[:limit])


def _build_selection(
    dataset: ModelReadyDataset,
    *,
    train_limit_per_task: int,
    validation_limit_per_task: int,
    seed: int,
) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    selection: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
    for task in _FULL_TASKS:
        for split, limit, offset in (
            ("train", train_limit_per_task, 0),
            ("validation", validation_limit_per_task, 100_003),
        ):
            rows = dataset.records_by_task_split.get((task, split), ())
            if not rows:
                raise ValueError(f"model-ready dataset has no {task}/{split} rows")
            selected = _natural_capped_records(
                rows,
                task=task,
                split=split,
                limit=limit,
                seed=seed + offset,
            )
            entities = [str(row["entity_id"]) for row in selected]
            if len(entities) != len(set(entities)):
                raise ValueError(f"{task}/{split} selection duplicates entity rows")
            selection[(task, split)] = selected
    return selection


def _selection_payload(
    selection: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    *,
    train_limit_per_task: int,
    validation_limit_per_task: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "policy": "natural_distribution_deterministic_cap",
        "seed": seed,
        "train_limit_per_task": train_limit_per_task,
        "validation_limit_per_task": validation_limit_per_task,
        "test_rows_used": False,
        "counts": {
            f"{task}:{split}": len(rows)
            for (task, split), rows in sorted(selection.items())
        },
        "view_item_ids": {
            f"{task}:{split}": [str(row["view_item_id"]) for row in rows]
            for (task, split), rows in sorted(selection.items())
        },
    }


def _task_batches(
    selection: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    *,
    tasks: Sequence[str],
    split: str,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    by_task: dict[str, list[tuple[dict[str, Any], ...]]] = {}
    for task in tasks:
        rows = list(selection[(task, split)])
        if shuffle:
            rows.sort(
                key=lambda row: _hash_order(
                    seed + epoch * 1_000_003,
                    split,
                    task,
                    str(row["view_item_id"]),
                )
            )
        else:
            rows.sort(key=lambda row: str(row["view_item_id"]))
        by_task[task] = [
            tuple(rows[index : index + batch_size])
            for index in range(0, len(rows), batch_size)
        ]
    batches: list[tuple[dict[str, Any], ...]] = []
    maximum = max(len(values) for values in by_task.values())
    for batch_index in range(maximum):
        task_order = list(tasks)
        if shuffle:
            task_order.sort(
                key=lambda task: _hash_order(
                    seed + epoch * 97,
                    split,
                    task,
                    str(batch_index),
                )
            )
        for task in task_order:
            values = by_task[task]
            if batch_index < len(values):
                batches.append(values[batch_index])
    return tuple(batches)


def _load_batch(
    dataset: ModelReadyDataset,
    records: Sequence[dict[str, Any]],
    model_config: TriQTOModelConfig,
    training_config: TrainingConfig,
):
    examples = [
        build_model_ready_multitask_example(
            load_model_ready_artifact(dataset, record), model_config
        )
        for record in records
    ]
    batch = collate_model_ready_multitask_examples(examples)
    validate_model_ready_batch_budget(batch, training_config)
    batch.model_batch.validate(model_config)
    return batch


def _train_reward_mean(
    dataset: ModelReadyDataset,
    records: Sequence[dict[str, Any]],
) -> float:
    total = 0.0
    count = 0
    for record in records:
        artifact = load_model_ready_artifact(dataset, record)
        targets = artifact.targets
        ranking = bool(
            np.asarray(targets.get("y_ranking_loss_mask", False)).reshape(-1)[0]
        )
        if not ranking:
            continue
        rewards = np.asarray(targets["y_candidate_reward"], dtype=np.float64).reshape(-1)
        eligible = np.asarray(
            targets.get(
                "y_candidate_eligible_mask",
                np.ones(rewards.size, dtype=np.bool_),
            ),
            dtype=np.bool_,
        ).reshape(-1)
        if eligible.size != rewards.size:
            raise ValueError("candidate reward/eligibility widths differ")
        total += float(rewards[eligible].sum())
        count += int(eligible.sum())
    return total / count if count else 0.0


def _stage_selection_metric(metrics: Mapping[str, Any], tasks: Sequence[str]) -> float:
    values: list[float] = []
    per_task = metrics["per_task"]
    for task in tasks:
        payload = per_task.get(task)
        if payload is None:
            raise ValueError(f"validation metrics miss task {task!r}")
        values.append(float(payload["losses"]["total"]))
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError("stage validation selection metric is non-finite")
    return result


def _run_epoch(
    *,
    dataset: ModelReadyDataset,
    record_batches: Sequence[Sequence[dict[str, Any]]],
    model: TriQTOModel,
    model_config: TriQTOModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: DeterministicLRScheduler | None,
    global_step: int,
    train_reward_mean: float,
    no_op_feature_index: int,
    label: str,
    progress_every_batches: int,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    if training != (scheduler is not None):
        raise ValueError("optimizer and scheduler must both be present or absent")
    model.train(training)
    accumulator = ModelReadyMetricAccumulator(
        train_reward_mean=train_reward_mean,
        no_op_feature_index=no_op_feature_index,
    )
    gradient_pre: list[float] = []
    gradient_post: list[float] = []
    clipped_steps = 0
    optimizer_steps = 0
    accumulation = training_config.gradient_accumulation_steps
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        window_items = 0
        window_batches = 0
        for batch_index, records in enumerate(record_batches):
            batch = _load_batch(
                dataset, records, model_config, training_config
            )
            batch = _move_tree(batch, device)
            output = model(batch.model_batch)
            losses = compute_model_ready_multitask_losses(
                output, batch, training_config.loss
            )
            accumulator.update(output, batch, losses)
            if progress_every_batches > 0 and (
                (batch_index + 1) % progress_every_batches == 0
                or batch_index + 1 == len(record_batches)
            ):
                print(
                    f"  {label}: {batch_index + 1:,}/{len(record_batches):,} batches "
                    f"items={accumulator.item_count:,} global_step={global_step:,}",
                    flush=True,
                )
            if training:
                window_items += batch.graph_count
                window_batches += 1
                # Equal item weighting inside each accumulation window. Final short
                # windows are corrected at step time by scaling all batch means by
                # their item count and dividing by the observed window item total.
                (losses["total"] * batch.graph_count).backward()
                should_step = (
                    window_batches >= accumulation
                    or batch_index + 1 == len(record_batches)
                )
                if should_step:
                    if window_items <= 0:
                        raise RuntimeError("gradient window contains no items")
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(window_items)
                    pre = clip_gradient_norm(
                        model, training_config.max_gradient_norm
                    )
                    post = finite_gradient_norm(model.parameters())
                    gradient_pre.append(pre)
                    gradient_post.append(post)
                    clipped_steps += int(pre > training_config.max_gradient_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_steps += 1
                    global_step += 1
                    window_items = 0
                    window_batches = 0
    metrics = accumulator.finalize()
    metrics["optimizer_steps"] = optimizer_steps
    metrics["gradient"] = {
        "pre_clip_mean": (
            sum(gradient_pre) / len(gradient_pre) if gradient_pre else 0.0
        ),
        "pre_clip_max": max(gradient_pre) if gradient_pre else 0.0,
        "post_clip_mean": (
            sum(gradient_post) / len(gradient_post) if gradient_post else 0.0
        ),
        "clipped_step_fraction": (
            clipped_steps / len(gradient_pre) if gradient_pre else 0.0
        ),
        "maximum_norm": training_config.max_gradient_norm,
    }
    return metrics, global_step


def _reconstruct_stage_state(
    run_root: Path, plans: Sequence[EpochPlan]
) -> dict[int, EarlyStoppingState]:
    states: dict[int, EarlyStoppingState] = {}
    for plan in plans:
        states.setdefault(plan.stage_index, EarlyStoppingState(0))
    for path in sorted((run_root / "metrics").glob("epoch-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stage_index = int(payload["stage_index"])
        state = states[stage_index]
        state.patience = int(payload["early_stopping_patience"])
        state.update(float(payload["selection_metric"]), int(payload["epoch"]))
    return states


def run_model_ready_full_training(
    *,
    model_ready_root: str | Path,
    output_root: str | Path,
    model_config: TriQTOModelConfig,
    training_config: TrainingConfig,
    train_limit_per_task: int = 0,
    validation_limit_per_task: int = 0,
    resume_checkpoint: str | Path | None = None,
    progress_every_batches: int = 25,
) -> FullTrainingResult:
    """Run the vectorized curriculum while preserving the immutable source."""
    if model_config.topology_input_dim != CANONICAL_TOPOLOGY_INPUT_DIM:
        raise ValueError("full model-ready training requires topology_input_dim=121")
    if training_config.topology_loss_weight != 0.0 or training_config.loss.topology_weight != 0.0:
        raise ValueError("lambda_top must remain exactly 0.0")
    if training_config.num_workers != 0:
        raise ValueError("v1 full runner requires num_workers=0 for deterministic IO")
    configured_tasks = {
        task for stage in training_config.stages for task in stage.tasks
    }
    unsupported = configured_tasks - set(_FULL_TASKS)
    if unsupported:
        raise ValueError(
            f"model-ready full trainer does not support tasks {sorted(unsupported)}"
        )
    if configured_tasks != set(_FULL_TASKS):
        raise ValueError(
            "full campaign config must cover diagnosis, action_ranking, "
            "born_prediction, joint_multitask, and hardware_masked"
        )
    for name, value in (
        ("train_limit_per_task", train_limit_per_task),
        ("validation_limit_per_task", validation_limit_per_task),
        ("progress_every_batches", progress_every_batches),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    dataset = load_model_ready_dataset(model_ready_root)
    source_manifest = (
        dataset.root / "manifests" / "processed_item_manifest.parquet"
    )
    source_hash_before = sha256_file(source_manifest)
    selection = _build_selection(
        dataset,
        train_limit_per_task=train_limit_per_task,
        validation_limit_per_task=validation_limit_per_task,
        seed=training_config.seed,
    )
    selection_payload = _selection_payload(
        selection,
        train_limit_per_task=train_limit_per_task,
        validation_limit_per_task=validation_limit_per_task,
        seed=training_config.seed,
    )
    data_spec = build_model_ready_data_spec(
        dataset, model_config, task="action_ranking"
    )
    no_op_feature_index = data_spec.action_feature_names.index("is_no_op")
    reward_mean = _train_reward_mean(
        dataset, selection[("action_ranking", "train")]
    )

    plans = build_epoch_plan(training_config)
    batch_plan: dict[tuple[int, str], tuple[tuple[dict[str, Any], ...], ...]] = {}
    total_optimizer_steps = 0
    for plan in plans:
        for split in ("train", "validation"):
            batches = _task_batches(
                selection,
                tasks=plan.tasks,
                split=split,
                batch_size=training_config.batch_size,
                seed=training_config.seed,
                epoch=plan.epoch,
                shuffle=split == "train",
            )
            if not batches:
                raise ValueError(
                    f"stage {plan.stage_name!r} has no {split} batches"
                )
            selected_count = sum(len(batch) for batch in batches)
            if selected_count > training_config.max_items:
                raise RuntimeError(
                    f"stage {plan.stage_name!r} {split} has {selected_count} items, "
                    f"exceeding max_items={training_config.max_items}"
                )
            batch_plan[(plan.epoch, split)] = batches
        total_optimizer_steps += math.ceil(
            len(batch_plan[(plan.epoch, "train")])
            / training_config.gradient_accumulation_steps
        )
    if total_optimizer_steps <= 0:
        raise ValueError("full campaign contains no optimizer steps")

    selection_hash = hashlib.sha256(_json_bytes(selection_payload)).hexdigest()
    architecture_id = model_architecture_id(model_config)
    model_config_identifier = model_config_id(model_config)
    base_recipe_id = training_recipe_id(
        dataset.training_view_dataset_id,
        architecture_id,
        model_config_identifier,
        training_config,
        data_spec.content_hash,
    )
    recipe_id = make_deterministic_id(
        "modelreadyfullrecipe",
        {
            "base_training_recipe_id": base_recipe_id,
            "schema": MODEL_READY_FULL_SCHEMA,
            "runner_version": MODEL_READY_FULL_RUNNER_VERSION,
            "source_manifest_sha256": dataset.manifest_sha256,
            "selection_sha256": selection_hash,
        },
    )
    operational_id = training_operational_config_id(training_config)
    run_id = make_deterministic_id(
        "modelreadyfullrun",
        {"recipe_id": recipe_id, "operational_config_id": operational_id},
    )
    output_parent = Path(output_root).expanduser().resolve()
    if output_parent == dataset.root or output_parent.is_relative_to(dataset.root):
        raise ValueError("training output must live outside the model-ready source")
    run_root = output_parent / run_id
    complete_marker = run_root / "model_ready_full_complete.json"
    if complete_marker.is_file():
        payload = json.loads(complete_marker.read_text(encoding="utf-8"))
        if payload.get("complete") is True:
            return FullTrainingResult(
                status="already_complete", output_root=run_root, summary=payload
            )
    if run_root.exists() and resume_checkpoint is None:
        raise FileExistsError(
            f"incomplete run exists at {run_root}; provide resume_checkpoint"
        )
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(run_root / "manifests" / "selection.json", selection_payload)
        _atomic_json(
            run_root / "manifests" / "model_config.json",
            model_config_to_dict(model_config),
        )
        _atomic_json(
            run_root / "manifests" / "training_config.json",
            training_config_to_dict(training_config),
        )
        _atomic_json(
            run_root / "manifests" / "data_spec.json", data_spec.to_dict()
        )
        _atomic_json(
            run_root / "model_ready_full_incomplete.json",
            {
                "complete": False,
                "run_id": run_id,
                "source_manifest_sha256": dataset.manifest_sha256,
                "selection_sha256": selection_hash,
                "test_rows_used": False,
                "lambda_top": 0.0,
            },
        )

    _seed_everything(
        training_config.seed, training_config.deterministic_algorithms
    )
    device = _resolve_device(training_config.device)
    model = TriQTOModel(model_config).to(device)
    optimizer = build_optimizer(model, training_config.optimizer)
    scheduler = DeterministicLRScheduler(
        optimizer,
        training_config.scheduler,
        total_steps=total_optimizer_steps,
    )
    schema_id = training_schema_id()
    global_step = 0
    start_epoch = 0
    resumed_from: str | None = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.is_relative_to(run_root):
            raise ValueError("resume checkpoint must belong to the same run root")
        metadata = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
            expected_training_run_id=run_id,
        )
        start_epoch = int(metadata["epoch_completed"]) + 1
        global_step = int(metadata["global_step"])
        resumed_from = str(metadata["checkpoint_id"])

    stage_states = _reconstruct_stage_state(run_root, plans)
    for stage_index in stage_states:
        stage_states[stage_index].patience = training_config.early_stopping_patience
    stopped_stages: set[int] = set()
    epoch_summaries: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    best_by_stage: dict[str, dict[str, Any]] = {}

    try:
        for plan in plans:
            if plan.epoch < start_epoch or plan.stage_index in stopped_stages:
                continue
            print(
                f"Epoch {plan.epoch + 1}/{len(plans)} stage={plan.stage_name} "
                f"tasks={','.join(plan.tasks)}",
                flush=True,
            )
            train_metrics, global_step = _run_epoch(
                dataset=dataset,
                record_batches=batch_plan[(plan.epoch, "train")],
                model=model,
                model_config=model_config,
                training_config=training_config,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                train_reward_mean=reward_mean,
                no_op_feature_index=no_op_feature_index,
                label=f"epoch-{plan.epoch:03d}/train",
                progress_every_batches=progress_every_batches,
            )
            validation_metrics, global_step = _run_epoch(
                dataset=dataset,
                record_batches=batch_plan[(plan.epoch, "validation")],
                model=model,
                model_config=model_config,
                training_config=training_config,
                device=device,
                optimizer=None,
                scheduler=None,
                global_step=global_step,
                train_reward_mean=reward_mean,
                no_op_feature_index=no_op_feature_index,
                label=f"epoch-{plan.epoch:03d}/validation",
                progress_every_batches=progress_every_batches,
            )
            selection_metric = _stage_selection_metric(
                validation_metrics, plan.tasks
            )
            state = stage_states[plan.stage_index]
            improved, should_stop = state.update(
                selection_metric, plan.epoch
            )
            epoch_payload = {
                "epoch": plan.epoch,
                "stage_index": plan.stage_index,
                "stage_name": plan.stage_name,
                "stage_epoch": plan.stage_epoch,
                "tasks": list(plan.tasks),
                "global_step": global_step,
                "learning_rate": scheduler.learning_rate,
                "selection_metric": selection_metric,
                "selection_metric_policy": "mean_per_task_total_loss",
                "early_stopping_patience": training_config.early_stopping_patience,
                "improved": improved,
                "train": train_metrics,
                "validation": validation_metrics,
                "lambda_top": 0.0,
            }
            epoch_summaries.append(epoch_payload)
            print(
                f"  epoch={plan.epoch} selection_metric={selection_metric:.6f} "
                f"improved={improved}",
                flush=True,
            )
            _atomic_json(
                run_root / "metrics" / f"epoch-{plan.epoch:03d}.json",
                epoch_payload,
            )
            checkpoint_path = (
                run_root
                / "artifacts"
                / "checkpoints"
                / f"epoch-{plan.epoch:03d}.npz"
            )
            checkpoint_payload = save_training_checkpoint(
                checkpoint_path,
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
                best_validation_loss=state.best_loss,
                best_epoch=state.best_epoch,
                kind="epoch",
            )
            record = {
                **checkpoint_payload,
                "kind": "epoch",
                "epoch": plan.epoch,
                "stage_index": plan.stage_index,
                "artifact_ref": checkpoint_path.relative_to(run_root).as_posix(),
                "selection_metric": selection_metric,
            }
            checkpoint_records.append(record)
            if improved:
                best_path = (
                    run_root
                    / "artifacts"
                    / "checkpoints"
                    / f"best-stage-{plan.stage_index:02d}-epoch-{plan.epoch:03d}.npz"
                )
                best_payload = save_training_checkpoint(
                    best_path,
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
                    best_validation_loss=state.best_loss,
                    best_epoch=state.best_epoch,
                    kind="best",
                )
                best_record = {
                    **best_payload,
                    "kind": "best",
                    "epoch": plan.epoch,
                    "stage_index": plan.stage_index,
                    "artifact_ref": best_path.relative_to(run_root).as_posix(),
                    "selection_metric": selection_metric,
                }
                best_by_stage[str(plan.stage_index)] = best_record
                checkpoint_records.append(best_record)
            if should_stop:
                stopped_stages.add(plan.stage_index)

        executed_epochs = sorted(
            int(path.stem.split("-")[-1])
            for path in (run_root / "metrics").glob("epoch-*.json")
        )
        if not executed_epochs:
            raise RuntimeError("full training produced no completed epochs")
        final_epoch = executed_epochs[-1]
        final_plan = next(plan for plan in plans if plan.epoch == final_epoch)
        final_state = stage_states[final_plan.stage_index]
        final_path = run_root / "artifacts" / "checkpoints" / "final.npz"
        final_payload = save_training_checkpoint(
            final_path,
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
            stage_index=final_plan.stage_index,
            global_step=global_step,
            best_validation_loss=final_state.best_loss,
            best_epoch=final_state.best_epoch,
            kind="final",
        )
        final_record = {
            **final_payload,
            "kind": "final",
            "epoch": final_epoch,
            "stage_index": final_plan.stage_index,
            "artifact_ref": final_path.relative_to(run_root).as_posix(),
        }
        checkpoint_records.append(final_record)

        source_hash_after = sha256_file(source_manifest)
        if source_hash_after != source_hash_before:
            raise RuntimeError("model-ready source manifest changed during training")
        all_metrics = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((run_root / "metrics").glob("epoch-*.json"))
        ]
        summary = {
            "schema": MODEL_READY_FULL_SCHEMA,
            "runner_version": MODEL_READY_FULL_RUNNER_VERSION,
            "complete": True,
            "status": "complete",
            "run_id": run_id,
            "recipe_id": recipe_id,
            "operational_config_id": operational_id,
            "training_schema_id": schema_id,
            "training_view_dataset_id": dataset.training_view_dataset_id,
            "source_root": str(dataset.root),
            "source_manifest_sha256": source_hash_after,
            "selection_sha256": selection_hash,
            "device": str(device),
            "global_step": global_step,
            "total_planned_optimizer_steps": total_optimizer_steps,
            "executed_epochs": executed_epochs,
            "stopped_stage_indices": sorted(stopped_stages),
            "resumed_from_checkpoint_id": resumed_from,
            "best_checkpoints_by_stage": best_by_stage,
            "final_checkpoint": final_record,
            "checkpoints": checkpoint_records,
            "epoch_metrics": all_metrics,
            "train_reward_mean": reward_mean,
            "test_rows_used": False,
            "lambda_top": 0.0,
            "scientific_boundaries": {
                "source_mutated": False,
                "test_split_used": False,
                "topology_loss_active": False,
                "action_head_topology_enabled": False,
                "born_prediction_head_topology_enabled": False,
                "validation_distribution_natural": True,
                "ranking_policy_claimed_success": False,
            },
        }
        _atomic_json(run_root / "reports" / "summary.json", summary)
        _atomic_json(run_root / "model_ready_full_complete.json", summary)
        incomplete = run_root / "model_ready_full_incomplete.json"
        if incomplete.exists():
            incomplete.unlink()
        return FullTrainingResult(
            status="complete", output_root=run_root, summary=summary
        )
    except Exception as exc:
        _atomic_json(
            run_root / "reports" / "failure.json",
            {
                "complete": False,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "global_step": global_step,
                "resume_supported": True,
            },
        )
        raise


__all__ = [
    "MODEL_READY_FULL_RUNNER_VERSION",
    "MODEL_READY_FULL_SCHEMA",
    "run_model_ready_full_training",
]
