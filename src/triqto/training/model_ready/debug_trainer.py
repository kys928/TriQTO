"""Small deterministic training runner for immutable model-ready artifacts.

The debug runner intentionally uses sequential graph microbatches inside each
logical batch. This keeps the variable-size graph/candidate path simple and
scientifically transparent while exercising real epochs, optimization,
validation, gradient clipping, scheduling, and pickle-free checkpoints.
"""
from __future__ import annotations

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
    ACTION_EDIT_TYPES,
    DISTORTION_LABELS,
    TriQTOModel,
    TriQTOModelConfig,
    model_architecture_id,
    model_config_id,
    model_config_to_dict,
)
from triqto.training.checkpoints import save_training_checkpoint
from triqto.training.config import TrainingConfig, training_config_to_dict
from triqto.training.constants import ACTION_EDIT_TYPE_MAP, DISTORTION_TO_COARSE_LABEL
from triqto.training.identities import training_operational_config_id, training_schema_id
from triqto.training.models import TrainingDataSpec
from triqto.training.optimizer import build_optimizer, clip_gradient_norm
from triqto.training.scheduler import DeterministicLRScheduler

from .adapter import build_model_ready_example
from .losses import compute_model_ready_action_losses
from .source import (
    load_model_ready_artifact,
    load_model_ready_dataset,
    select_model_ready_record,
    sha256_file,
)
from .types import (
    CANONICAL_TOPOLOGY_INPUT_DIM,
    MODEL_READY_ADAPTER_VERSION,
    ModelReadyActionTargets,
    ModelReadyDataset,
)

MODEL_READY_DEBUG_SCHEMA = "triqto.training.model_ready_debug.v1"
MODEL_READY_DEBUG_RUNNER_VERSION = "triqto.training.model_ready_debug_runner.v1"
_ALLOWED_DEBUG_TASKS = {"action_ranking", "joint_multitask", "hardware_masked"}
_LOSS_KEYS = (
    "action_should_act",
    "action_rank_distribution",
    "action_reward",
    "topology",
    "total",
)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        raise RuntimeError("training configuration requested CUDA, but CUDA is unavailable")
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


def _record_key(seed: int, split: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{item_id}".encode("utf-8")).hexdigest()


def _balanced_records(
    dataset: ModelReadyDataset,
    *,
    task: str,
    split: str,
    limit: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    if task not in _ALLOWED_DEBUG_TASKS:
        raise ValueError(f"unsupported model-ready debug task {task!r}")
    if split not in {"train", "validation"}:
        raise ValueError("debug selection may use only train or validation")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("selection limit must be a positive integer")
    rows = list(dataset.records_by_task_split.get((task, split), ()))
    if not rows:
        raise ValueError(f"no model-ready rows for task={task!r}, split={split!r}")
    if any(row.get("should_act") is None for row in rows):
        raise ValueError(f"task {task!r} contains rows without should_act targets")

    positives = sorted(
        (row for row in rows if bool(row["should_act"])),
        key=lambda row: _record_key(seed, split, str(row["view_item_id"])),
    )
    negatives = sorted(
        (row for row in rows if not bool(row["should_act"])),
        key=lambda row: _record_key(seed, split, str(row["view_item_id"])),
    )
    if not positives or not negatives:
        raise ValueError("debug subset must contain both should-act classes")

    target = min(limit, len(rows))
    selected: list[dict[str, Any]] = []
    positive_index = 0
    negative_index = 0
    choose_positive = True
    while len(selected) < target and (
        positive_index < len(positives) or negative_index < len(negatives)
    ):
        if choose_positive and positive_index < len(positives):
            selected.append(dict(positives[positive_index]))
            positive_index += 1
        elif not choose_positive and negative_index < len(negatives):
            selected.append(dict(negatives[negative_index]))
            negative_index += 1
        elif positive_index < len(positives):
            selected.append(dict(positives[positive_index]))
            positive_index += 1
        elif negative_index < len(negatives):
            selected.append(dict(negatives[negative_index]))
            negative_index += 1
        choose_positive = not choose_positive

    if len(selected) < 2 or len({bool(row["should_act"]) for row in selected}) != 2:
        raise ValueError("selected debug subset does not exercise both should-act classes")
    entity_ids = [str(row["entity_id"]) for row in selected]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("debug task selection contains duplicate entity rows")
    return tuple(selected)


def _epoch_batches(
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    ordered = list(records)
    if shuffle:
        ordered.sort(
            key=lambda row: _record_key(
                seed + epoch * 1_000_003,
                str(row["split"]),
                str(row["view_item_id"]),
            )
        )
    return tuple(
        tuple(ordered[index : index + batch_size])
        for index in range(0, len(ordered), batch_size)
    )


def _array_strings(value: np.ndarray | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in np.asarray(value).reshape(-1).tolist())


def build_model_ready_data_spec(
    dataset: ModelReadyDataset,
    model_config: TriQTOModelConfig,
    *,
    task: str,
) -> TrainingDataSpec:
    """Build checkpoint metadata for already-scaled model-ready inputs."""
    action_record = select_model_ready_record(dataset, task=task, split="train")
    action_artifact = load_model_ready_artifact(dataset, action_record)
    action_names = _array_strings(
        action_artifact.inputs.get("x_action_candidate_feature_names")
    )
    if len(action_names) != model_config.action_candidate_feature_dim:
        raise ValueError("model-ready action feature names do not match model width")

    topology_names: tuple[str, ...] = ()
    try:
        topology_record = select_model_ready_record(
            dataset,
            task="joint_multitask",
            split="train",
            topology_required=True,
        )
        topology_artifact = load_model_ready_artifact(dataset, topology_record)
        combined = _array_strings(
            topology_artifact.inputs.get("x_topology_feature_names")
        )
        alignment = tuple(
            f"alignment:{name}"
            for name in _array_strings(
                topology_artifact.inputs.get("x_topology_alignment_feature_names")
            )
        )
        topology_names = combined + alignment
        if len(topology_names) != CANONICAL_TOPOLOGY_INPUT_DIM:
            raise ValueError("canonical topology checkpoint vocabulary must have width 121")
        if len(set(topology_names)) != len(topology_names):
            raise ValueError("canonical topology checkpoint vocabulary contains duplicates")
    except LookupError:
        topology_names = ()

    backend_names = _array_strings(action_artifact.inputs.get("x_backend_feature_names"))
    if not backend_names:
        backend_names = tuple(
            f"backend_feature_{index}" for index in range(model_config.backend_input_dim)
        )
    if len(backend_names) != model_config.backend_input_dim:
        raise ValueError("backend feature names do not match model width")

    spec = TrainingDataSpec(
        training_view_dataset_id=dataset.training_view_dataset_id,
        distortion_labels=DISTORTION_LABELS,
        distortion_mapping=tuple(sorted(DISTORTION_TO_COARSE_LABEL.items())),
        action_edit_types=ACTION_EDIT_TYPES,
        action_edit_mapping=tuple(sorted(ACTION_EDIT_TYPE_MAP.items())),
        action_feature_names=action_names,
        action_feature_mean=tuple(0.0 for _ in action_names),
        action_feature_std=tuple(1.0 for _ in action_names),
        topology_feature_names=topology_names,
        topology_feature_mean=tuple(0.0 for _ in topology_names),
        topology_feature_std=tuple(1.0 for _ in topology_names),
        backend_feature_names=backend_names,
        backend_feature_mean=tuple(0.0 for _ in backend_names),
        backend_feature_std=tuple(1.0 for _ in backend_names),
        topology_input_dim=model_config.topology_input_dim,
        normalize_action_features=False,
        normalize_topology_features=False,
        normalize_backend_features=False,
        adapter_version=MODEL_READY_ADAPTER_VERSION,
    )
    spec.validate()
    return spec


def _selected_target_index(targets: ModelReadyActionTargets, graph: int) -> int | None:
    local = (targets.candidate_batch == graph) & targets.candidate_selected_mask
    indices = torch.nonzero(local, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    if indices.numel() != 1:
        raise ValueError("ranking target selects more than one candidate")
    return int(indices[0])


def _run_split(
    *,
    dataset: ModelReadyDataset,
    records: Sequence[dict[str, Any]],
    model: TriQTOModel,
    model_config: TriQTOModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler: DeterministicLRScheduler | None,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    if training != (scheduler is not None):
        raise ValueError("optimizer and scheduler must either both be present or both absent")
    model.train(training)
    batches = _epoch_batches(
        records,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        epoch=epoch,
        shuffle=training,
    )
    loss_sums = {name: 0.0 for name in _LOSS_KEYS}
    item_count = 0
    positive_count = 0
    ranking_active_count = 0
    gate_correct = 0
    ranking_correct = 0
    gradient_norm_sum = 0.0
    optimizer_steps = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in batches:
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            for record in batch:
                artifact = load_model_ready_artifact(dataset, record)
                example = build_model_ready_example(artifact, model_config)
                example = _move_tree(example, device)
                output = model(example.model_batch)
                losses = compute_model_ready_action_losses(
                    output,
                    example.action_targets,
                    should_act_weight=training_config.loss.action_selection_weight,
                    ranking_weight=training_config.loss.action_rank_distribution_weight,
                    reward_weight=training_config.loss.action_reward_weight,
                )
                if training:
                    (losses["total"] / len(batch)).backward()
                item_count += 1
                for name in _LOSS_KEYS:
                    loss_sums[name] += float(losses[name].detach().cpu())
                target_positive = bool(example.action_targets.should_act[0] > 0.5)
                positive_count += int(target_positive)
                predicted_positive = bool(
                    output.action_ranking.should_act_probability[0].detach() >= 0.5
                )
                gate_correct += int(predicted_positive == target_positive)
                if bool(example.action_targets.ranking_loss_mask[0]):
                    ranking_active_count += 1
                    candidate_mask = (
                        example.action_targets.candidate_target_mask
                        & output.action_ranking.candidate_available_mask
                    )
                    candidate_indices = torch.nonzero(
                        candidate_mask, as_tuple=False
                    ).flatten()
                    if candidate_indices.numel() == 0:
                        raise ValueError("ranking-active debug item has no candidate")
                    best_local = int(
                        candidate_indices[
                            torch.argmax(
                                output.action_ranking.candidate_scores.index_select(
                                    0, candidate_indices
                                )
                            )
                        ]
                    )
                    target_index = _selected_target_index(example.action_targets, 0)
                    ranking_correct += int(target_index == best_local)
            if training:
                assert optimizer is not None and scheduler is not None
                gradient_norm_sum += clip_gradient_norm(
                    model, training_config.max_gradient_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                global_step += 1

    if item_count == 0:
        raise ValueError("debug split contains no items")
    metrics: dict[str, Any] = {
        "item_count": item_count,
        "logical_batch_count": len(batches),
        "optimizer_steps": optimizer_steps,
        "gradient_norm_mean": (
            gradient_norm_sum / optimizer_steps if optimizer_steps else 0.0
        ),
        "should_act_positive_count": positive_count,
        "should_act_negative_count": item_count - positive_count,
        "should_act_accuracy": gate_correct / item_count,
        "ranking_active_count": ranking_active_count,
        "ranking_top1_accuracy": (
            ranking_correct / ranking_active_count if ranking_active_count else None
        ),
        "losses": {
            name: loss_sums[name] / item_count for name in _LOSS_KEYS
        },
    }
    for value in metrics["losses"].values():
        if not math.isfinite(float(value)):
            raise FloatingPointError("debug epoch produced a non-finite loss")
    return metrics, global_step


def _selection_payload(
    *,
    task: str,
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    return {
        "policy": "balanced_debug_only_not_representative",
        "task": task,
        "seed": seed,
        "train_view_item_ids": [str(row["view_item_id"]) for row in train_records],
        "validation_view_item_ids": [
            str(row["view_item_id"]) for row in validation_records
        ],
        "test_rows_used": False,
    }


def run_model_ready_debug_training(
    *,
    model_ready_root: str | Path,
    output_root: str | Path,
    model_config: TriQTOModelConfig,
    training_config: TrainingConfig,
    task: str = "action_ranking",
    train_items: int = 64,
    validation_items: int = 32,
) -> dict[str, Any]:
    """Run a small immutable train/validation experiment and publish checkpoints."""
    if not isinstance(model_config, TriQTOModelConfig):
        raise TypeError("model_config must be TriQTOModelConfig")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be TrainingConfig")
    if model_config.topology_input_dim != CANONICAL_TOPOLOGY_INPUT_DIM:
        raise ValueError("model-ready debug training requires topology_input_dim=121")
    if training_config.topology_loss_weight != 0.0 or training_config.loss.topology_weight != 0.0:
        raise ValueError("lambda_top must remain exactly 0.0")
    configured_tasks = tuple(
        task_name for stage in training_config.stages for task_name in stage.tasks
    )
    if set(configured_tasks) != {task}:
        raise ValueError(
            "model-ready debug training config must contain only the selected task"
        )
    epochs = sum(stage.epochs for stage in training_config.stages)
    if epochs <= 0:
        raise ValueError("debug training must contain at least one epoch")
    if training_config.gradient_accumulation_steps != 1:
        raise ValueError("debug runner requires gradient_accumulation_steps=1")
    if training_config.checkpoint_every_epochs != 1:
        raise ValueError("debug runner requires checkpoint_every_epochs=1")
    if training_config.early_stopping_patience != 0:
        raise ValueError("debug runner does not apply early stopping")
    if not training_config.keep_best_checkpoint:
        raise ValueError("debug runner requires keep_best_checkpoint=true")
    if train_items > training_config.max_items or validation_items > training_config.max_items:
        raise ValueError("debug selection exceeds training_config.max_items")

    dataset = load_model_ready_dataset(model_ready_root)
    source_manifest_path = (
        dataset.root / "manifests" / "processed_item_manifest.parquet"
    )
    source_manifest_hash_before = sha256_file(source_manifest_path)
    train_records = _balanced_records(
        dataset,
        task=task,
        split="train",
        limit=train_items,
        seed=training_config.seed,
    )
    validation_records = _balanced_records(
        dataset,
        task=task,
        split="validation",
        limit=validation_items,
        seed=training_config.seed + 17,
    )
    selection = _selection_payload(
        task=task,
        train_records=train_records,
        validation_records=validation_records,
        seed=training_config.seed,
    )
    data_spec = build_model_ready_data_spec(dataset, model_config, task=task)

    identity = {
        "schema": MODEL_READY_DEBUG_SCHEMA,
        "runner_version": MODEL_READY_DEBUG_RUNNER_VERSION,
        "training_view_dataset_id": dataset.training_view_dataset_id,
        "source_manifest_sha256": dataset.manifest_sha256,
        "model_config": model_config_to_dict(model_config),
        "training_config": training_config_to_dict(training_config),
        "data_spec_sha256": data_spec.content_hash,
        "selection": selection,
    }
    recipe_id = make_deterministic_id("modelreadydebugrecipe", identity)
    operational_id = training_operational_config_id(training_config)
    run_id = make_deterministic_id(
        "modelreadydebugrun",
        {
            "recipe_id": recipe_id,
            "operational_config_id": operational_id,
        },
    )
    output_parent = Path(output_root).expanduser().resolve()
    source_root = dataset.root.resolve()
    if output_parent == source_root or output_parent.is_relative_to(source_root):
        raise ValueError("debug training output must live outside the model-ready source")
    final_root = output_parent / run_id
    completion_path = final_root / "model_ready_debug_complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("complete") is True:
            return {"status": "already_complete", "output_root": str(final_root), **completion}
    if final_root.exists():
        raise FileExistsError(f"incomplete debug output already exists: {final_root}")

    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _seed_everything(training_config.seed, training_config.deterministic_algorithms)
        device = _resolve_device(training_config.device)
        model = TriQTOModel(model_config).to(device)
        optimizer = build_optimizer(model, training_config.optimizer)
        steps_per_epoch = math.ceil(len(train_records) / training_config.batch_size)
        scheduler = DeterministicLRScheduler(
            optimizer,
            training_config.scheduler,
            total_steps=steps_per_epoch * epochs,
        )

        schema_id = training_schema_id()
        architecture_id = model_architecture_id(model_config)
        model_config_identifier = model_config_id(model_config)
        global_step = 0
        best_epoch = 0
        best_validation_loss = math.inf
        best_checkpoint: dict[str, Any] | None = None
        epoch_records: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []

        _atomic_write_json(staging / "manifests" / "selection.json", selection)
        _atomic_write_json(staging / "manifests" / "identity.json", identity)
        _atomic_write_json(
            staging / "manifests" / "model_config.json",
            model_config_to_dict(model_config),
        )
        _atomic_write_json(
            staging / "manifests" / "training_config.json",
            training_config_to_dict(training_config),
        )
        _atomic_write_json(
            staging / "manifests" / "data_spec.json", data_spec.to_dict()
        )

        for epoch in range(epochs):
            train_metrics, global_step = _run_split(
                dataset=dataset,
                records=train_records,
                model=model,
                model_config=model_config,
                training_config=training_config,
                device=device,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
            )
            validation_metrics, global_step = _run_split(
                dataset=dataset,
                records=validation_records,
                model=model,
                model_config=model_config,
                training_config=training_config,
                device=device,
                epoch=epoch,
                optimizer=None,
                scheduler=None,
                global_step=global_step,
            )
            validation_loss = float(validation_metrics["losses"]["total"])
            epoch_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": scheduler.learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
                "lambda_top": 0.0,
            }
            epoch_records.append(epoch_payload)
            _atomic_write_json(
                staging / "metrics" / f"epoch-{epoch:03d}.json", epoch_payload
            )

            improved = validation_loss < best_validation_loss
            checkpoint_best_loss = (
                validation_loss if improved else best_validation_loss
            )
            checkpoint_best_epoch = epoch if improved else best_epoch
            epoch_checkpoint_path = (
                staging / "artifacts" / "checkpoints" / f"epoch-{epoch:03d}.npz"
            )
            epoch_checkpoint = save_training_checkpoint(
                epoch_checkpoint_path,
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
                epoch_completed=epoch,
                stage_index=0,
                global_step=global_step,
                best_validation_loss=checkpoint_best_loss,
                best_epoch=checkpoint_best_epoch,
                kind="epoch",
            )
            checkpoints.append(
                {
                    **epoch_checkpoint,
                    "kind": "epoch",
                    "epoch": epoch,
                    "artifact_ref": epoch_checkpoint_path.relative_to(staging).as_posix(),
                }
            )
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_path = (
                    staging / "artifacts" / "checkpoints" / f"best-{epoch:03d}.npz"
                )
                best_checkpoint = save_training_checkpoint(
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
                    epoch_completed=epoch,
                    stage_index=0,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    best_epoch=best_epoch,
                    kind="best",
                )
                best_checkpoint = {
                    **best_checkpoint,
                    "kind": "best",
                    "epoch": epoch,
                    "artifact_ref": best_path.relative_to(staging).as_posix(),
                }
                checkpoints.append(best_checkpoint)

        final_checkpoint_path = staging / "artifacts" / "checkpoints" / "final.npz"
        final_checkpoint = save_training_checkpoint(
            final_checkpoint_path,
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
            epoch_completed=epochs - 1,
            stage_index=0,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            kind="final",
        )
        final_record = {
            **final_checkpoint,
            "kind": "final",
            "epoch": epochs - 1,
            "artifact_ref": final_checkpoint_path.relative_to(staging).as_posix(),
        }
        checkpoints.append(final_record)

        source_manifest_hash_after = sha256_file(source_manifest_path)
        if source_manifest_hash_after != source_manifest_hash_before:
            raise RuntimeError("model-ready source manifest changed during debug training")
        summary = {
            "schema": MODEL_READY_DEBUG_SCHEMA,
            "runner_version": MODEL_READY_DEBUG_RUNNER_VERSION,
            "complete": True,
            "run_id": run_id,
            "recipe_id": recipe_id,
            "operational_config_id": operational_id,
            "training_schema_id": schema_id,
            "model_architecture_id": architecture_id,
            "model_config_id": model_config_identifier,
            "training_view_dataset_id": dataset.training_view_dataset_id,
            "source_manifest_sha256": source_manifest_hash_after,
            "source_root": str(dataset.root),
            "task": task,
            "epochs": epochs,
            "global_step": global_step,
            "train_items": len(train_records),
            "validation_items": len(validation_records),
            "test_rows_used": False,
            "selection_policy": selection["policy"],
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_checkpoint": best_checkpoint,
            "final_checkpoint": final_record,
            "checkpoints": checkpoints,
            "epoch_metrics": epoch_records,
            "lambda_top": 0.0,
            "device": str(device),
            "microbatch_policy": (
                "one variable-size graph forward at a time; gradients accumulated "
                "across each logical batch"
            ),
            "scientific_boundaries": {
                "balanced_subset_metrics_representative": False,
                "test_split_used": False,
                "topology_loss_active": False,
                "action_head_topology_enabled": False,
                "born_prediction_head_topology_enabled": False,
                "source_mutated": False,
            },
        }
        _atomic_write_json(staging / "reports" / "summary.json", summary)
        _atomic_write_json(
            staging / "model_ready_debug_complete.json",
            {key: value for key, value in summary.items() if key != "epoch_metrics"},
        )
        os.replace(staging, final_root)
        return {"status": "complete", "output_root": str(final_root), **summary}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "MODEL_READY_DEBUG_RUNNER_VERSION",
    "MODEL_READY_DEBUG_SCHEMA",
    "build_model_ready_data_spec",
    "run_model_ready_debug_training",
]
