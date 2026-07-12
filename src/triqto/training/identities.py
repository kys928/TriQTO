"""Deterministic Phase 14 training recipe, run, and checkpoint identities."""
from __future__ import annotations

from typing import Any

from triqto.core.ids import make_deterministic_id
from triqto.model import model_schema_id

from .config import TrainingConfig, training_config_to_dict
from .constants import (
    ACTION_EDIT_TYPE_MAP,
    DISTORTION_TO_COARSE_LABEL,
    INPUT_GROUP_TO_STREAM,
    PHASE12_TO_MODEL_HEAD,
    TRAINING_ADAPTER_VERSION,
    TRAINING_ARTIFACT_VERSION,
    TRAINING_BATCHING_VERSION,
    TRAINING_CHECKPOINT_VERSION,
    TRAINING_CURRICULUM_VERSION,
    TRAINING_LOSS_VERSION,
    TRAINING_SCHEMA_VERSION,
    TRAINING_SOURCE_CONTRACT_VERSION,
)


def training_schema_id() -> str:
    return make_deterministic_id(
        "trainschema",
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "source_contract": TRAINING_SOURCE_CONTRACT_VERSION,
            "adapter": TRAINING_ADAPTER_VERSION,
            "batching": TRAINING_BATCHING_VERSION,
            "loss": TRAINING_LOSS_VERSION,
            "curriculum": TRAINING_CURRICULUM_VERSION,
            "checkpoint": TRAINING_CHECKPOINT_VERSION,
            "artifacts": TRAINING_ARTIFACT_VERSION,
            "model_schema_id": model_schema_id(),
            "distortion_mapping": DISTORTION_TO_COARSE_LABEL,
            "edit_mapping": ACTION_EDIT_TYPE_MAP,
            "phase12_head_mapping": PHASE12_TO_MODEL_HEAD,
            "input_group_to_stream": INPUT_GROUP_TO_STREAM,
            "test_split_used_for_optimization": False,
            "audit_only_used_for_gradient": False,
            "topology_loss_weight": 0.0,
            "unsafe_pickle_checkpoint": False,
        },
    )


def scientific_training_config_payload(config: TrainingConfig) -> dict[str, Any]:
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    payload = training_config_to_dict(config)
    for key in (
        "run_name",
        "device",
        "num_workers",
        "checkpoint_every_epochs",
        "keep_best_checkpoint",
        "max_items",
        "max_nodes_per_batch",
        "max_edges_per_batch",
        "max_gates_per_batch",
        "max_candidates_per_batch",
        "max_outcomes_per_batch",
        "max_hilbert_amplitudes_per_batch",
    ):
        payload.pop(key)
    return payload


def operational_training_config_payload(config: TrainingConfig) -> dict[str, Any]:
    payload = training_config_to_dict(config)
    scientific = scientific_training_config_payload(config)
    return {key: value for key, value in payload.items() if key not in scientific}


def training_recipe_id(
    training_view_dataset_id: str,
    model_architecture_id: str,
    model_config_id: str,
    config: TrainingConfig,
    data_spec_hash: str,
) -> str:
    return make_deterministic_id(
        "trainrecipe",
        {
            "training_schema_id": training_schema_id(),
            "training_view_dataset_id": training_view_dataset_id,
            "model_architecture_id": model_architecture_id,
            "model_config_id": model_config_id,
            "scientific_training_config": scientific_training_config_payload(config),
            "data_spec_hash": data_spec_hash,
        },
    )


def training_operational_config_id(config: TrainingConfig) -> str:
    return make_deterministic_id(
        "trainopconfig",
        operational_training_config_payload(config),
    )


def training_run_id(recipe_id: str, operational_config_id: str) -> str:
    return make_deterministic_id(
        "trainrun",
        {
            "training_recipe_id": recipe_id,
            "operational_config_id": operational_config_id,
        },
    )


def training_checkpoint_id(
    run_id: str,
    *,
    epoch_completed: int,
    global_step: int,
    kind: str,
) -> str:
    return make_deterministic_id(
        "traincheckpoint",
        {
            "training_run_id": run_id,
            "epoch_completed": epoch_completed,
            "global_step": global_step,
            "kind": kind,
        },
    )


__all__ = [
    "operational_training_config_payload",
    "scientific_training_config_payload",
    "training_checkpoint_id",
    "training_operational_config_id",
    "training_recipe_id",
    "training_run_id",
    "training_schema_id",
]
