"""Deterministic scientific and operational identities for Phase 15."""
from __future__ import annotations

import hashlib
from typing import Any

from triqto.core.ids import canonical_json, make_deterministic_id

from .config import EvaluationConfig, evaluation_config_to_dict
from .constants import (
    EVALUATION_ARTIFACT_VERSION,
    EVALUATION_METRIC_VERSION,
    EVALUATION_SCHEMA_VERSION,
)


def evaluation_schema_id() -> str:
    return make_deterministic_id(
        "evalschema",
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "artifact_version": EVALUATION_ARTIFACT_VERSION,
            "metric_version": EVALUATION_METRIC_VERSION,
            "heldout_split": "test",
            "hardware_execution": False,
        },
    )


def scientific_evaluation_config_payload(config: EvaluationConfig) -> dict[str, Any]:
    payload = evaluation_config_to_dict(config)
    for name in (
        "run_name",
        "batch_size",
        "max_items",
        "device",
    ):
        payload.pop(name, None)
    return payload


def evaluation_recipe_id(
    training_view_dataset_id: str,
    training_run_id: str,
    checkpoint_id: str,
    config: EvaluationConfig,
    *,
    baseline_suite_id: str | None,
) -> str:
    return make_deterministic_id(
        "evalrecipe",
        {
            "training_view_dataset_id": training_view_dataset_id,
            "training_run_id": training_run_id,
            "checkpoint_id": checkpoint_id,
            "evaluation_schema_id": evaluation_schema_id(),
            "scientific_config": scientific_evaluation_config_payload(config),
            "baseline_suite_id": baseline_suite_id,
        },
    )


def evaluation_operational_config_id(config: EvaluationConfig) -> str:
    return make_deterministic_id(
        "evalops",
        {
            "run_name": config.run_name,
            "batch_size": config.batch_size,
            "max_items": config.max_items,
            "device": config.device,
        },
    )


def evaluation_run_id(recipe_id: str, operational_id: str) -> str:
    return make_deterministic_id(
        "evalrun",
        {"recipe_id": recipe_id, "operational_id": operational_id},
    )


def evaluation_item_id(
    run_id: str,
    view_item_id: str,
    ablation: str,
) -> str:
    return make_deterministic_id(
        "evalitem",
        {
            "evaluation_run_id": run_id,
            "view_item_id": view_item_id,
            "ablation": ablation,
            "artifact_version": EVALUATION_ARTIFACT_VERSION,
        },
    )


def evaluation_aggregate_id(
    run_id: str,
    task: str,
    ablation: str,
    group_dimension: str,
    group_value: str,
) -> str:
    return make_deterministic_id(
        "evalagg",
        {
            "evaluation_run_id": run_id,
            "task": task,
            "ablation": ablation,
            "group_dimension": group_dimension,
            "group_value": group_value,
        },
    )


def evaluation_baseline_id(
    run_id: str,
    sample_id: str,
    task: str,
    baseline_name: str,
) -> str:
    return make_deterministic_id(
        "evalbase",
        {
            "evaluation_run_id": run_id,
            "sample_id": sample_id,
            "task": task,
            "baseline_name": baseline_name,
        },
    )


def evaluation_item_content_hash(payload: dict[str, Any], arrays: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(payload).encode("utf-8"))
    digest.update(b"\0")
    for name in sorted(arrays):
        array = arrays[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


__all__ = [
    "evaluation_aggregate_id",
    "evaluation_baseline_id",
    "evaluation_item_content_hash",
    "evaluation_item_id",
    "evaluation_operational_config_id",
    "evaluation_recipe_id",
    "evaluation_run_id",
    "evaluation_schema_id",
    "scientific_evaluation_config_payload",
]
