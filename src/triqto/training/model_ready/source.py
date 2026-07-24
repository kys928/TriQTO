"""Strict loading of immutable Phase 12 model-ready artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .types import (
    MODEL_READY_SOURCE_SCHEMA,
    TOPOLOGY_ATTACHMENT_SCHEMA,
    ModelReadyArtifact,
    ModelReadyDataset,
)

_ALLOWED_TASKS = {
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "joint_multitask",
    "hardware_masked",
}
_ALLOWED_SPLITS = {"train", "validation", "test"}
_METADATA_ARRAYS = {
    "schema_version",
    "view_item_id",
    "training_view_id",
    "task",
    "split",
    "split_group_id",
    "entity_id",
    "preprocessing_metadata_json_utf8",
    "topology_attachment_metadata_json_utf8",
}
_FORBIDDEN_X_TOKENS = (
    "privileged",
    "oracle",
    "target_reward",
    "target_rank",
    "target_selected",
    "dominates_baseline",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to load the model-ready manifest"
        ) from exc
    return pq


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_path(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"artifact_ref must be relative: {reference!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact_ref escapes model-ready root: {reference!r}") from exc
    return resolved


def scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar value")
    return str(array.reshape(-1)[0])


def scalar_bool(value: np.ndarray, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind != "b":
        raise TypeError(f"{name} must be one boolean scalar")
    return bool(array.reshape(-1)[0])


def load_model_ready_dataset(
    root: str | Path,
    *,
    verify_artifact_files: bool = True,
) -> ModelReadyDataset:
    """Load the strict model-ready manifest without deserializing NPZ bodies."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(base)
    completion_path = base / "preprocessed_complete.json"
    manifest_path = base / "manifests" / "processed_item_manifest.parquet"
    contract_path = base / "manifests" / "model_input_contract.json"
    weights_path = base / "manifests" / "should_act_class_weights.json"
    for path in (completion_path, manifest_path, contract_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    marker = _read_json(completion_path)
    if marker.get("complete") is not True:
        raise ValueError("preprocessed_complete.json complete must be true")
    if float(marker.get("lambda_top", 0.0)) != 0.0:
        raise ValueError("lambda_top must remain exactly 0.0")
    schema = marker.get("schema_version")
    if schema not in {MODEL_READY_SOURCE_SCHEMA, TOPOLOGY_ATTACHMENT_SCHEMA}:
        raise ValueError(f"unsupported model-ready schema_version {schema!r}")

    observed_manifest_hash = sha256_file(manifest_path)
    expected_manifest_hash = str(marker.get("processed_item_manifest_sha256") or "")
    if expected_manifest_hash and observed_manifest_hash != expected_manifest_hash:
        raise ValueError("processed_item_manifest.parquet SHA-256 mismatch")

    contract = _read_json(contract_path)
    if contract.get("model_inputs") != "arrays beginning with x_ only":
        raise ValueError("model input contract does not require x_* inputs")
    if contract.get("model_targets") != "arrays beginning with y_ only":
        raise ValueError("model input contract does not require y_* targets")
    attachment = contract.get("topology_attachment")
    if not isinstance(attachment, Mapping):
        raise ValueError("topology attachment contract is missing")
    if float(attachment.get("lambda_top", -1.0)) != 0.0:
        raise ValueError("topology attachment lambda_top must remain 0.0")
    head_policy = attachment.get("head_policy")
    if not isinstance(head_policy, Mapping):
        raise ValueError("topology head policy is missing")
    if bool(head_policy.get("joint_multitask.action_ranking", True)):
        raise ValueError("action ranking must be forbidden from topology")
    if bool(head_policy.get("joint_multitask.born_prediction", True)):
        raise ValueError("Born prediction must be forbidden from topology")

    raw_weights = _read_json(weights_path)
    class_weights = {
        name: float(raw_weights[name]) for name in ("negative", "positive")
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in class_weights.values()):
        raise ValueError("should-act class weights must be finite and positive")

    pq = _require_pyarrow()
    rows = pq.read_table(manifest_path).to_pylist()
    if not rows:
        raise ValueError("processed model-ready manifest is empty")
    required_columns = {
        "view_item_id",
        "training_view_id",
        "training_view_dataset_id",
        "task",
        "split",
        "split_group_id",
        "entity_id",
        "artifact_ref",
        "content_hash",
        "has_action_candidates",
        "deployable_candidate_count",
        "topology_available_mask",
    }
    missing = required_columns - set(rows[0])
    if missing:
        raise ValueError(f"processed manifest misses columns {sorted(missing)}")

    expected_count = marker.get("accepted_count")
    counts = marker.get("counts")
    if expected_count is None and isinstance(counts, Mapping):
        expected_count = counts.get("published_model_items")
    if expected_count is not None and int(expected_count) != len(rows):
        raise ValueError("processed manifest row count does not match completion marker")

    ids: set[str] = set()
    refs: set[str] = set()
    dataset_ids: set[str] = set()
    split_by_group: dict[str, str] = {}
    entity_identity: dict[str, tuple[str, str]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        item_id = str(row["view_item_id"])
        task = str(row["task"])
        split = str(row["split"])
        group = str(row["split_group_id"])
        entity = str(row["entity_id"])
        reference = str(row["artifact_ref"])
        content_hash = str(row["content_hash"])
        dataset_id = str(row["training_view_dataset_id"])
        if not item_id or item_id in ids:
            raise ValueError(f"duplicate or blank view_item_id {item_id!r}")
        if not reference or reference in refs:
            raise ValueError(f"duplicate or blank artifact_ref {reference!r}")
        if task not in _ALLOWED_TASKS:
            raise ValueError(f"unsupported model-ready task {task!r}")
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"unsupported model-ready split {split!r}")
        if not _HEX64.fullmatch(content_hash):
            raise ValueError(f"invalid content_hash for {item_id}")
        earlier_split = split_by_group.setdefault(group, split)
        if earlier_split != split:
            raise ValueError(f"split_group_id {group} crosses partitions")
        earlier_entity = entity_identity.setdefault(entity, (split, group))
        if earlier_entity != (split, group):
            raise ValueError(f"entity_id {entity} crosses split identities")
        artifact_path = safe_artifact_path(base, reference)
        if verify_artifact_files and not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        ids.add(item_id)
        refs.add(reference)
        dataset_ids.add(dataset_id)
        grouped.setdefault((task, split), []).append(row)
        normalized_rows.append(row)
    if len(dataset_ids) != 1:
        raise ValueError("manifest contains multiple training_view_dataset_id values")

    topology_marker = base / "topology_attachment_complete.json"
    if topology_marker.is_file():
        topology_completion = _read_json(topology_marker)
        boundaries = topology_completion.get("scientific_boundaries")
        if topology_completion.get("complete") is not True:
            raise ValueError("topology attachment marker is incomplete")
        if float(topology_completion.get("lambda_top", -1.0)) != 0.0:
            raise ValueError("topology attachment marker has nonzero lambda_top")
        if not isinstance(boundaries, Mapping):
            raise ValueError("topology scientific boundaries are missing")
        if bool(boundaries.get("action_head_topology_enabled", True)):
            raise ValueError("topology attachment enables the action head")
        if bool(boundaries.get("born_prediction_head_topology_enabled", True)):
            raise ValueError("topology attachment enables the Born head")
        if bool(boundaries.get("topology_supervised_target_present", True)):
            raise ValueError("topology supervised targets must remain absent")

    records_by_task_split = {
        key: tuple(sorted(values, key=lambda row: str(row["view_item_id"])))
        for key, values in grouped.items()
    }
    return ModelReadyDataset(
        root=base,
        completion_marker=marker,
        input_contract=contract,
        class_weights=class_weights,
        records=tuple(sorted(normalized_rows, key=lambda row: str(row["view_item_id"]))),
        records_by_task_split=records_by_task_split,
        training_view_dataset_id=next(iter(dataset_ids)),
        manifest_sha256=observed_manifest_hash,
    )


def load_model_ready_artifact(
    dataset: ModelReadyDataset,
    record: Mapping[str, Any],
) -> ModelReadyArtifact:
    """Hash-check one NPZ and partition it strictly into x, y, and metadata."""
    if not isinstance(dataset, ModelReadyDataset):
        raise TypeError("dataset must be ModelReadyDataset")
    row = dict(record)
    path = safe_artifact_path(dataset.root, str(row["artifact_ref"]))
    if sha256_file(path) != str(row["content_hash"]):
        raise ValueError(f"artifact SHA-256 mismatch for {row['view_item_id']}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    inputs: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    metadata: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if value.dtype.kind == "O":
            raise TypeError(f"object-dtype array is forbidden: {name}")
        if name in _METADATA_ARRAYS:
            metadata[name] = value
        elif name.startswith("x_"):
            lowered = name.lower()
            if any(token in lowered for token in _FORBIDDEN_X_TOKENS):
                raise ValueError(f"privileged/target-derived input is forbidden: {name}")
            inputs[name] = value
        elif name.startswith("y_"):
            targets[name] = value
        else:
            raise ValueError(f"unclassified scientific array {name!r}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"array {name} contains non-finite values")
    forbidden_topology_targets = [name for name in targets if name.startswith("y_topology")]
    if forbidden_topology_targets:
        raise ValueError(f"topology targets are forbidden: {forbidden_topology_targets}")
    expected_identity = {
        "view_item_id": str(row["view_item_id"]),
        "task": str(row["task"]),
        "split": str(row["split"]),
        "split_group_id": str(row["split_group_id"]),
        "entity_id": str(row["entity_id"]),
    }
    for name, expected in expected_identity.items():
        actual = scalar_text(metadata[name], name)
        if actual != expected:
            raise ValueError(f"artifact/manifest mismatch for {name}: {actual!r} != {expected!r}")
    return ModelReadyArtifact(
        record=row,
        inputs=inputs,
        targets=targets,
        metadata=metadata,
    )


def select_model_ready_record(
    dataset: ModelReadyDataset,
    *,
    task: str,
    split: str,
    topology_required: bool = False,
) -> dict[str, Any]:
    rows = dataset.records_by_task_split.get((task, split), ())
    for row in rows:
        if not topology_required or bool(row.get("topology_available_mask")):
            return dict(row)
    raise LookupError(
        f"no model-ready row for task={task!r}, split={split!r}, "
        f"topology_required={topology_required}"
    )
