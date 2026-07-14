"""Safe Phase 15 item artifacts, typed manifests, reports, and publication."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from triqto.graph.utils import (
    resolve_safe_file,
    strict_json_load,
    write_strict_json,
)
from triqto.storage import ManifestReader, ManifestWriter
from triqto.storage.evaluation_schema import (
    EvaluationAggregateRecordV1,
    EvaluationBaselineRecordV1,
    EvaluationItemRecordV1,
)
from triqto.training import snapshot_managed_files

from .config import evaluation_config_to_dict, load_evaluation_config
from .constants import EVALUATION_ARTIFACT_VERSION
from .identities import evaluation_item_content_hash
from .models import EvaluationItemResult, EvaluationRunResult

_METADATA_ARRAY = "__metadata_json_utf8__"


def _json_bytes(payload: dict[str, Any]) -> np.ndarray:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def _artifact_payload(item: EvaluationItemResult) -> dict[str, Any]:
    return {
        "artifact_version": EVALUATION_ARTIFACT_VERSION,
        "evaluation_item_id": item.evaluation_item_id,
        "evaluation_run_id": item.evaluation_run_id,
        "view_item_id": item.view_item_id,
        "entity_id": item.entity_id,
        "task": item.task,
        "split": item.split,
        "ablation": item.ablation,
        "family": item.family,
        "n_qubits": item.n_qubits,
        "distortion_id": item.distortion_id,
        "metrics": item.metrics,
        "calibration": item.calibration,
        "predicted_action_id": item.predicted_action_id,
        "target_action_id": item.target_action_id,
        "target_action_rank": item.target_action_rank,
        "metadata": item.metadata,
    }


def save_evaluation_item_artifact(
    item: EvaluationItemResult,
    path: str | Path,
) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Evaluation artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, value in sorted(item.arrays.items()):
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise TypeError(f"Evaluation array {name} must not use object dtype")
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"Evaluation array {name} contains non-finite values")
        arrays[name] = np.ascontiguousarray(array)
    payload = _artifact_payload(item)
    content_hash = evaluation_item_content_hash(payload, arrays)
    metadata = {**payload, "content_hash": content_hash}
    np.savez_compressed(
        target,
        **arrays,
        **{_METADATA_ARRAY: _json_bytes(metadata)},
    )
    loaded = load_evaluation_item_artifact(
        target,
        expected_content_hash=content_hash,
    )
    if loaded["evaluation_item_id"] != item.evaluation_item_id:
        raise ValueError("Evaluation item immediate readback ID mismatch")
    return content_hash


def load_evaluation_item_artifact(
    path: str | Path,
    *,
    expected_content_hash: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    with np.load(target, allow_pickle=False) as artifact:
        if _METADATA_ARRAY not in artifact.files:
            raise ValueError("Evaluation artifact metadata is missing")
        metadata_array = artifact[_METADATA_ARRAY]
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise TypeError("Evaluation metadata must be one-dimensional uint8")
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
        arrays = {
            name: artifact[name].copy()
            for name in artifact.files
            if name != _METADATA_ARRAY
        }
    if not isinstance(metadata, dict):
        raise TypeError("Evaluation artifact metadata must be a dictionary")
    if metadata.get("artifact_version") != EVALUATION_ARTIFACT_VERSION:
        raise ValueError("Unsupported evaluation artifact version")
    content_hash = metadata.pop("content_hash", None)
    actual_hash = evaluation_item_content_hash(metadata, arrays)
    if content_hash != actual_hash:
        raise ValueError("Evaluation artifact logical content hash mismatch")
    if expected_content_hash is not None and content_hash != expected_content_hash:
        raise ValueError("Evaluation artifact expected content hash mismatch")
    for name, array in arrays.items():
        if array.dtype.kind == "O":
            raise TypeError(f"Evaluation array {name} uses object dtype")
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"Evaluation array {name} contains non-finite values")
    return {**metadata, "content_hash": content_hash, "arrays": arrays}


def _relative_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _item_record(item: EvaluationItemResult) -> EvaluationItemRecordV1:
    return EvaluationItemRecordV1(
        evaluation_item_id=item.evaluation_item_id,
        evaluation_run_id=item.evaluation_run_id,
        view_item_id=item.view_item_id,
        entity_id=item.entity_id,
        task=item.task,
        split=item.split,
        ablation=item.ablation,
        family=item.family,
        n_qubits=item.n_qubits,
        distortion_id=item.distortion_id,
        predicted_action_id=item.predicted_action_id,
        target_action_id=item.target_action_id,
        target_action_rank=item.target_action_rank,
        metrics=dict(item.metrics),
        calibration=dict(item.calibration),
        artifact_ref=item.artifact_ref,
        content_hash=item.content_hash,
        metadata=dict(item.metadata),
    )


def _aggregate_record(value: Any) -> EvaluationAggregateRecordV1:
    return EvaluationAggregateRecordV1(
        evaluation_aggregate_id=value.evaluation_aggregate_id,
        evaluation_run_id=value.evaluation_run_id,
        task=value.task,
        ablation=value.ablation,
        group_dimension=value.group_dimension,
        group_value=value.group_value,
        item_count=value.item_count,
        metrics=dict(value.metrics),
        calibration=dict(value.calibration),
        metadata=dict(value.metadata),
    )


def _baseline_record(value: Any) -> EvaluationBaselineRecordV1:
    return EvaluationBaselineRecordV1(
        evaluation_baseline_id=value.evaluation_baseline_id,
        evaluation_run_id=value.evaluation_run_id,
        sample_id=value.sample_id,
        baseline_name=value.baseline_name,
        learned_action_id=value.learned_action_id,
        baseline_action_id=value.baseline_action_id,
        objective_before=value.objective_before,
        learned_objective_after=value.learned_objective_after,
        baseline_objective_after=value.baseline_objective_after,
        learned_minus_baseline=value.learned_minus_baseline,
        learned_success=value.learned_success,
        baseline_success=value.baseline_success,
        baseline_privileged=value.baseline_privileged,
        metadata=dict(value.metadata),
    )


def _report_payloads(result: EvaluationRunResult) -> dict[str, Any]:
    full_overall = [
        value
        for value in result.aggregates
        if value.ablation == "full" and value.group_dimension == "overall"
    ]
    generalization = [
        value
        for value in result.aggregates
        if value.group_dimension in {"family", "n_qubits", "distortion_id"}
    ]
    calibration = [
        value
        for value in result.aggregates
        if value.calibration
    ]
    ablations = [
        value
        for value in result.aggregates
        if value.group_dimension == "overall"
    ]
    baseline_summary: dict[str, dict[str, float]] = {}
    for name in sorted(
        {row.baseline_name for row in result.baseline_comparisons}
    ):
        rows = [
            row
            for row in result.baseline_comparisons
            if row.baseline_name == name
        ]
        baseline_summary[name] = {
            "comparison_count": float(len(rows)),
            "learned_win_fraction": float(
                sum(row.learned_minus_baseline < 0.0 for row in rows)
                / len(rows)
            ),
            "mean_learned_minus_baseline": float(
                np.mean([row.learned_minus_baseline for row in rows])
            ),
            "baseline_privileged": float(rows[0].baseline_privileged),
        }
    return {
        "summary": {
            "claim_boundary": {
                "heldout_test_evaluation": True,
                "hardware_execution": False,
                "quantum_advantage": False,
                "universal_correction": False,
            },
            "overall": [asdict(value) for value in full_overall],
        },
        "generalization": [asdict(value) for value in generalization],
        "calibration": [asdict(value) for value in calibration],
        "ablations": [asdict(value) for value in ablations],
        "baselines": baseline_summary,
    }


def _snapshot_hash_from_marker(root: Path, marker_name: str) -> str:
    marker = strict_json_load(root / marker_name)
    if not isinstance(marker, dict):
        raise TypeError(f"{marker_name} must contain a mapping")
    managed = marker.get("managed_files")
    if not isinstance(managed, list):
        raise TypeError(f"{marker_name} managed_files must be a list")
    return snapshot_managed_files(root, managed).aggregate_sha256


def _verify_result_sources(result: EvaluationRunResult) -> None:
    phase12 = _snapshot_hash_from_marker(
        result.training_view_root,
        "training_view_complete.json",
    )
    if phase12 != result.summary["phase12_snapshot_hash"]:
        raise RuntimeError("Managed Phase 12 files changed before publication")
    phase14 = _snapshot_hash_from_marker(
        result.training_run_root,
        "training_complete.json",
    )
    if phase14 != result.summary["phase14_snapshot_hash"]:
        raise RuntimeError("Managed Phase 14 files changed before publication")
    if result.baseline_root is not None:
        phase10 = _snapshot_hash_from_marker(
            result.baseline_root,
            "baseline_complete.json",
        )
        if phase10 != result.summary["phase10_snapshot_hash"]:
            raise RuntimeError("Managed Phase 10 files changed before publication")


def write_evaluation_dataset(
    result: EvaluationRunResult,
    output_root: str | Path,
) -> Path:
    """Write a fresh immutable Phase 15 root through sibling staging."""
    if not isinstance(result, EvaluationRunResult):
        raise TypeError("result must be EvaluationRunResult")
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Evaluation output root already exists: {output}")
    for source in (
        result.training_view_root,
        result.training_run_root,
        result.phase7_root,
        result.graph_root,
        result.action_root,
        result.baseline_root,
    ):
        if source is None:
            continue
        resolved = Path(source).expanduser().resolve(strict=False)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError(
                f"Evaluation output root {output} overlaps source root {resolved}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected evaluation staging root: {staging}")
    staging.mkdir()
    try:
        write_strict_json(
            staging / "evaluation_config.json",
            evaluation_config_to_dict(result.config),
        )
        if load_evaluation_config(staging / "evaluation_config.json") != result.config:
            raise ValueError("Persisted evaluation config mismatch")

        item_records: list[EvaluationItemRecordV1] = []
        for item in sorted(
            result.item_results,
            key=lambda value: (
                value.task,
                value.ablation,
                value.view_item_id,
            ),
        ):
            item.artifact_ref = f"artifacts/items/{item.evaluation_item_id}.npz"
            item.content_hash = save_evaluation_item_artifact(
                item,
                staging / item.artifact_ref,
            )
            item_records.append(_item_record(item))
        aggregate_records = [
            _aggregate_record(value)
            for value in result.aggregates
        ]
        baseline_records = [
            _baseline_record(value)
            for value in result.baseline_comparisons
        ]

        writer = ManifestWriter(staging / "manifests")
        writer.write_records(
            "evaluation_item_manifest",
            item_records,
            overwrite=False,
        )
        writer.write_records(
            "evaluation_aggregate_manifest",
            aggregate_records,
            overwrite=False,
        )
        if baseline_records:
            writer.write_records(
                "evaluation_baseline_manifest",
                baseline_records,
                overwrite=False,
            )

        write_strict_json(staging / "evaluation_summary.json", result.summary)
        reports = _report_payloads(result)
        for name, payload in reports.items():
            write_strict_json(
                staging / "reports" / f"{name}.json",
                payload,
            )

        reader = ManifestReader(staging / "manifests")
        loaded_items = reader.read_typed_records(
            "evaluation_item_manifest",
            EvaluationItemRecordV1,
        )
        loaded_aggregates = reader.read_typed_records(
            "evaluation_aggregate_manifest",
            EvaluationAggregateRecordV1,
        )
        if len(loaded_items) != len(item_records):
            raise ValueError("Evaluation item manifest count mismatch")
        if len(loaded_aggregates) != len(aggregate_records):
            raise ValueError("Evaluation aggregate manifest count mismatch")
        item_ids: set[str] = set()
        for record in loaded_items:
            record.validate()
            if record.evaluation_item_id in item_ids:
                raise ValueError(
                    f"Duplicate evaluation item {record.evaluation_item_id}"
                )
            item_ids.add(record.evaluation_item_id)
            artifact = load_evaluation_item_artifact(
                resolve_safe_file(
                    staging,
                    record.artifact_ref,
                    "evaluation artifact",
                ),
                expected_content_hash=record.content_hash,
            )
            if artifact["evaluation_item_id"] != record.evaluation_item_id:
                raise ValueError("Evaluation manifest/artifact join mismatch")
            if artifact["evaluation_run_id"] != result.evaluation_run_id:
                raise ValueError("Evaluation artifact run ID mismatch")
            if artifact["split"] != "test":
                raise ValueError("Persisted evaluation artifact is not test split")
        aggregate_ids: set[str] = set()
        for record in loaded_aggregates:
            record.validate()
            if record.evaluation_aggregate_id in aggregate_ids:
                raise ValueError(
                    f"Duplicate evaluation aggregate {record.evaluation_aggregate_id}"
                )
            aggregate_ids.add(record.evaluation_aggregate_id)
        if baseline_records:
            loaded_baselines = reader.read_typed_records(
                "evaluation_baseline_manifest",
                EvaluationBaselineRecordV1,
            )
            if len(loaded_baselines) != len(baseline_records):
                raise ValueError("Evaluation baseline manifest count mismatch")
            baseline_ids: set[str] = set()
            for record in loaded_baselines:
                record.validate()
                if record.evaluation_baseline_id in baseline_ids:
                    raise ValueError(
                        "Duplicate evaluation baseline comparison "
                        f"{record.evaluation_baseline_id}"
                    )
                baseline_ids.add(record.evaluation_baseline_id)

        managed_before_marker = sorted(_relative_files(staging))
        managed_files = sorted(
            [*managed_before_marker, "evaluation_complete.json"]
        )
        completion = {
            "complete": True,
            "evaluation_schema_id": result.evaluation_schema_id,
            "evaluation_recipe_id": result.evaluation_recipe_id,
            "operational_config_id": result.operational_config_id,
            "evaluation_run_id": result.evaluation_run_id,
            "training_view_dataset_id": result.training_view_dataset_id,
            "training_run_id": result.training_run_id,
            "checkpoint_id": result.checkpoint_id,
            "heldout_split": "test",
            "evaluation_item_count": len(item_records),
            "aggregate_count": len(aggregate_records),
            "baseline_comparison_count": len(baseline_records),
            "phase12_snapshot_hash": result.summary["phase12_snapshot_hash"],
            "phase14_snapshot_hash": result.summary["phase14_snapshot_hash"],
            "phase10_snapshot_hash": result.summary["phase10_snapshot_hash"],
            "topology_loss_weight": 0.0,
            "hardware_execution_performed": False,
            "training_performed": False,
            "managed_files": managed_files,
        }
        write_strict_json(staging / "evaluation_complete.json", completion)
        if _relative_files(staging) != set(managed_files):
            raise ValueError("Phase 15 managed inventory mismatch")
        if strict_json_load(staging / "evaluation_complete.json") != completion:
            raise ValueError("Phase 15 completion marker readback mismatch")
        _verify_result_sources(result)
        if output.exists():
            raise FileExistsError(
                f"Evaluation output root appeared during publication: {output}"
            )
        os.replace(staging, output)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "load_evaluation_item_artifact",
    "save_evaluation_item_artifact",
    "write_evaluation_dataset",
]
