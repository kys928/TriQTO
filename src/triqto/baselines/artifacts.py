"""Strict NPZ artifacts and immutable publication for Phase 10 baselines."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np

from triqto.graph import snapshot_managed_files
from triqto.graph.utils import (
    json_copy,
    resolve_safe_file,
    strict_json_load,
    strict_json_loads,
    write_strict_json,
)
from triqto.storage.baseline_schema import BaselineResultRecord
from triqto.storage.manifest import ManifestReader, ManifestWriter

from .config import (
    BaselineSuiteConfig,
    load_baseline_config,
    save_baseline_config,
)
from .constants import (
    BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION,
    RESULT_ARRAY_NAMES,
    RESULT_METADATA_ARRAY_NAME,
)
from .identities import baseline_result_content_hash
from .models import BaselineResult, BaselineSuiteResult, BaselineWriteResult
from .validators import validate_baseline_dataset_joins, validate_baseline_result


def _json_bytes(payload: Mapping[str, Any]) -> np.ndarray:
    text = json.dumps(
        json_copy(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def _decode_json_bytes(array: np.ndarray, name: str) -> dict[str, Any]:
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional uint8 array")
    try:
        text = array.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    payload = strict_json_loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must decode to a JSON object")
    return dict(payload)


def _result_metadata(result: BaselineResult) -> dict[str, Any]:
    return {
        "artifact_schema_version": BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION,
        "baseline_result_id": result.baseline_result_id,
        "baseline_suite_id": result.baseline_suite_id,
        "sample_id": result.sample_id,
        "graph_pair_id": result.graph_pair_id,
        "baseline_name": result.baseline_name,
        "source_circuit_id": result.source_circuit_id,
        "clean_target_run_id": result.clean_target_run_id,
        "selected_action_id": result.selected_action_id,
        "objective_before": result.objective_before,
        "objective_after": result.objective_after,
        "objective_improvement": result.objective_improvement,
        "success": result.success,
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "metadata": result.metadata,
        "content_hash": baseline_result_content_hash(result),
    }


def save_baseline_result_artifact(
    result: BaselineResult,
    config: BaselineSuiteConfig,
    path: str | Path,
) -> Path:
    validate_baseline_result(result, config, require_hash=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        metric_names=result.metric_names,
        baseline_metric_values=result.baseline_metric_values,
        result_metric_values=result.result_metric_values,
        improvement_values=result.improvement_values,
        outcome_bitstrings=result.outcome_bitstrings,
        exact_probabilities=result.exact_probabilities,
        parameter_vector=result.parameter_vector,
        **{RESULT_METADATA_ARRAY_NAME: _json_bytes(_result_metadata(result))},
    )
    return target


def load_baseline_result_artifact(
    path: str | Path,
    config: BaselineSuiteConfig,
    expected_content_hash: str | None = None,
) -> BaselineResult:
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        expected_names = set(RESULT_ARRAY_NAMES) | {RESULT_METADATA_ARRAY_NAME}
        actual_names = set(payload.files)
        if actual_names != expected_names:
            raise ValueError(
                "Baseline result artifact array-name mismatch; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        arrays = {name: payload[name].copy() for name in RESULT_ARRAY_NAMES}
        metadata = _decode_json_bytes(
            payload[RESULT_METADATA_ARRAY_NAME], RESULT_METADATA_ARRAY_NAME
        )
    expected_metadata_keys = {
        "artifact_schema_version",
        "baseline_result_id",
        "baseline_suite_id",
        "sample_id",
        "graph_pair_id",
        "baseline_name",
        "source_circuit_id",
        "clean_target_run_id",
        "selected_action_id",
        "objective_before",
        "objective_after",
        "objective_improvement",
        "success",
        "evaluations",
        "iterations",
        "metadata",
        "content_hash",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError("Baseline result artifact metadata-key mismatch")
    if (
        metadata["artifact_schema_version"]
        != BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported baseline result artifact schema version")
    if not isinstance(metadata["metadata"], Mapping):
        raise TypeError("Baseline result metadata.metadata must be a mapping")
    result = BaselineResult(
        baseline_result_id=metadata["baseline_result_id"],
        baseline_suite_id=metadata["baseline_suite_id"],
        sample_id=metadata["sample_id"],
        graph_pair_id=metadata["graph_pair_id"],
        baseline_name=metadata["baseline_name"],
        source_circuit_id=metadata["source_circuit_id"],
        clean_target_run_id=metadata["clean_target_run_id"],
        selected_action_id=metadata["selected_action_id"],
        metric_names=arrays["metric_names"],
        baseline_metric_values=arrays["baseline_metric_values"],
        result_metric_values=arrays["result_metric_values"],
        improvement_values=arrays["improvement_values"],
        outcome_bitstrings=arrays["outcome_bitstrings"],
        exact_probabilities=arrays["exact_probabilities"],
        parameter_vector=arrays["parameter_vector"],
        objective_before=metadata["objective_before"],
        objective_after=metadata["objective_after"],
        objective_improvement=metadata["objective_improvement"],
        success=metadata["success"],
        evaluations=metadata["evaluations"],
        iterations=metadata["iterations"],
        metadata=dict(metadata["metadata"]),
        content_hash=metadata["content_hash"],
    )
    validate_baseline_result(result, config, require_hash=True)
    if expected_content_hash is not None and result.content_hash != expected_content_hash:
        raise ValueError("Baseline result content_hash does not match manifest")
    return result


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _verify_result_sources(result: BaselineSuiteResult) -> None:
    checks = (
        ("Phase 7", result.phase7_source_root, result.phase7_snapshot),
        ("Phase 8", result.graph_source_root, result.graph_snapshot),
        ("Phase 9", result.action_source_root, result.action_snapshot),
    )
    for name, root, expected in checks:
        actual = snapshot_managed_files(
            root,
            tuple(entry.reference for entry in expected.entries),
        )
        if actual != expected:
            raise RuntimeError(f"{name} managed source files changed during Phase 10")


def write_baseline_dataset(
    result: BaselineSuiteResult,
    output_root: str | Path,
) -> BaselineWriteResult:
    """Publish a validated Phase 10 result dataset into a fresh immutable root."""
    if not isinstance(result, BaselineSuiteResult):
        raise TypeError("result must be BaselineSuiteResult")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Baseline output root already exists: {output}")
    resolved_output = output.resolve()
    for source_name, source_root in (
        ("Phase 7", result.phase7_source_root),
        ("Phase 8", result.graph_source_root),
        ("Phase 9", result.action_source_root),
    ):
        resolved_source = Path(source_root).resolve()
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError(
                f"Baseline output root must not be inside the {source_name} source root"
            )
    _verify_result_sources(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected existing staging directory: {staging}")

    try:
        (staging / "manifests").mkdir(parents=True)
        (staging / "artifacts" / "results").mkdir(parents=True)
        managed: list[str] = []
        save_baseline_config(result.config, staging / "baseline_config.json")
        managed.append("baseline_config.json")
        write_strict_json(staging / "baseline_summary.json", result.summary)
        managed.append("baseline_summary.json")

        for item in result.results:
            reference = f"artifacts/results/{item.baseline_result_id}.npz"
            save_baseline_result_artifact(item, result.config, staging / reference)
            managed.append(reference)
        writer = ManifestWriter(staging / "manifests")
        writer.write_records(
            "baseline_result_manifest", result.result_records, overwrite=False
        )
        managed.append("manifests/baseline_result_manifest.parquet")

        persisted_config = load_baseline_config(staging / "baseline_config.json")
        if persisted_config != result.config:
            raise ValueError("Persisted baseline config mismatch")
        reader = ManifestReader(staging / "manifests")
        records = reader.read_typed_records(
            "baseline_result_manifest", BaselineResultRecord
        )
        loaded: dict[str, BaselineResult] = {}
        for record in records:
            record.validate()
            item = load_baseline_result_artifact(
                resolve_safe_file(
                    staging,
                    record.artifact_ref,
                    f"BaselineResultRecord {record.baseline_result_id}.artifact_ref",
                ),
                persisted_config,
                record.content_hash,
            )
            if item.baseline_result_id in loaded:
                raise ValueError(
                    f"Duplicate persisted baseline result {item.baseline_result_id}"
                )
            loaded[item.baseline_result_id] = item
        validate_baseline_dataset_joins(
            records,
            results_by_id=loaded,
            config=persisted_config,
        )

        if len(set(managed)) != len(managed):
            raise ValueError("Managed Phase 10 file inventory contains duplicates")
        expected_before_marker = set(managed)
        actual_before_marker = _relative_file_set(staging)
        if actual_before_marker != expected_before_marker:
            raise ValueError(
                "Staging baseline dataset inventory mismatch; "
                f"missing={sorted(expected_before_marker - actual_before_marker)}, "
                f"unexpected={sorted(actual_before_marker - expected_before_marker)}"
            )

        managed_files = tuple(sorted([*managed, "baseline_complete.json"]))
        completion = {
            "complete": True,
            "source_scientific_generation_id": (
                result.source_scientific_generation_id
            ),
            "graph_conversion_id": result.graph_conversion_id,
            "action_engine_id": result.action_engine_id,
            "baseline_suite_id": result.baseline_suite_id,
            "operational_config_id": result.operational_config_id,
            "baseline_schema_id": result.baseline_schema_id,
            "result_count": len(result.results),
            "sample_count": result.summary["source_sample_count"],
            "phase7_snapshot_hash": result.phase7_snapshot.aggregate_sha256,
            "graph_snapshot_hash": result.graph_snapshot.aggregate_sha256,
            "action_snapshot_hash": result.action_snapshot.aggregate_sha256,
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "baseline_complete.json", completion)
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed baseline file inventory does not match staging")
        if strict_json_load(staging / "baseline_complete.json") != completion:
            raise ValueError("baseline_complete.json content mismatch")
        _verify_result_sources(result)

        if output.exists():
            raise FileExistsError(
                f"Baseline output root appeared during publication: {output}"
            )
        os.replace(staging, output)
        manifest_paths = (
            output / "manifests" / "baseline_result_manifest.parquet",
        )
        artifact_paths = tuple(
            sorted(
                [
                    output / reference
                    for reference in managed_files
                    if reference.startswith("artifacts/")
                ],
                key=lambda path: path.as_posix(),
            )
        )
        written_paths = tuple(
            sorted(
                [output / reference for reference in managed_files],
                key=lambda path: path.as_posix(),
            )
        )
        return BaselineWriteResult(
            output_root=output,
            baseline_complete_path=output / "baseline_complete.json",
            manifest_paths=manifest_paths,
            artifact_paths=artifact_paths,
            written_paths=written_paths,
            managed_files=managed_files,
            result_count=len(result.results),
            sample_count=result.summary["source_sample_count"],
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "load_baseline_result_artifact",
    "save_baseline_result_artifact",
    "write_baseline_dataset",
]
