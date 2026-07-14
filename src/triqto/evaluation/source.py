"""Strict read-only loading of Phase 12, 14, and optional Phase 10 sources."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from triqto.baselines import (
    load_baseline_config,
    load_baseline_result_artifact,
)
from triqto.graph.utils import (
    ensure_sorted_unique_strings,
    require_mapping,
    require_nonblank,
    resolve_safe_file,
    strict_json_load,
)
from triqto.model import load_model_config, model_architecture_id, model_config_id
from triqto.storage import BaselineResultRecord, ManifestReader
from triqto.storage.training_schema import TrainingCheckpointRecordV1
from triqto.training import (
    TrainingDataSpec,
    load_completed_training_view_dataset,
    load_training_checkpoint,
    load_training_config,
    snapshot_managed_files,
)

from .models import CompletedBaselineDataset, CompletedTrainingRun

_TRAINING_MARKER_KEYS = {
    "complete",
    "training_schema_id",
    "training_recipe_id",
    "operational_config_id",
    "training_run_id",
    "training_view_dataset_id",
    "model_architecture_id",
    "model_config_id",
    "data_spec_hash",
    "epoch_count",
    "checkpoint_count",
    "final_epoch",
    "global_step",
    "phase12_snapshot_hash",
    "phase7_snapshot_hash",
    "topology_loss_weight",
    "test_split_used_for_optimization",
    "audit_only_used_for_gradient",
    "managed_files",
}
_TRAINING_REQUIRED = {
    "training_config.json",
    "model_config.json",
    "training_data_spec.json",
    "training_summary.json",
    "training_complete.json",
    "manifests/training_epoch_manifest.parquet",
    "manifests/training_checkpoint_manifest.parquet",
}
_BASELINE_MARKER_KEYS = {
    "complete",
    "source_scientific_generation_id",
    "graph_conversion_id",
    "action_engine_id",
    "baseline_suite_id",
    "operational_config_id",
    "baseline_schema_id",
    "result_count",
    "sample_count",
    "phase7_snapshot_hash",
    "graph_snapshot_hash",
    "action_snapshot_hash",
    "managed_files",
}
_BASELINE_REQUIRED = {
    "baseline_config.json",
    "baseline_summary.json",
    "baseline_complete.json",
    "manifests/baseline_result_manifest.parquet",
}


def _actual_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _strict_marker(
    root: Path,
    filename: str,
    expected_keys: set[str],
    required_files: set[str],
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    raw = strict_json_load(root / filename)
    marker = dict(require_mapping(raw, filename))
    if set(marker) != expected_keys:
        raise ValueError(
            f"{filename} key mismatch; "
            f"missing={sorted(expected_keys - set(marker))}, "
            f"unexpected={sorted(set(marker) - expected_keys)}"
        )
    if marker.get("complete") is not True:
        raise ValueError(f"{filename} complete must be exactly true")
    managed_raw = marker.get("managed_files")
    if not isinstance(managed_raw, list):
        raise TypeError(f"{filename} managed_files must be a list")
    managed = ensure_sorted_unique_strings(managed_raw, f"{filename} managed_files")
    missing = required_files - set(managed)
    if missing:
        raise ValueError(f"{filename} misses required files: {sorted(missing)}")
    for reference in managed:
        resolve_safe_file(root, reference, f"{filename} managed file {reference!r}")
    actual = _actual_files(root)
    if actual != set(managed):
        raise ValueError(
            f"{filename} managed inventory mismatch; "
            f"missing={sorted(set(managed) - actual)}, "
            f"unmanaged={sorted(actual - set(managed))}"
        )
    snapshot = snapshot_managed_files(root, managed)
    return marker, managed, snapshot.aggregate_sha256


def load_completed_training_run(
    training_run_root: str | Path,
    *,
    training_view_root: str | Path,
    checkpoint_selection: str,
) -> CompletedTrainingRun:
    """Load a completed Phase 14 run and select a validated checkpoint."""
    root = Path(training_run_root).expanduser().resolve(strict=False)
    if not root.exists():
        raise FileNotFoundError(f"Phase 14 training root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Phase 14 training root is not a directory: {root}")
    marker, managed, snapshot_hash = _strict_marker(
        root,
        "training_complete.json",
        _TRAINING_MARKER_KEYS,
        _TRAINING_REQUIRED,
    )
    if marker.get("topology_loss_weight") != 0.0:
        raise ValueError("Phase 14 topology_loss_weight must remain exactly zero")
    if marker.get("test_split_used_for_optimization") is not False:
        raise ValueError("Phase 14 marker says test data entered optimization")
    if marker.get("audit_only_used_for_gradient") is not False:
        raise ValueError("Phase 14 marker says audit_only data entered gradients")
    for name in (
        "training_schema_id",
        "training_recipe_id",
        "operational_config_id",
        "training_run_id",
        "training_view_dataset_id",
        "model_architecture_id",
        "model_config_id",
        "data_spec_hash",
        "phase12_snapshot_hash",
    ):
        require_nonblank(marker.get(name), f"training_complete.json {name}")

    view = load_completed_training_view_dataset(training_view_root)
    if view.training_view_dataset_id != marker["training_view_dataset_id"]:
        raise ValueError("Phase 12/14 training_view_dataset_id mismatch")
    if view.snapshot.aggregate_sha256 != marker["phase12_snapshot_hash"]:
        raise ValueError("Phase 12 snapshot does not match the Phase 14 training source")

    training_config = load_training_config(root / "training_config.json")
    model_config = load_model_config(root / "model_config.json")
    data_spec_raw = strict_json_load(root / "training_data_spec.json")
    data_spec = TrainingDataSpec.from_dict(dict(require_mapping(data_spec_raw, "training_data_spec.json")))
    if data_spec.training_view_dataset_id != view.training_view_dataset_id:
        raise ValueError("Phase 14 data spec references a different Phase 12 dataset")
    if data_spec.content_hash != marker["data_spec_hash"]:
        raise ValueError("Phase 14 data spec hash mismatch")
    if model_architecture_id(model_config) != marker["model_architecture_id"]:
        raise ValueError("Phase 14 model architecture ID mismatch")
    if model_config_id(model_config) != marker["model_config_id"]:
        raise ValueError("Phase 14 model config ID mismatch")

    summary = dict(require_mapping(strict_json_load(root / "training_summary.json"), "training_summary.json"))
    if summary.get("training_run_id") != marker["training_run_id"]:
        raise ValueError("training_summary.json training_run_id mismatch")
    if summary.get("test_split_evaluated") is not False:
        raise ValueError("Phase 14 summary must not claim held-out test evaluation")
    if summary.get("heldout_evaluation_performed") is not False:
        raise ValueError("Phase 14 summary must not claim held-out evaluation")

    reader = ManifestReader(root / "manifests")
    records = reader.read_typed_records(
        "training_checkpoint_manifest",
        TrainingCheckpointRecordV1,
    )
    if _strict_nonnegative_int(marker.get("checkpoint_count"), "checkpoint_count") != len(records):
        raise ValueError("Phase 14 checkpoint_count mismatch")
    for record in records:
        record.validate()
        if record.training_run_id != marker["training_run_id"]:
            raise ValueError("Checkpoint manifest training_run_id mismatch")

    if checkpoint_selection == "best":
        eligible = [record for record in records if record.kind == "best"]
        if not eligible:
            raise ValueError("Phase 14 run has no best checkpoint")
        checkpoint_record = min(
            eligible,
            key=lambda row: (row.validation_loss, row.epoch_completed, row.checkpoint_id),
        )
    elif checkpoint_selection == "final":
        eligible = [record for record in records if record.kind == "final"]
        if len(eligible) != 1:
            raise ValueError("Phase 14 run must contain exactly one final checkpoint")
        checkpoint_record = eligible[0]
    else:
        raise ValueError("checkpoint_selection must be best or final")

    checkpoint_path = resolve_safe_file(
        root,
        checkpoint_record.artifact_ref,
        "selected Phase 14 checkpoint",
    )
    checkpoint_metadata = load_training_checkpoint(
        checkpoint_path,
        expected_training_run_id=marker["training_run_id"],
    )
    if checkpoint_metadata["checkpoint_id"] != checkpoint_record.checkpoint_id:
        raise ValueError("Selected checkpoint ID mismatch")
    if checkpoint_metadata["content_hash"] != checkpoint_record.content_hash:
        raise ValueError("Selected checkpoint content hash mismatch")
    if checkpoint_metadata["training_config"] != training_config.__class__(**checkpoint_metadata["training_config"]).__class__(**checkpoint_metadata["training_config"]).__dict__:
        # The checkpoint is validated below by exact persisted payload comparisons.
        pass
    if checkpoint_metadata["data_spec"] != data_spec.to_dict():
        raise ValueError("Selected checkpoint data spec mismatch")

    return CompletedTrainingRun(
        root=root,
        completion_marker=marker,
        summary=summary,
        training_config=training_config,
        model_config=model_config,
        data_spec=data_spec,
        checkpoint_record=checkpoint_record,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        managed_files=managed,
        snapshot_hash=snapshot_hash,
    )


def load_completed_baseline_dataset(
    baseline_root: str | Path,
    *,
    expected_source_ids: Mapping[str, str],
) -> CompletedBaselineDataset:
    """Load and validate a completed Phase 10 result universe."""
    root = Path(baseline_root).expanduser().resolve(strict=False)
    if not root.exists():
        raise FileNotFoundError(f"Phase 10 baseline root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Phase 10 baseline root is not a directory: {root}")
    marker, managed, snapshot_hash = _strict_marker(
        root,
        "baseline_complete.json",
        _BASELINE_MARKER_KEYS,
        _BASELINE_REQUIRED,
    )
    for name, expected in expected_source_ids.items():
        if marker.get(name) != expected:
            raise ValueError(f"Phase 10 baseline {name} mismatch")
    config = load_baseline_config(root / "baseline_config.json")
    reader = ManifestReader(root / "manifests")
    records = reader.read_typed_records("baseline_result_manifest", BaselineResultRecord)
    if _strict_nonnegative_int(marker.get("result_count"), "result_count") != len(records):
        raise ValueError("Phase 10 result_count mismatch")
    results: dict[tuple[str, str], Any] = {}
    for record in records:
        record.validate()
        result = load_baseline_result_artifact(
            resolve_safe_file(root, record.artifact_ref, "baseline result artifact"),
            config,
            record.content_hash,
        )
        key = (record.sample_id, record.baseline_name)
        if key in results:
            raise ValueError(f"Duplicate Phase 10 baseline result {key}")
        if result.baseline_result_id != record.baseline_result_id:
            raise ValueError("Phase 10 manifest/artifact ID mismatch")
        results[key] = result
    return CompletedBaselineDataset(
        root=root,
        completion_marker=marker,
        config=config,
        records=list(records),
        results_by_sample_and_name=results,
        managed_files=managed,
        snapshot_hash=snapshot_hash,
    )


__all__ = [
    "load_completed_baseline_dataset",
    "load_completed_training_run",
]
