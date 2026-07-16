"""Read-only validation of completed Phase 7, Phase 8, and Phase 9 sources."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from triqto.actions import (
    action_engine_id,
    action_operational_config_id,
    action_schema_id,
    load_action_artifact,
    load_action_config,
    load_action_engine_sources,
    load_candidate_circuit,
    load_rollout_artifact,
    validate_action_dataset_joins,
)
from triqto.actions.sharded_artifacts import ShardedActionReader
from triqto.graph import snapshot_managed_files
from triqto.graph.utils import (
    ensure_sorted_unique_strings,
    require_mapping,
    require_nonblank,
    resolve_safe_file,
    strict_json_load,
)
from triqto.storage import (
    ActionCandidateRecordV1,
    ActionRolloutRecord,
    ManifestReader,
)

from .models import BaselineSources, CompletedActionDataset

_ACTION_MARKER_KEYS = {
    "complete",
    "source_scientific_generation_id",
    "graph_conversion_id",
    "action_engine_id",
    "operational_config_id",
    "action_schema_id",
    "candidate_count",
    "rollout_count",
    "phase7_snapshot_hash",
    "graph_snapshot_hash",
    "managed_files",
}
_ACTION_REQUIRED_MANAGED = {
    "action_config.json",
    "action_summary.json",
    "action_complete.json",
    "manifests/action_candidate_manifest.parquet",
    "manifests/action_rollout_manifest.parquet",
}


def _strict_nonnegative_int(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"action_complete.json {name} must be an integer and not bool")
    if value < 0:
        raise ValueError(f"action_complete.json {name} must be nonnegative")
    return value


def _actual_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def load_completed_action_dataset(
    action_root: str | Path,
    *,
    phase7: Any,
    graph: Any,
) -> CompletedActionDataset:
    """Load and fully validate one immutable Phase 9 action dataset."""
    root = Path(action_root)
    if not root.exists():
        raise FileNotFoundError(f"Phase 9 action root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Phase 9 action root is not a directory: {root}")

    marker_path = root / "action_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Phase 9 completion marker missing: {marker_path}")
    marker_raw = strict_json_load(marker_path)
    marker = dict(require_mapping(marker_raw, "action_complete.json"))
    if set(marker) != _ACTION_MARKER_KEYS:
        raise ValueError(
            "action_complete.json key mismatch; "
            f"missing={sorted(_ACTION_MARKER_KEYS - set(marker))}, "
            f"unexpected={sorted(set(marker) - _ACTION_MARKER_KEYS)}"
        )
    if marker.get("complete") is not True:
        raise ValueError("action_complete.json complete must be exactly true")
    managed_raw = marker.get("managed_files")
    if not isinstance(managed_raw, list):
        raise TypeError("action_complete.json managed_files must be a list")
    managed_files = ensure_sorted_unique_strings(managed_raw, "managed_files")
    missing_required = _ACTION_REQUIRED_MANAGED - set(managed_files)
    if missing_required:
        raise ValueError(
            "action_complete.json is missing required files: "
            f"{sorted(missing_required)}"
        )
    for reference in managed_files:
        resolve_safe_file(root, reference, f"managed_files[{reference!r}]")
    actual_files = _actual_file_set(root)
    if actual_files != set(managed_files):
        raise ValueError(
            "Phase 9 managed file inventory mismatch; "
            f"missing={sorted(set(managed_files) - actual_files)}, "
            f"unmanaged={sorted(actual_files - set(managed_files))}"
        )

    snapshot = snapshot_managed_files(root, managed_files)
    config = load_action_config(root / "action_config.json")
    expected_engine_id = action_engine_id(
        phase7.source_scientific_generation_id,
        graph.completion_marker["graph_conversion_id"],
        config,
    )
    expected_values = {
        "source_scientific_generation_id": phase7.source_scientific_generation_id,
        "graph_conversion_id": graph.completion_marker["graph_conversion_id"],
        "action_engine_id": expected_engine_id,
        "operational_config_id": action_operational_config_id(config),
        "action_schema_id": action_schema_id(),
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": graph.snapshot.aggregate_sha256,
    }
    for name, expected in expected_values.items():
        require_nonblank(marker.get(name), f"action_complete.json {name}")
        if marker.get(name) != expected:
            raise ValueError(f"action_complete.json {name} mismatch")

    summary_raw = strict_json_load(root / "action_summary.json")
    summary = dict(require_mapping(summary_raw, "action_summary.json"))
    for name, expected in expected_values.items():
        if summary.get(name) != expected:
            raise ValueError(f"action_summary.json {name} mismatch")

    reader = ManifestReader(root / "manifests")
    candidate_records = reader.read_typed_records(
        "action_candidate_manifest", ActionCandidateRecordV1
    )
    rollout_records = reader.read_typed_records(
        "action_rollout_manifest", ActionRolloutRecord
    )
    if _strict_nonnegative_int(marker, "candidate_count") != len(candidate_records):
        raise ValueError("action_complete.json candidate_count mismatch")
    if _strict_nonnegative_int(marker, "rollout_count") != len(rollout_records):
        raise ValueError("action_complete.json rollout_count mismatch")
    if summary.get("candidate_count") != len(candidate_records):
        raise ValueError("action_summary.json candidate_count mismatch")
    if summary.get("rollout_count") != len(rollout_records):
        raise ValueError("action_summary.json rollout_count mismatch")

    candidates_by_id: dict[str, Any] = {}
    circuits_by_id: dict[str, Any] = {}
    candidate_record_by_action: dict[str, Any] = {}
    rollouts_by_id: dict[str, Any] = {}
    rollouts_by_sample: dict[str, list[Any]] = {}

    with ShardedActionReader(root, config) as shard_reader:
        for record in candidate_records:
            record.validate()
            if record.action_id in candidates_by_id:
                raise ValueError(f"Duplicate Phase 9 action {record.action_id}")
            if record.candidate_circuit_id in circuits_by_id:
                raise ValueError(
                    f"Duplicate Phase 9 candidate circuit {record.candidate_circuit_id}"
                )
            action_path = resolve_safe_file(
                root,
                record.action_ref,
                f"ActionCandidateRecordV1 {record.action_id}.action_ref",
            )
            circuit_path = resolve_safe_file(
                root,
                record.circuit_ref,
                f"ActionCandidateRecordV1 {record.action_id}.circuit_ref",
            )
            if record.action_ref.endswith(".zip"):
                candidate = shard_reader.load_candidate(
                    record.action_ref,
                    record.action_id,
                    record.content_hash,
                )
            else:
                candidate = load_action_artifact(
                    action_path,
                    config,
                    record.content_hash,
                )
            if record.circuit_ref.endswith(".zip"):
                circuit = shard_reader.load_circuit(
                    record.circuit_ref,
                    record.candidate_circuit_id,
                    record.circuit_hash,
                )
            else:
                circuit = load_candidate_circuit(
                    circuit_path,
                    record.circuit_hash,
                )
            candidates_by_id[record.action_id] = candidate
            circuits_by_id[record.candidate_circuit_id] = circuit
            candidate_record_by_action[record.action_id] = record

        for record in rollout_records:
            record.validate()
            candidate_record = candidate_record_by_action.get(record.action_id)
            if candidate_record is None:
                raise ValueError(
                    f"Action rollout {record.rollout_id} references missing candidate"
                )
            circuit = circuits_by_id.get(record.candidate_circuit_id)
            if circuit is None:
                raise ValueError(
                    f"Action rollout {record.rollout_id} references missing circuit"
                )
            rollout_path = resolve_safe_file(
                root,
                record.rollout_ref,
                f"ActionRolloutRecord {record.rollout_id}.rollout_ref",
            )
            if record.rollout_ref.endswith(".zip"):
                rollout = shard_reader.load_rollout(
                    record.rollout_ref,
                    record.rollout_id,
                    circuit,
                    record.content_hash,
                )
            else:
                rollout = load_rollout_artifact(
                    rollout_path,
                    circuit,
                    record.content_hash,
                )
            if rollout.rollout_id in rollouts_by_id:
                raise ValueError(f"Duplicate Phase 9 rollout {rollout.rollout_id}")
            rollouts_by_id[rollout.rollout_id] = rollout
            rollouts_by_sample.setdefault(rollout.sample_id, []).append(rollout)

    validate_action_dataset_joins(
        list(candidate_records),
        list(rollout_records),
        candidates_by_id=candidates_by_id,
        rollouts_by_id=rollouts_by_id,
        source_samples=phase7.samples,
        graph_pair_records=graph.pair_records,
        config=config,
    )
    normalized_by_sample: dict[str, tuple[Any, ...]] = {}
    for sample_id, rollouts in rollouts_by_sample.items():
        normalized_by_sample[sample_id] = tuple(
            sorted(rollouts, key=lambda item: (item.rank, item.action_id))
        )
    source_sample_ids = {sample.sample_id for sample in phase7.samples}
    if set(normalized_by_sample) != source_sample_ids:
        raise ValueError("Phase 9 rollouts do not cover the Phase 7 samples exactly")

    return CompletedActionDataset(
        root=root,
        config=config,
        completion_marker=marker,
        summary=summary,
        candidate_records=list(candidate_records),
        rollout_records=list(rollout_records),
        candidates_by_id=candidates_by_id,
        circuits_by_id=circuits_by_id,
        rollouts_by_id=rollouts_by_id,
        rollouts_by_sample_id=normalized_by_sample,
        managed_files=managed_files,
        snapshot=snapshot,
    )


def load_baseline_sources(
    phase7_root: str | Path,
    graph_root: str | Path,
    action_root: str | Path,
) -> BaselineSources:
    """Cross-validate the exact Phase 7/8/9 chain consumed by Phase 10."""
    action_sources = load_action_engine_sources(phase7_root, graph_root)
    action = load_completed_action_dataset(
        action_root,
        phase7=action_sources.phase7,
        graph=action_sources.graph,
    )
    return BaselineSources(
        phase7=action_sources.phase7,
        graph=action_sources.graph,
        action=action,
    )


def verify_baseline_source_snapshots(sources: BaselineSources) -> None:
    """Prove no managed source file changed during Phase 10 work."""
    checks = (
        (
            "Phase 7",
            sources.phase7.source_root,
            sources.phase7.source_snapshot,
        ),
        ("Phase 8", sources.graph.root, sources.graph.snapshot),
        ("Phase 9", sources.action.root, sources.action.snapshot),
    )
    for name, root, expected in checks:
        actual = snapshot_managed_files(
            root,
            tuple(entry.reference for entry in expected.entries),
        )
        if actual != expected:
            raise RuntimeError(f"{name} managed source files changed during Phase 10")


__all__ = [
    "load_baseline_sources",
    "load_completed_action_dataset",
    "verify_baseline_source_snapshots",
]
