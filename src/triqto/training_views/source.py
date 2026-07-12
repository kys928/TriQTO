"""Read-only validation of completed Phase 7/8/9/11 sources for Phase 12."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from triqto.baselines import load_baseline_sources
from triqto.graph import snapshot_managed_files
from triqto.graph.utils import (
    ensure_sorted_unique_strings,
    require_mapping,
    require_nonblank,
    resolve_safe_file,
    strict_json_load,
)
from triqto.storage import ManifestReader, TopologyGroupRecordV1
from triqto.topology import (
    load_topology_config,
    load_topology_group_artifact,
    topology_audit_id,
    topology_operational_config_id,
    topology_schema_id,
    validate_topology_dataset_joins,
)

from .models import CompletedTopologyDataset, TrainingViewSources

_TOPOLOGY_MARKER_KEYS = {
    "complete",
    "source_scientific_generation_id",
    "graph_conversion_id",
    "action_engine_id",
    "topology_audit_id",
    "operational_config_id",
    "topology_schema_id",
    "group_count",
    "point_count",
    "phase7_snapshot_hash",
    "graph_snapshot_hash",
    "action_snapshot_hash",
    "topology_loss_weight",
    "managed_files",
}
_TOPOLOGY_REQUIRED_MANAGED = {
    "topology_config.json",
    "topology_summary.json",
    "topology_complete.json",
    "manifests/topology_group_manifest.parquet",
}


def _actual_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _strict_nonnegative_int(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"topology_complete.json {name} must be integer and not bool")
    if value < 0:
        raise ValueError(f"topology_complete.json {name} must be nonnegative")
    return value


def load_completed_topology_dataset(
    topology_root: str | Path,
    *,
    phase7: Any,
    graph: Any,
    action: Any,
) -> CompletedTopologyDataset:
    """Load and fully validate one immutable Phase 11 topology dataset."""
    root = Path(topology_root)
    if not root.exists():
        raise FileNotFoundError(f"Phase 11 topology root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Phase 11 topology root is not a directory: {root}")
    marker_path = root / "topology_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Phase 11 completion marker missing: {marker_path}")
    marker_raw = strict_json_load(marker_path)
    marker = dict(require_mapping(marker_raw, "topology_complete.json"))
    if set(marker) != _TOPOLOGY_MARKER_KEYS:
        raise ValueError(
            "topology_complete.json key mismatch; "
            f"missing={sorted(_TOPOLOGY_MARKER_KEYS - set(marker))}, "
            f"unexpected={sorted(set(marker) - _TOPOLOGY_MARKER_KEYS)}"
        )
    if marker.get("complete") is not True:
        raise ValueError("topology_complete.json complete must be exactly true")
    if marker.get("topology_loss_weight") != 0.0:
        raise ValueError("Phase 11 topology_loss_weight must remain exactly zero")
    managed_raw = marker.get("managed_files")
    if not isinstance(managed_raw, list):
        raise TypeError("topology_complete.json managed_files must be a list")
    managed_files = ensure_sorted_unique_strings(managed_raw, "managed_files")
    missing_required = _TOPOLOGY_REQUIRED_MANAGED - set(managed_files)
    if missing_required:
        raise ValueError(
            "topology_complete.json is missing required files: "
            f"{sorted(missing_required)}"
        )
    for reference in managed_files:
        resolve_safe_file(root, reference, f"managed_files[{reference!r}]")
    actual_files = _actual_file_set(root)
    if actual_files != set(managed_files):
        raise ValueError(
            "Phase 11 managed inventory mismatch; "
            f"missing={sorted(set(managed_files) - actual_files)}, "
            f"unmanaged={sorted(actual_files - set(managed_files))}"
        )
    snapshot = snapshot_managed_files(root, managed_files)
    config = load_topology_config(root / "topology_config.json")
    expected_audit_id = topology_audit_id(
        phase7.source_scientific_generation_id,
        graph.completion_marker["graph_conversion_id"],
        action.completion_marker["action_engine_id"],
        config,
    )
    expected_values = {
        "source_scientific_generation_id": phase7.source_scientific_generation_id,
        "graph_conversion_id": graph.completion_marker["graph_conversion_id"],
        "action_engine_id": action.completion_marker["action_engine_id"],
        "topology_audit_id": expected_audit_id,
        "operational_config_id": topology_operational_config_id(config),
        "topology_schema_id": topology_schema_id(),
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": action.snapshot.aggregate_sha256,
    }
    for name, expected in expected_values.items():
        require_nonblank(marker.get(name), f"topology_complete.json {name}")
        if marker.get(name) != expected:
            raise ValueError(f"topology_complete.json {name} mismatch")

    summary_raw = strict_json_load(root / "topology_summary.json")
    summary = dict(require_mapping(summary_raw, "topology_summary.json"))
    for name, expected in expected_values.items():
        if summary.get(name) != expected:
            raise ValueError(f"topology_summary.json {name} mismatch")
    if summary.get("topology_loss_weight") != 0.0:
        raise ValueError("topology_summary.json topology_loss_weight must be zero")
    if summary.get("topology_mode") != "audit_and_feature_only":
        raise ValueError("topology_summary.json topology_mode mismatch")
    if summary.get("latent_topology_available") is not False:
        raise ValueError("Phase 11 cannot expose latent topology before a model exists")

    reader = ManifestReader(root / "manifests")
    records = reader.read_typed_records(
        "topology_group_manifest",
        TopologyGroupRecordV1,
    )
    if _strict_nonnegative_int(marker, "group_count") != len(records):
        raise ValueError("topology_complete.json group_count mismatch")
    if summary.get("group_count") != len(records):
        raise ValueError("topology_summary.json group_count mismatch")

    groups_by_id: dict[str, Any] = {}
    records_by_id: dict[str, Any] = {}
    point_total = 0
    for record in records:
        record.validate()
        if record.topology_group_id in records_by_id:
            raise ValueError(
                f"Duplicate Phase 11 topology group {record.topology_group_id}"
            )
        group = load_topology_group_artifact(
            resolve_safe_file(
                root,
                record.artifact_ref,
                f"TopologyGroupRecordV1 {record.topology_group_id}.artifact_ref",
            ),
            config,
            record.content_hash,
        )
        groups_by_id[group.topology_group_id] = group
        records_by_id[record.topology_group_id] = record
        point_total += int(group.point_ids.size)
    if _strict_nonnegative_int(marker, "point_count") != point_total:
        raise ValueError("topology_complete.json point_count mismatch")
    if summary.get("total_group_point_count") != point_total:
        raise ValueError("topology_summary.json total_group_point_count mismatch")
    validate_topology_dataset_joins(
        records,
        groups_by_id=groups_by_id,
        config=config,
    )
    return CompletedTopologyDataset(
        root=root,
        config=config,
        completion_marker=marker,
        summary=summary,
        records=list(records),
        groups_by_id=groups_by_id,
        records_by_id=records_by_id,
        managed_files=managed_files,
        snapshot=snapshot,
    )


def load_training_view_sources(
    phase7_root: str | Path,
    graph_root: str | Path,
    action_root: str | Path,
    topology_root: str | Path,
) -> TrainingViewSources:
    """Cross-validate the exact Phase 7/8/9/11 chain consumed by Phase 12."""
    earlier = load_baseline_sources(phase7_root, graph_root, action_root)
    topology = load_completed_topology_dataset(
        topology_root,
        phase7=earlier.phase7,
        graph=earlier.graph,
        action=earlier.action,
    )
    return TrainingViewSources(
        phase7=earlier.phase7,
        graph=earlier.graph,
        action=earlier.action,
        topology=topology,
    )


def verify_training_view_source_snapshots(sources: TrainingViewSources) -> None:
    """Prove no managed source file changed during Phase 12 work."""
    checks = (
        ("Phase 7", sources.phase7.source_root, sources.phase7.source_snapshot),
        ("Phase 8", sources.graph.root, sources.graph.snapshot),
        ("Phase 9", sources.action.root, sources.action.snapshot),
        ("Phase 11", sources.topology.root, sources.topology.snapshot),
    )
    for name, root, expected in checks:
        actual = snapshot_managed_files(
            root,
            tuple(entry.reference for entry in expected.entries),
        )
        if actual != expected:
            raise RuntimeError(f"{name} managed source files changed during Phase 12")


__all__ = [
    "load_completed_topology_dataset",
    "load_training_view_sources",
    "verify_training_view_source_snapshots",
]
