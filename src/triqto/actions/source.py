"""Read-only cross-validation of completed Phase 7 and Phase 8 action sources."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from triqto.graph import (
    graph_conversion_id,
    graph_operational_config_id,
    graph_schema_id,
    load_completed_phase7_dataset,
    load_graph_artifact,
    load_graph_config,
    load_pair_artifact,
    snapshot_managed_files,
    validate_graph_dataset_joins,
)
from triqto.graph.utils import (
    ensure_sorted_unique_strings,
    require_mapping,
    require_nonblank,
    resolve_safe_file,
    strict_json_load,
)
from triqto.storage import GraphPairRecord, GraphRecord, ManifestReader

from .models import ActionEngineSources, CompletedGraphDataset

_GRAPH_MARKER_KEYS = {
    "complete",
    "source_scientific_generation_id",
    "graph_conversion_id",
    "operational_config_id",
    "graph_schema_id",
    "graph_count",
    "pair_count",
    "source_snapshot_hash",
    "managed_files",
}
_GRAPH_REQUIRED_MANAGED = {
    "graph_config.json",
    "graph_summary.json",
    "graph_complete.json",
    "manifests/graph_manifest.parquet",
    "manifests/graph_pair_manifest.parquet",
}


def _strict_marker_int(marker: Mapping[str, Any], name: str) -> int:
    value = marker.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"graph_complete.json {name} must be an integer and not bool")
    if value < 0:
        raise ValueError(f"graph_complete.json {name} must be nonnegative")
    return value


def _actual_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def load_completed_graph_dataset(
    graph_root: str | Path,
    *,
    source_samples: list[Any] | None = None,
) -> CompletedGraphDataset:
    """Load and fully validate one immutable Phase 8 graph dataset."""
    root = Path(graph_root)
    if not root.exists():
        raise FileNotFoundError(f"Phase 8 graph root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Phase 8 graph root is not a directory: {root}")

    marker_path = root / "graph_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Phase 8 completion marker missing: {marker_path}")
    marker_raw = strict_json_load(marker_path)
    marker = dict(require_mapping(marker_raw, "graph_complete.json"))
    if set(marker) != _GRAPH_MARKER_KEYS:
        raise ValueError(
            "graph_complete.json key mismatch; "
            f"missing={sorted(_GRAPH_MARKER_KEYS - set(marker))}, "
            f"unexpected={sorted(set(marker) - _GRAPH_MARKER_KEYS)}"
        )
    if marker.get("complete") is not True:
        raise ValueError("graph_complete.json complete must be exactly true")
    managed_raw = marker.get("managed_files")
    if not isinstance(managed_raw, list):
        raise TypeError("graph_complete.json managed_files must be a list")
    managed_files = ensure_sorted_unique_strings(managed_raw, "managed_files")
    missing_required = _GRAPH_REQUIRED_MANAGED - set(managed_files)
    if missing_required:
        raise ValueError(
            "graph_complete.json is missing required files: "
            f"{sorted(missing_required)}"
        )
    for reference in managed_files:
        resolve_safe_file(root, reference, f"managed_files[{reference!r}]")
    actual_files = _actual_file_set(root)
    if actual_files != set(managed_files):
        raise ValueError(
            "Phase 8 managed file inventory mismatch; "
            f"missing={sorted(set(managed_files) - actual_files)}, "
            f"unmanaged={sorted(actual_files - set(managed_files))}"
        )

    snapshot = snapshot_managed_files(root, managed_files)
    config = load_graph_config(root / "graph_config.json")
    if marker.get("operational_config_id") != graph_operational_config_id(config):
        raise ValueError("graph_complete.json operational_config_id mismatch")
    if marker.get("graph_schema_id") != graph_schema_id():
        raise ValueError("graph_complete.json graph_schema_id mismatch")
    expected_conversion_id = graph_conversion_id(
        marker.get("source_scientific_generation_id")
    )
    if marker.get("graph_conversion_id") != expected_conversion_id:
        raise ValueError("graph_complete.json graph_conversion_id mismatch")
    for name in (
        "source_scientific_generation_id",
        "graph_conversion_id",
        "source_snapshot_hash",
    ):
        require_nonblank(marker.get(name), f"graph_complete.json {name}")

    summary_raw = strict_json_load(root / "graph_summary.json")
    summary = dict(require_mapping(summary_raw, "graph_summary.json"))
    for name in (
        "source_scientific_generation_id",
        "graph_conversion_id",
        "operational_config_id",
        "graph_schema_id",
        "source_snapshot_hash",
    ):
        if summary.get(name) != marker.get(name):
            raise ValueError(f"graph_summary.json {name} does not match completion marker")

    reader = ManifestReader(root / "manifests")
    graph_records = reader.read_typed_records("graph_manifest", GraphRecord)
    pair_records = reader.read_typed_records("graph_pair_manifest", GraphPairRecord)
    for record in [*graph_records, *pair_records]:
        record.validate()
    if _strict_marker_int(marker, "graph_count") != len(graph_records):
        raise ValueError("graph_complete.json graph_count mismatch")
    if _strict_marker_int(marker, "pair_count") != len(pair_records):
        raise ValueError("graph_complete.json pair_count mismatch")
    if summary.get("graph_count") != len(graph_records):
        raise ValueError("graph_summary.json graph_count mismatch")
    if summary.get("pair_count") != len(pair_records):
        raise ValueError("graph_summary.json pair_count mismatch")

    graphs_by_id: dict[str, Any] = {}
    for record in graph_records:
        if record.graph_id in graphs_by_id:
            raise ValueError(f"Duplicate GraphRecord graph_id {record.graph_id}")
        path = resolve_safe_file(
            root,
            record.graph_ref,
            f"GraphRecord {record.graph_id}.graph_ref",
        )
        graphs_by_id[record.graph_id] = load_graph_artifact(
            path,
            record.content_hash,
        )

    pairs_by_id: dict[str, Any] = {}
    pair_records_by_sample_id: dict[str, GraphPairRecord] = {}
    for record in pair_records:
        if record.graph_pair_id in pairs_by_id:
            raise ValueError(
                f"Duplicate GraphPairRecord graph_pair_id {record.graph_pair_id}"
            )
        if record.sample_id in pair_records_by_sample_id:
            raise ValueError(
                f"Multiple GraphPairRecords reference sample {record.sample_id}"
            )
        path = resolve_safe_file(
            root,
            record.pair_ref,
            f"GraphPairRecord {record.graph_pair_id}.pair_ref",
        )
        pairs_by_id[record.graph_pair_id] = load_pair_artifact(
            path,
            record.content_hash,
        )
        pair_records_by_sample_id[record.sample_id] = record

    validate_graph_dataset_joins(
        graph_records,
        pair_records,
        source_samples=source_samples,
        graphs_by_id=graphs_by_id,
        pairs_by_id=pairs_by_id,
        root=root,
    )
    if source_samples is not None:
        source_ids = {sample.sample_id for sample in source_samples}
        if set(pair_records_by_sample_id) != source_ids:
            raise ValueError(
                "Phase 8 graph pairs do not cover the Phase 7 sample set exactly"
            )
        if summary.get("source_sample_count") != len(source_ids):
            raise ValueError("graph_summary.json source_sample_count mismatch")

    return CompletedGraphDataset(
        root=root,
        config=config,
        completion_marker=marker,
        summary=summary,
        graph_records=list(graph_records),
        pair_records=list(pair_records),
        graphs_by_id=graphs_by_id,
        pairs_by_id=pairs_by_id,
        pair_records_by_sample_id=pair_records_by_sample_id,
        managed_files=managed_files,
        snapshot=snapshot,
    )


def load_action_engine_sources(
    phase7_root: str | Path,
    graph_root: str | Path,
) -> ActionEngineSources:
    """Cross-validate completed Phase 7 and Phase 8 roots for Phase 9."""
    phase7 = load_completed_phase7_dataset(phase7_root)
    graph = load_completed_graph_dataset(
        graph_root,
        source_samples=phase7.samples,
    )
    marker = graph.completion_marker
    if (
        marker["source_scientific_generation_id"]
        != phase7.source_scientific_generation_id
    ):
        raise ValueError(
            "Phase 8 source scientific generation ID does not match Phase 7"
        )
    if marker["source_snapshot_hash"] != phase7.source_snapshot.aggregate_sha256:
        raise ValueError("Phase 8 source snapshot hash does not match Phase 7")
    return ActionEngineSources(phase7=phase7, graph=graph)


def verify_action_source_snapshots(sources: ActionEngineSources) -> None:
    """Prove neither managed source root changed during Phase 9 work."""
    phase7_actual = snapshot_managed_files(
        sources.phase7.source_root,
        tuple(entry.reference for entry in sources.phase7.source_snapshot.entries),
    )
    if phase7_actual != sources.phase7.source_snapshot:
        raise RuntimeError("Phase 7 managed source files changed during Phase 9")
    graph_actual = snapshot_managed_files(
        sources.graph.root,
        tuple(entry.reference for entry in sources.graph.snapshot.entries),
    )
    if graph_actual != sources.graph.snapshot:
        raise RuntimeError("Phase 8 managed source files changed during Phase 9")


__all__ = [
    "load_action_engine_sources",
    "load_completed_graph_dataset",
    "verify_action_source_snapshots",
]
