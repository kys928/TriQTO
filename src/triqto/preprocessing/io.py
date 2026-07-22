"""Immutable Phase 7 ingestion and safe artifact loading."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from triqto.data_generation.artifacts import validate_dataset_joins, verify_dataset_references
from triqto.storage import (
    CircuitRecord,
    DistortionRecord,
    ManifestReader,
    MetricRecord,
    SimulationRecord,
)
from triqto.storage.schema import DatasetSampleRecord

from .records import FileInventoryRecord


@dataclass(slots=True)
class Phase7Source:
    root: Path
    completion_marker: dict[str, Any]
    samples: list[DatasetSampleRecord]
    circuits: dict[str, CircuitRecord]
    simulations: dict[str, SimulationRecord]
    distortions: dict[str, DistortionRecord]
    metrics: dict[str, MetricRecord]
    shot_runs_by_source_run: dict[str, list[SimulationRecord]]
    raw_rows: dict[str, dict[str, dict[str, Any]]]
    inventory: list[FileInventoryRecord]


def strict_json_load(path: str | Path) -> Any:
    source = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {source}")

    return json.loads(source.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_path(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("artifact reference must be nonblank text")
    ref = Path(reference)
    if ref.is_absolute() or ref.as_posix() != reference:
        raise ValueError(f"artifact reference must be normalized relative POSIX path: {reference}")
    resolved_root = root.resolve()
    resolved = (root / ref).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"artifact reference escapes dataset root: {reference}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact does not exist: {reference}")
    return resolved


def inventory_files(root: str | Path) -> list[FileInventoryRecord]:
    source = Path(root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"dataset root does not exist: {source}")
    records: list[FileInventoryRecord] = []
    for path in sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(source).as_posix(),
    ):
        stat = path.stat()
        suffix = path.suffix.lower().lstrip(".") or "none"
        records.append(
            FileInventoryRecord(
                relative_path=path.relative_to(source).as_posix(),
                size_bytes=stat.st_size,
                sha256=sha256_file(path),
                format=suffix,
                modified_time_ns=stat.st_mtime_ns,
            )
        )
    return records


def inventory_digest(inventory: Iterable[FileInventoryRecord]) -> str:
    payload = [record.to_dict() for record in inventory]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_inventory_unchanged(
    before: Iterable[FileInventoryRecord], after: Iterable[FileInventoryRecord]
) -> None:
    before_map = {record.relative_path: record for record in before}
    after_map = {record.relative_path: record for record in after}
    if before_map.keys() != after_map.keys():
        missing = sorted(before_map.keys() - after_map.keys())
        added = sorted(after_map.keys() - before_map.keys())
        raise RuntimeError(
            f"raw dataset changed during preprocessing; missing={missing}, added={added}"
        )
    changed = [
        path
        for path in sorted(before_map)
        if (
            before_map[path].size_bytes != after_map[path].size_bytes
            or before_map[path].sha256 != after_map[path].sha256
        )
    ]
    if changed:
        raise RuntimeError(f"raw dataset files changed during preprocessing: {changed}")


def _index_unique(records: Iterable[Any], field_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        value = getattr(record, field_name)
        if value in result:
            raise ValueError(f"duplicate {field_name} {value!r} in Phase 7 source")
        result[value] = record
    return result


def load_phase7_source(root: str | Path) -> Phase7Source:
    source = Path(root).expanduser().resolve()
    marker_path = source / "dataset_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Phase 7 dataset must be complete before preprocessing: {marker_path}"
        )
    marker = strict_json_load(marker_path)
    if not isinstance(marker, Mapping) or marker.get("complete") is not True:
        raise ValueError("dataset_complete.json must declare complete=true")
    inventory = inventory_files(source)
    reader = ManifestReader(source / "manifests")

    def load_rows(name: str, record_type: type[Any], id_field: str) -> tuple[list[Any], dict[str, dict[str, Any]]]:
        rows = reader.read_records(name)
        records = [record_type.from_dict(dict(row)) for row in rows]
        indexed_rows: dict[str, dict[str, Any]] = {}
        for record, row in zip(records, rows):
            record.validate()
            record_id = str(getattr(record, id_field))
            if record_id in indexed_rows:
                raise ValueError(f"duplicate {id_field} {record_id!r} in {name}")
            indexed_rows[record_id] = dict(row)
        return records, indexed_rows

    samples, raw_samples = load_rows("sample_manifest", DatasetSampleRecord, "sample_id")
    circuit_records, raw_circuits = load_rows("circuit_manifest", CircuitRecord, "circuit_id")
    simulation_records, raw_simulations = load_rows("simulation_manifest", SimulationRecord, "run_id")
    distortion_records, raw_distortions = load_rows("distortion_manifest", DistortionRecord, "distortion_id")
    metric_records, raw_metrics = load_rows("metric_manifest", MetricRecord, "metric_id")
    validate_dataset_joins(
        samples,
        circuit_records,
        simulation_records,
        distortion_records,
        metric_records,
    )
    require_statevectors = any(
        record.statevector_ref is not None
        for record in simulation_records
        if record.simulation_mode == "ideal_statevector"
    )
    verify_dataset_references(
        source,
        circuit_records,
        simulation_records,
        require_statevectors=require_statevectors,
    )
    circuits = _index_unique(circuit_records, "circuit_id")
    simulations = _index_unique(simulation_records, "run_id")
    distortions = _index_unique(distortion_records, "distortion_id")
    metrics = _index_unique(metric_records, "metric_id")
    shot_runs_by_source: dict[str, list[SimulationRecord]] = {}
    for record in simulation_records:
        if record.simulation_mode != "ideal_shot":
            continue
        source_run_id = record.metadata.get("source_run_id")
        if isinstance(source_run_id, str) and source_run_id:
            shot_runs_by_source.setdefault(source_run_id, []).append(record)
    for records in shot_runs_by_source.values():
        records.sort(key=lambda item: item.run_id)
    return Phase7Source(
        root=source,
        completion_marker=dict(marker),
        samples=sorted(samples, key=lambda item: item.sample_id),
        circuits=circuits,
        simulations=simulations,
        distortions=distortions,
        metrics=metrics,
        shot_runs_by_source_run=shot_runs_by_source,
        raw_rows={
            "samples": raw_samples,
            "circuits": raw_circuits,
            "simulations": raw_simulations,
            "distortions": raw_distortions,
            "metrics": raw_metrics,
        },
        inventory=inventory,
    )


def load_qpy_circuit(root: Path, record: CircuitRecord) -> Any:
    reference = record.metadata.get("artifact_ref")
    path = safe_artifact_path(root, reference)
    try:
        from qiskit import qpy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Qiskit QPY support is required for preprocessing") from exc
    with path.open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError(
            f"circuit artifact {reference} must contain exactly one circuit, got {len(circuits)}"
        )
    return circuits[0]


def load_probabilities(root: Path, record: SimulationRecord) -> dict[str, float]:
    if record.probabilities_ref is None:
        raise ValueError(f"simulation {record.run_id} has no probability reference")
    payload = strict_json_load(safe_artifact_path(root, record.probabilities_ref))
    if not isinstance(payload, Mapping):
        raise TypeError(f"probability artifact for {record.run_id} must be a mapping")
    return {str(key): float(value) for key, value in payload.items()}


def load_counts(root: Path, record: SimulationRecord) -> dict[str, int]:
    if record.counts_ref is None:
        raise ValueError(f"simulation {record.run_id} has no count reference")
    payload = strict_json_load(safe_artifact_path(root, record.counts_ref))
    if not isinstance(payload, Mapping):
        raise TypeError(f"count artifact for {record.run_id} must be a mapping")
    return {str(key): int(value) for key, value in payload.items()}


def load_statevector(root: Path, record: SimulationRecord) -> np.ndarray | None:
    if record.statevector_ref is None:
        return None
    path = safe_artifact_path(root, record.statevector_ref)
    array = np.load(path, allow_pickle=False)
    return np.asarray(array, dtype=np.complex128)
