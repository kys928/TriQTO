"""Read-only validation and loading of completed Phase 7 datasets."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import numbers
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qpy

from triqto.data_generation import (
    config_from_dict,
    config_id,
    scientific_generation_id,
    validate_dataset_joins,
    verify_dataset_references,
)
from triqto.storage.manifest import ManifestReader
from triqto.storage.schema import (
    CircuitRecord,
    DatasetSampleRecord,
    DistortionRecord,
    MetricRecord,
    MeasurementSettingRecord,
    SimulationRecord,
)

from .evidence import validate_count_mapping, validate_probability_mapping
from .models import CompletedPhase7Dataset, SourceFileEntry, SourceFileSnapshot
from .utils import (
    ensure_sorted_unique_strings,
    require_mapping,
    require_nonblank,
    resolve_safe_file,
    strict_json_load,
)

_PHASE7_MANIFEST_NAMES = (
    "sample_manifest",
    "circuit_manifest",
    "simulation_manifest",
    "measurement_setting_manifest",
    "distortion_manifest",
    "metric_manifest",
)
_REQUIRED_MANAGED_FILES = {
    "generation_config.json",
    "dataset_summary.json",
    "dataset_complete.json",
    *{f"manifests/{name}.parquet" for name in _PHASE7_MANIFEST_NAMES},
}


def snapshot_managed_files(
    root: str | Path,
    managed_files: tuple[str, ...],
) -> SourceFileSnapshot:
    base = Path(root)
    entries: list[SourceFileEntry] = []
    aggregate = hashlib.sha256()
    for reference in managed_files:
        path = resolve_safe_file(base, reference, f"managed_files[{reference!r}]")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        entry = SourceFileEntry(reference, size, digest)
        entries.append(entry)
        aggregate.update(reference.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return SourceFileSnapshot(tuple(entries), f"sha256:{aggregate.hexdigest()}")


def verify_source_snapshot(
    root: str | Path,
    expected: SourceFileSnapshot,
) -> None:
    actual = snapshot_managed_files(
        root,
        tuple(entry.reference for entry in expected.entries),
    )
    if actual != expected:
        raise RuntimeError("Phase 7 source managed files changed during Phase 8")


def _typed_manifests(
    root: Path,
) -> tuple[
    list[DatasetSampleRecord],
    list[CircuitRecord],
    list[SimulationRecord],
    list[MeasurementSettingRecord],
    list[DistortionRecord],
    list[MetricRecord],
]:
    reader = ManifestReader(root / "manifests")
    return (
        reader.read_typed_records("sample_manifest", DatasetSampleRecord),
        reader.read_typed_records("circuit_manifest", CircuitRecord),
        reader.read_typed_records("simulation_manifest", SimulationRecord),
        reader.read_typed_records("measurement_setting_manifest", MeasurementSettingRecord),
        reader.read_typed_records("distortion_manifest", DistortionRecord),
        reader.read_typed_records("metric_manifest", MetricRecord),
    )


def _load_one_qpy(path: Path, circuit_id: str) -> QuantumCircuit:
    with path.open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError(
            f"CircuitRecord {circuit_id} QPY artifact must contain exactly one circuit"
        )
    circuit = circuits[0]
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"CircuitRecord {circuit_id} QPY payload is not QuantumCircuit")
    return circuit


def _require_marker_int(marker: Mapping[str, Any], name: str) -> int:
    value = marker.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"dataset_complete.json {name} must be an integer and not bool")
    return value


def load_completed_phase7_dataset(
    source_root: str | Path,
) -> CompletedPhase7Dataset:
    """Validate and load a completed Phase 7 dataset without opening statevectors."""
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"Phase 7 source root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Phase 7 source root is not a directory: {root}")

    marker_path = root / "dataset_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Phase 7 completion marker missing: {marker_path}")
    marker_raw = strict_json_load(marker_path)
    marker = dict(require_mapping(marker_raw, "dataset_complete.json"))
    if marker.get("complete") is not True:
        raise ValueError("dataset_complete.json complete must be exactly true")
    if "managed_files" not in marker:
        raise ValueError("dataset_complete.json managed_files is required")
    managed_raw = marker["managed_files"]
    if not isinstance(managed_raw, list):
        raise TypeError("dataset_complete.json managed_files must be a list")
    managed_files = ensure_sorted_unique_strings(managed_raw, "managed_files")
    for reference in managed_files:
        resolve_safe_file(root, reference, f"managed_files[{reference!r}]")
    missing_required = _REQUIRED_MANAGED_FILES - set(managed_files)
    if missing_required:
        raise ValueError(
            "dataset_complete.json managed_files is missing required Phase 7 files: "
            f"{sorted(missing_required)}"
        )

    source_snapshot = snapshot_managed_files(root, managed_files)

    config_payload_raw = strict_json_load(root / "generation_config.json")
    config_payload = dict(require_mapping(config_payload_raw, "generation_config.json"))
    generation_config = config_from_dict(config_payload)
    expected_scientific_id = scientific_generation_id(generation_config)
    expected_config_id = config_id(generation_config)
    if marker.get("scientific_generation_id") != expected_scientific_id:
        raise ValueError("dataset_complete.json scientific_generation_id mismatch")
    if marker.get("config_id") != expected_config_id:
        raise ValueError("dataset_complete.json config_id mismatch")
    if marker.get("dataset_name") != generation_config.dataset_name:
        raise ValueError("dataset_complete.json dataset_name mismatch")
    if marker.get("schema_version") != generation_config.schema_version:
        raise ValueError("dataset_complete.json schema_version mismatch")
    if _require_marker_int(marker, "manifest_count") != len(_PHASE7_MANIFEST_NAMES):
        raise ValueError("dataset_complete.json manifest_count mismatch")

    summary = strict_json_load(root / "dataset_summary.json")
    require_mapping(summary, "dataset_summary.json")

    samples, circuits, simulations, measurement_settings, distortions, metrics = _typed_manifests(root)
    if _require_marker_int(marker, "sample_count") != len(samples):
        raise ValueError("dataset_complete.json sample_count mismatch")
    validate_dataset_joins(
        samples,
        circuits,
        simulations,
        distortions,
        metrics,
        measurement_settings,
    )
    verify_dataset_references(
        root,
        circuits,
        simulations,
        require_statevectors=generation_config.store_statevectors,
    )

    circuit_records_by_id = {record.circuit_id: record for record in circuits}
    exact_runs = {
        record.run_id: record
        for record in simulations
        if record.simulation_mode == "ideal_statevector"
    }
    if not exact_runs:
        raise ValueError("Phase 7 dataset has no ideal_statevector simulation records")
    probability_runs = {
        record.run_id: record
        for record in simulations
        if record.simulation_mode in {
            "ideal_statevector",
            "ideal_measurement_probabilities",
        }
    }

    required_artifact_refs: set[str] = set()
    circuits_by_id: dict[str, QuantumCircuit] = {}
    for record in circuits:
        metadata = require_mapping(record.metadata, f"CircuitRecord {record.circuit_id}.metadata")
        reference = metadata.get("artifact_ref")
        if not isinstance(reference, str):
            raise ValueError(
                f"CircuitRecord {record.circuit_id} metadata.artifact_ref is required"
            )
        required_artifact_refs.add(reference)
        path = resolve_safe_file(
            root,
            reference,
            f"CircuitRecord {record.circuit_id}.metadata.artifact_ref",
        )
        circuits_by_id[record.circuit_id] = _load_one_qpy(path, record.circuit_id)

    probabilities_by_run_id: dict[str, dict[str, float]] = {}
    for run_id, record in probability_runs.items():
        if record.probabilities_ref is None:
            raise ValueError(f"SimulationRecord {run_id} probabilities_ref is required")
        required_artifact_refs.add(record.probabilities_ref)
        path = resolve_safe_file(
            root,
            record.probabilities_ref,
            f"SimulationRecord {run_id}.probabilities_ref",
        )
        payload_raw = strict_json_load(path)
        payload = require_mapping(payload_raw, f"probability artifact for run {run_id}")
        circuit_record = circuit_records_by_id.get(record.circuit_id)
        if circuit_record is None:
            raise ValueError(
                f"SimulationRecord {run_id} references missing CircuitRecord {record.circuit_id}"
            )
        validate_probability_mapping(payload, circuit_record.n_qubits)
        probabilities_by_run_id[run_id] = dict(payload)
        if record.statevector_ref is not None:
            required_artifact_refs.add(record.statevector_ref)

    shot_records_by_exact_run_id: dict[str, SimulationRecord] = {}
    counts_by_exact_run_id: dict[str, dict[str, int]] = {}
    for record in simulations:
        if record.simulation_mode != "ideal_shot":
            continue
        metadata = require_mapping(record.metadata, f"SimulationRecord {record.run_id}.metadata")
        source_run_id = require_nonblank(
            metadata.get("source_run_id"),
            f"SimulationRecord {record.run_id}.metadata.source_run_id",
        )
        if source_run_id not in probability_runs:
            raise ValueError(
                f"SimulationRecord {record.run_id} source_run_id references missing "
                f"probability run {source_run_id}"
            )
        if source_run_id in shot_records_by_exact_run_id:
            raise ValueError(
                f"Multiple ideal_shot records reference exact run {source_run_id}; "
                "Phase 8 v1 rejects ambiguous supplemental count sources"
            )
        if metadata.get("sampling_source") != "sampled_from_exact_born_probabilities":
            raise ValueError(
                f"SimulationRecord {record.run_id} metadata.sampling_source is invalid"
            )
        if record.counts_ref is None:
            raise ValueError(f"SimulationRecord {record.run_id} counts_ref is required")
        raw_shots = record.shots
        if isinstance(raw_shots, bool) or not isinstance(raw_shots, numbers.Real):
            raise TypeError(
                f"SimulationRecord {record.run_id} shots must be a positive integer"
            )
        numeric_shots = float(raw_shots)
        if (
            not math.isfinite(numeric_shots)
            or not numeric_shots.is_integer()
            or numeric_shots <= 0
        ):
            raise ValueError(
                f"SimulationRecord {record.run_id} shots must be a positive integer"
            )
        shots = int(numeric_shots)
        record.shots = shots
        required_artifact_refs.add(record.counts_ref)
        path = resolve_safe_file(
            root,
            record.counts_ref,
            f"SimulationRecord {record.run_id}.counts_ref",
        )
        payload_raw = strict_json_load(path)
        payload = require_mapping(payload_raw, f"count artifact for run {record.run_id}")
        exact_record = probability_runs[source_run_id]
        circuit_record = circuit_records_by_id[exact_record.circuit_id]
        validate_count_mapping(payload, circuit_record.n_qubits, shots)
        shot_records_by_exact_run_id[source_run_id] = record
        counts_by_exact_run_id[source_run_id] = dict(payload)

    unmanaged_required = required_artifact_refs - set(managed_files)
    if unmanaged_required:
        raise ValueError(
            "Phase 7 completion marker does not manage referenced artifacts: "
            f"{sorted(unmanaged_required)}"
        )

    verify_source_snapshot(root, source_snapshot)
    return CompletedPhase7Dataset(
        source_root=root,
        generation_config=generation_config,
        generation_config_payload=config_payload,
        source_scientific_generation_id=expected_scientific_id,
        source_config_id=expected_config_id,
        samples=samples,
        circuits=circuits,
        simulations=simulations,
        distortions=distortions,
        metrics=metrics,
        measurement_settings=measurement_settings,
        circuits_by_id=circuits_by_id,
        probabilities_by_run_id=probabilities_by_run_id,
        counts_by_exact_run_id=counts_by_exact_run_id,
        shot_records_by_exact_run_id=shot_records_by_exact_run_id,
        statevector_storage_enabled=generation_config.store_statevectors,
        completion_marker=marker,
        managed_files=managed_files,
        source_snapshot=source_snapshot,
    )


__all__ = [
    "load_completed_phase7_dataset",
    "snapshot_managed_files",
    "verify_source_snapshot",
]
