"""Artifact writer and integrity validators for Phase 7 generated datasets."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from triqto.storage import (
    CircuitRecord,
    DistortionRecord,
    ManifestReader,
    ManifestWriter,
    MetricRecord,
    SimulationRecord,
)
from triqto.storage.schema import DatasetSampleRecord

from .records import DatasetGenerationResult, DatasetWriteResult
from .specs import config_to_dict

MANIFEST_NAMES = {
    "sample_manifest": "sample_manifest",
    "circuit_manifest": "circuit_manifest",
    "simulation_manifest": "simulation_manifest",
    "distortion_manifest": "distortion_manifest",
    "metric_manifest": "metric_manifest",
}
DATASET_COMPLETE_NAME = "dataset_complete.json"
EMPTY_MAP_ENCODING = "parquet_null_normalized_to_empty_dict"


def _load_qpy_module() -> Any:
    try:
        from qiskit import qpy  # type: ignore
    except Exception as exc:  # pragma: no cover - environment/API dependent
        raise RuntimeError("Qiskit QPY support is required to persist circuit artifacts.") from exc
    if not hasattr(qpy, "dump") or not hasattr(qpy, "load"):
        raise RuntimeError("Qiskit QPY support is required to persist circuit artifacts.")
    return qpy


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return path


def _relative_ref(relative_path: str) -> str:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"Artifact reference must be relative: {relative_path}")
    return candidate.as_posix()


def _records_with_artifact_refs(
    result: DatasetGenerationResult,
) -> tuple[list[CircuitRecord], list[SimulationRecord]]:
    circuit_records: list[CircuitRecord] = []
    for record in result.circuit_records:
        metadata = dict(record.metadata)
        metadata["artifact_ref"] = _relative_ref(f"artifacts/circuits/{record.circuit_id}.qpy")
        circuit_records.append(replace(record, metadata=metadata))

    simulation_records: list[SimulationRecord] = []
    for record in result.simulation_records:
        metadata = dict(record.metadata)
        if record.simulation_mode == "ideal_statevector":
            simulation_records.append(
                replace(
                    record,
                    statevector_ref=metadata.pop("statevector_ref", None),
                    probabilities_ref=metadata.pop("probabilities_ref", None),
                    metadata=metadata,
                )
            )
        elif record.simulation_mode == "ideal_shot":
            simulation_records.append(
                replace(
                    record,
                    counts_ref=metadata.pop("counts_ref", None),
                    metadata=metadata,
                )
            )
        else:
            simulation_records.append(record)
    return circuit_records, simulation_records


def _unique_sorted(paths: list[Path]) -> list[Path]:
    return sorted(set(paths), key=lambda path: path.as_posix())


def _planned_paths(result: DatasetGenerationResult, root: Path) -> dict[str, list[Path]]:
    circuits = [
        root / "artifacts" / "circuits" / f"{record.circuit_id}.qpy"
        for record in result.circuit_records
    ]
    probabilities: list[Path] = []
    statevectors: list[Path] = []
    for sample in result.samples:
        probabilities.extend(
            [
                root / "artifacts" / "probabilities" / f"{sample.clean_run_id}.json",
                root / "artifacts" / "probabilities" / f"{sample.distorted_run_id}.json",
            ]
        )
        if result.config.store_statevectors:
            statevectors.extend(
                [
                    root / "artifacts" / "statevectors" / f"{sample.clean_run_id}.npy",
                    root / "artifacts" / "statevectors" / f"{sample.distorted_run_id}.npy",
                ]
            )
    counts = [
        root / "artifacts" / "counts" / f"{record.run_id}.json"
        for record in result.simulation_records
        if record.simulation_mode == "ideal_shot"
    ]
    return {
        "circuits": _unique_sorted(circuits),
        "probabilities": _unique_sorted(probabilities),
        "statevectors": _unique_sorted(statevectors),
        "counts": _unique_sorted(counts),
    }


def _safe_reference_path(root: Path, record_id: str, field_name: str, reference: Any) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(
            f"Record {record_id} field {field_name} has empty or non-string artifact reference: {reference!r}"
        )
    if Path(reference).as_posix() != reference:
        raise ValueError(
            f"Record {record_id} field {field_name} is not a normalized POSIX path: {reference}"
        )
    ref_path = Path(reference)
    if ref_path.is_absolute():
        raise ValueError(
            f"Record {record_id} field {field_name} has absolute artifact reference: {reference}"
        )
    resolved_root = root.resolve()
    resolved_target = (root / ref_path).resolve()
    if resolved_target == resolved_root:
        raise ValueError(
            f"Record {record_id} field {field_name} references dataset root: {reference}"
        )
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Record {record_id} field {field_name} escapes dataset root: {reference}"
        ) from exc
    return resolved_target


def _safe_managed_relative_path(root: Path, reference: str) -> Path:
    path = _safe_reference_path(root, "dataset_complete", "managed_files", reference)
    return path.relative_to(root.resolve())


def _require_file(root: Path, record_id: str, field_name: str, reference: Any) -> Path:
    path = _safe_reference_path(root, record_id, field_name, reference)
    if not path.exists():
        raise FileNotFoundError(
            f"Record {record_id} field {field_name} references missing artifact: {reference}"
        )
    if not path.is_file():
        raise ValueError(
            f"Record {record_id} field {field_name} does not reference a file: {reference}"
        )
    return path


def _require_absent(record_id: str, field_name: str, reference: Any) -> None:
    if reference is not None:
        raise ValueError(
            f"Record {record_id} field {field_name} must be absent for this simulation mode: {reference}"
        )


def _require_mapping_field(record: Any, field_name: str) -> Mapping[str, Any]:
    value = getattr(record, field_name, None)
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{type(record).__name__}.{field_name} must be a mapping, got {type(value).__name__}"
        )
    return value


def _require_nonblank_string_field(record: Any, field_name: str) -> str:
    value = getattr(record, field_name, None)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{type(record).__name__}.{field_name} must be a nonblank string")
    return value


def _validate_record_shape(record: Any) -> None:
    for field_name in getattr(record, "required_fields", ()):
        _require_nonblank_string_field(record, field_name)

    if hasattr(record, "metadata"):
        _require_mapping_field(record, "metadata")

    if isinstance(record, DatasetSampleRecord):
        _require_nonblank_string_field(record, "schema_version")
        _require_nonblank_string_field(record, "family")
        _require_mapping_field(record, "parameter_bindings")
    elif isinstance(record, CircuitRecord):
        _require_nonblank_string_field(record, "family")
    elif isinstance(record, SimulationRecord):
        _require_nonblank_string_field(record, "simulation_mode")
    elif isinstance(record, DistortionRecord):
        _require_nonblank_string_field(record, "distortion_type")


def verify_dataset_references(
    root: str | Path,
    circuit_records: list[Any],
    simulation_records: list[Any],
    *,
    require_statevectors: bool,
) -> None:
    """Verify Phase 7 manifest references by record type and simulation mode."""
    root_path = Path(root)
    for record in circuit_records:
        if not isinstance(record, CircuitRecord):
            raise TypeError(f"Expected CircuitRecord, got {type(record).__name__}")
        metadata = _require_mapping_field(record, "metadata")
        _require_file(
            root_path,
            record.circuit_id,
            "metadata.artifact_ref",
            metadata.get("artifact_ref"),
        )

    for record in simulation_records:
        if not isinstance(record, SimulationRecord):
            raise TypeError(f"Expected SimulationRecord, got {type(record).__name__}")
        metadata = _require_mapping_field(record, "metadata")
        if record.simulation_mode == "ideal_statevector":
            _require_file(root_path, record.run_id, "probabilities_ref", record.probabilities_ref)
            _require_absent(record.run_id, "counts_ref", record.counts_ref)
            if require_statevectors:
                _require_file(root_path, record.run_id, "statevector_ref", record.statevector_ref)
            else:
                _require_absent(record.run_id, "statevector_ref", record.statevector_ref)
        elif record.simulation_mode == "ideal_shot":
            _require_file(root_path, record.run_id, "counts_ref", record.counts_ref)
            _require_absent(record.run_id, "probabilities_ref", record.probabilities_ref)
            _require_absent(record.run_id, "statevector_ref", record.statevector_ref)
            if not metadata.get("source_run_id"):
                raise ValueError(
                    f"Record {record.run_id} field metadata.source_run_id is required for ideal_shot"
                )
            if metadata.get("sampling_source") != "sampled_from_exact_born_probabilities":
                raise ValueError(
                    f"Record {record.run_id} field metadata.sampling_source is invalid for ideal_shot"
                )
        else:
            raise ValueError(
                f"Record {record.run_id} has unsupported simulation_mode: {record.simulation_mode}"
            )


def _record_id(record: Any, id_field: str) -> str:
    value = getattr(record, id_field, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{type(record).__name__}.{id_field} is required for manifest integrity validation"
        )
    return value


def _index_unique(records: list[Any], id_field: str, record_type: type) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, record_type):
            raise TypeError(
                f"Expected {record_type.__name__} in {record_type.__name__} manifest, got {type(record).__name__}"
            )
        try:
            _validate_record_shape(record)
            record.validate()
        except Exception as exc:
            record_id = getattr(record, id_field, "<missing>")
            raise ValueError(
                f"Invalid {record_type.__name__} {id_field}={record_id}: {exc}"
            ) from exc
        record_id = _record_id(record, id_field)
        if record_id in indexed:
            raise ValueError(
                f"Duplicate {record_type.__name__} {id_field} {record_id} in manifest"
            )
        indexed[record_id] = record
    return indexed


def _lookup(
    index: dict[str, Any],
    sample_id: str,
    field_name: str,
    record_type: str,
    record_id: str,
) -> Any:
    try:
        return index[record_id]
    except KeyError as exc:
        raise ValueError(
            f"Sample {sample_id} {field_name} references missing {record_type} {record_id}"
        ) from exc


def _validate_phase7_metric_record(record: MetricRecord) -> None:
    if not record.born_metrics:
        raise ValueError(
            f"MetricRecord {record.metric_id} born_metrics must be nonempty for Phase 7"
        )
    if (
        record.hilbert_metrics != {}
        or record.parameter_metrics != {}
        or record.topology_metrics != {}
    ):
        raise ValueError(
            f"MetricRecord {record.metric_id} deferred metric maps must be empty for Phase 7"
        )
    if record.hilbert_available_mask is not False:
        raise ValueError(
            f"MetricRecord {record.metric_id} hilbert_available_mask must be False for Phase 7"
        )
    if record.metadata.get("metric_family") != "born":
        raise ValueError(
            f"MetricRecord {record.metric_id} metadata.metric_family must be 'born'"
        )
    support_size = record.metadata.get("support_size")
    if (
        not isinstance(support_size, int)
        or isinstance(support_size, bool)
        or support_size < 0
    ):
        raise ValueError(
            f"MetricRecord {record.metric_id} metadata.support_size must be a nonnegative integer"
        )


def validate_dataset_joins(
    sample_records: list[Any],
    circuit_records: list[Any],
    simulation_records: list[Any],
    distortion_records: list[Any],
    metric_records: list[Any],
) -> None:
    """Validate uniqueness and semantic joins among Phase 7 manifest records."""
    samples = _index_unique(sample_records, "sample_id", DatasetSampleRecord)
    circuits = _index_unique(circuit_records, "circuit_id", CircuitRecord)
    simulations = _index_unique(simulation_records, "run_id", SimulationRecord)
    distortions = _index_unique(distortion_records, "distortion_id", DistortionRecord)
    metrics = _index_unique(metric_records, "metric_id", MetricRecord)

    for metric in metrics.values():
        _validate_phase7_metric_record(metric)

    for sample in samples.values():
        clean_circuit = _lookup(
            circuits,
            sample.sample_id,
            "clean_circuit_id",
            "CircuitRecord",
            sample.clean_circuit_id,
        )
        distorted_circuit = _lookup(
            circuits,
            sample.sample_id,
            "distorted_circuit_id",
            "CircuitRecord",
            sample.distorted_circuit_id,
        )
        if clean_circuit.metadata.get("role") != "clean":
            raise ValueError(
                f"Sample {sample.sample_id} clean_circuit_id does not point to a clean CircuitRecord"
            )
        if distorted_circuit.metadata.get("role") != "distorted":
            raise ValueError(
                f"Sample {sample.sample_id} distorted_circuit_id does not point to a distorted CircuitRecord"
            )
        if clean_circuit.family != sample.family or distorted_circuit.family != sample.family:
            raise ValueError(
                f"Sample {sample.sample_id} family does not match joined CircuitRecords"
            )
        if clean_circuit.n_qubits != sample.n_qubits or distorted_circuit.n_qubits != sample.n_qubits:
            raise ValueError(
                f"Sample {sample.sample_id} n_qubits does not match joined CircuitRecords"
            )
        if clean_circuit.metadata.get("parameter_bindings") != sample.parameter_bindings:
            raise ValueError(
                f"Sample {sample.sample_id} clean circuit parameter_bindings mismatch"
            )
        if distorted_circuit.metadata.get("parameter_bindings") != sample.parameter_bindings:
            raise ValueError(
                f"Sample {sample.sample_id} distorted circuit parameter_bindings mismatch"
            )
        if distorted_circuit.metadata.get("source_clean_circuit_id") != sample.clean_circuit_id:
            raise ValueError(
                f"Sample {sample.sample_id} distorted circuit source_clean_circuit_id mismatch"
            )

        clean_run = _lookup(
            simulations,
            sample.sample_id,
            "clean_run_id",
            "SimulationRecord",
            sample.clean_run_id,
        )
        distorted_run = _lookup(
            simulations,
            sample.sample_id,
            "distorted_run_id",
            "SimulationRecord",
            sample.distorted_run_id,
        )
        if clean_run.simulation_mode != "ideal_statevector" or clean_run.circuit_id != sample.clean_circuit_id:
            raise ValueError(
                f"Sample {sample.sample_id} clean_run_id does not point to clean ideal_statevector run"
            )
        if distorted_run.simulation_mode != "ideal_statevector" or distorted_run.circuit_id != sample.distorted_circuit_id:
            raise ValueError(
                f"Sample {sample.sample_id} distorted_run_id does not point to distorted ideal_statevector run"
            )
        if (
            clean_run.backend_name != "qiskit.quantum_info.Statevector"
            or clean_run.metadata.get("sampling_source") != "exact_statevector"
        ):
            raise ValueError(
                f"Sample {sample.sample_id} clean_run_id has invalid backend/source metadata"
            )
        if (
            distorted_run.backend_name != "qiskit.quantum_info.Statevector"
            or distorted_run.metadata.get("sampling_source") != "exact_statevector"
        ):
            raise ValueError(
                f"Sample {sample.sample_id} distorted_run_id has invalid backend/source metadata"
            )

        distortion = _lookup(
            distortions,
            sample.sample_id,
            "distortion_id",
            "DistortionRecord",
            sample.distortion_id,
        )
        if distortion.circuit_id != sample.clean_circuit_id:
            raise ValueError(f"Sample {sample.sample_id} distortion circuit_id mismatch")
        if distortion.metadata.get("distorted_circuit_id") != sample.distorted_circuit_id:
            raise ValueError(
                f"Sample {sample.sample_id} distortion distorted_circuit_id mismatch"
            )

        metric = _lookup(
            metrics,
            sample.sample_id,
            "metric_id",
            "MetricRecord",
            sample.metric_id,
        )
        if (
            metric.run_id != sample.distorted_run_id
            or metric.circuit_id != sample.distorted_circuit_id
            or metric.distortion_id != sample.distortion_id
        ):
            raise ValueError(
                f"Sample {sample.sample_id} metric record IDs do not match sample joins"
            )
        if (
            metric.metadata.get("clean_run_id") != sample.clean_run_id
            or metric.metadata.get("distorted_run_id") != sample.distorted_run_id
        ):
            raise ValueError(f"Sample {sample.sample_id} metric run metadata mismatch")
        if metric.metadata.get("sample_id") != sample.sample_id:
            raise ValueError(f"Sample {sample.sample_id} metric sample_id mismatch")


def _write_circuits(
    result: DatasetGenerationResult,
    artifact_paths: dict[str, list[Path]],
    overwrite: bool,
) -> list[Path]:
    qpy = _load_qpy_module()
    circuits_by_id = {
        sample.clean_circuit_id: sample.clean_circuit for sample in result.samples
    }
    circuits_by_id.update(
        {
            sample.distorted_circuit_id: sample.distorted_circuit
            for sample in result.samples
        }
    )
    written: list[Path] = []
    for path in artifact_paths["circuits"]:
        circuit_id = path.stem
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing circuit artifact: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            qpy.dump(circuits_by_id[circuit_id], handle)
        written.append(path)
    return written


def _write_probabilities(
    result: DatasetGenerationResult,
    root: Path,
    overwrite: bool,
) -> list[Path]:
    written: list[Path] = []
    seen: set[str] = set()
    for sample in result.samples:
        for run_id, probabilities in (
            (sample.clean_run_id, sample.clean_result.probabilities),
            (sample.distorted_run_id, sample.distorted_result.probabilities),
        ):
            if run_id in seen:
                continue
            seen.add(run_id)
            written.append(
                _write_json(
                    root / "artifacts" / "probabilities" / f"{run_id}.json",
                    probabilities,
                    overwrite=overwrite,
                )
            )
    return written


def _write_statevectors(
    result: DatasetGenerationResult,
    root: Path,
    overwrite: bool,
) -> list[Path]:
    if not result.config.store_statevectors:
        return []
    written: list[Path] = []
    seen: set[str] = set()
    for sample in result.samples:
        for run_id, statevector in (
            (sample.clean_run_id, sample.clean_result.statevector),
            (sample.distorted_run_id, sample.distorted_result.statevector),
        ):
            if run_id in seen:
                continue
            seen.add(run_id)
            path = root / "artifacts" / "statevectors" / f"{run_id}.npy"
            if path.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing statevector artifact: {path}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(statevector.data))
            written.append(path)
    return written


def _shot_counts_by_run(result: DatasetGenerationResult) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for sample in result.samples:
        if sample.clean_shot_run_id and sample.clean_shot_result is not None:
            counts[sample.clean_shot_run_id] = sample.clean_shot_result.counts
        if sample.distorted_shot_run_id and sample.distorted_shot_result is not None:
            counts[sample.distorted_shot_run_id] = sample.distorted_shot_result.counts
    return counts


def _write_counts(
    result: DatasetGenerationResult,
    root: Path,
    overwrite: bool,
) -> list[Path]:
    written: list[Path] = []
    for run_id, counts in sorted(_shot_counts_by_run(result).items()):
        written.append(
            _write_json(
                root / "artifacts" / "counts" / f"{run_id}.json",
                counts,
                overwrite=overwrite,
            )
        )
    return written


def _metric_records_for_manifest(result: DatasetGenerationResult) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for record in result.metric_records:
        row = record.to_dict()
        row["metadata"] = dict(row["metadata"])
        row["metadata"]["empty_metric_map_storage_encoding"] = EMPTY_MAP_ENCODING
        row["hilbert_metrics"] = None
        row["parameter_metrics"] = None
        row["topology_metrics"] = None
        encoded.append(row)
    return encoded


def _managed_files(result: DatasetGenerationResult) -> list[str]:
    root = Path(".")
    artifact_paths = _planned_paths(result, root)
    managed = [
        "generation_config.json",
        "dataset_summary.json",
        DATASET_COMPLETE_NAME,
        *[f"manifests/{name}.parquet" for name in MANIFEST_NAMES.values()],
        *[
            path.as_posix().removeprefix("./")
            for paths in artifact_paths.values()
            for path in paths
        ],
    ]
    return sorted(set(managed))


def _completion_marker_payload(
    result: DatasetGenerationResult,
    managed_files: list[str],
) -> dict[str, Any]:
    return {
        "complete": True,
        "dataset_name": result.dataset_name,
        "schema_version": result.schema_version,
        "scientific_generation_id": result.scientific_generation_id,
        "config_id": result.config_id,
        "sample_count": len(result.samples),
        "manifest_count": len(MANIFEST_NAMES),
        "managed_files": sorted(managed_files),
    }


def _validate_completion_marker(
    root: Path,
    result: DatasetGenerationResult,
    managed_files: list[str],
) -> None:
    marker = root / DATASET_COMPLETE_NAME
    if not marker.is_file():
        raise FileNotFoundError(f"Completion marker missing: {marker}")
    payload = json.loads(marker.read_text())
    expected = _completion_marker_payload(result, managed_files)
    if payload != expected:
        raise ValueError("dataset_complete.json content does not match committed dataset")
    for entry in payload["managed_files"]:
        _safe_managed_relative_path(root, entry)


def _read_typed_records(
    root: Path,
) -> tuple[
    list[DatasetSampleRecord],
    list[CircuitRecord],
    list[SimulationRecord],
    list[DistortionRecord],
    list[MetricRecord],
]:
    reader = ManifestReader(root / "manifests")
    return (
        reader.read_typed_records("sample_manifest", DatasetSampleRecord),
        reader.read_typed_records("circuit_manifest", CircuitRecord),
        reader.read_typed_records("simulation_manifest", SimulationRecord),
        reader.read_typed_records("distortion_manifest", DistortionRecord),
        reader.read_typed_records("metric_manifest", MetricRecord),
    )


def _validate_persisted_dataset(
    root: Path,
    result: DatasetGenerationResult,
    managed_files: list[str],
    *,
    require_marker: bool,
) -> None:
    for entry in managed_files:
        if entry == DATASET_COMPLETE_NAME and not require_marker:
            continue
        path = root / _safe_managed_relative_path(root, entry)
        if not path.is_file():
            raise FileNotFoundError(
                f"Managed Phase 7 file is missing or not a file: {entry}"
            )
    if not require_marker and (root / DATASET_COMPLETE_NAME).exists():
        raise ValueError("Completion marker must not exist before publication commit")
    (
        sample_records,
        circuit_records,
        simulation_records,
        distortion_records,
        metric_records,
    ) = _read_typed_records(root)
    validate_dataset_joins(
        sample_records,
        circuit_records,
        simulation_records,
        distortion_records,
        metric_records,
    )
    verify_dataset_references(
        root,
        circuit_records,
        simulation_records,
        require_statevectors=result.config.store_statevectors,
    )
    if require_marker:
        _validate_completion_marker(root, result, managed_files)


def _is_missing_manifest_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _legacy_managed_files_from_manifests(root: Path) -> list[str]:
    """Derive explicitly referenced legacy artifacts without scanning directories."""
    managed = [
        "generation_config.json",
        "dataset_summary.json",
        DATASET_COMPLETE_NAME,
        *[f"manifests/{name}.parquet" for name in MANIFEST_NAMES.values()],
    ]
    reader = ManifestReader(root / "manifests")

    circuit_manifest = reader.manifest_path("circuit_manifest")
    if circuit_manifest.is_file():
        for row in reader.read_records("circuit_manifest"):
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("Legacy circuit_manifest metadata must be a mapping")
            reference = metadata.get("artifact_ref")
            if _is_missing_manifest_value(reference):
                continue
            if not isinstance(reference, str):
                raise ValueError("Legacy circuit artifact_ref must be a string")
            _safe_managed_relative_path(root, reference)
            managed.append(reference)

    simulation_manifest = reader.manifest_path("simulation_manifest")
    if simulation_manifest.is_file():
        for row in reader.read_records("simulation_manifest"):
            run_id = row.get("run_id", "<unknown>")
            for field_name in (
                "statevector_ref",
                "counts_ref",
                "probabilities_ref",
            ):
                reference = row.get(field_name)
                if _is_missing_manifest_value(reference):
                    continue
                if not isinstance(reference, str):
                    raise ValueError(
                        f"Legacy SimulationRecord {run_id} field {field_name} must be a string"
                    )
                _safe_managed_relative_path(root, reference)
                managed.append(reference)

    return sorted(set(managed))


def _existing_managed_files(root: Path) -> list[str]:
    marker = root / DATASET_COMPLETE_NAME
    if not marker.exists():
        return []
    try:
        payload = json.loads(marker.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing completion marker is malformed: {marker}") from exc

    if "managed_files" not in payload:
        return _legacy_managed_files_from_manifests(root)

    managed = payload["managed_files"]
    if not isinstance(managed, list):
        raise ValueError("Existing completion marker managed_files must be a list")

    safe: list[str] = []
    for entry in managed:
        if not isinstance(entry, str):
            raise ValueError(
                "Existing completion marker managed_files contains non-string entry"
            )
        _safe_managed_relative_path(root, entry)
        safe.append(entry)
    return sorted(set(safe))


def _relative_paths_to_publish(managed_files: list[str]) -> list[str]:
    return [entry for entry in managed_files if entry != DATASET_COMPLETE_NAME]


def _backup_existing_file(root: Path, backup_root: Path, relative: str) -> bool:
    source = root / relative
    if not source.exists():
        return False
    if not source.is_file():
        raise ValueError(
            f"Known Phase 7 target is not a file and will not be replaced: {relative}"
        )
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    return True


def _restore_backed_up_files(
    root: Path,
    backup_root: Path,
    backed_up: list[str],
) -> None:
    for relative in reversed(backed_up):
        source = backup_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Rollback backup is missing managed file: {relative}")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Rollback destination is not a file: {relative}")
            destination.unlink()
        os.replace(source, destination)


def _remove_published_paths(root: Path, relatives: list[str]) -> None:
    for relative in sorted(set(relatives), reverse=True):
        path = root / relative
        if path.exists() and path.is_file():
            path.unlink()
    for parent in [
        root / "artifacts" / "counts",
        root / "artifacts" / "statevectors",
        root / "artifacts" / "probabilities",
        root / "artifacts" / "circuits",
        root / "artifacts",
        root / "manifests",
    ]:
        try:
            parent.rmdir()
        except OSError:
            pass


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _write_completion_marker(
    root: Path,
    result: DatasetGenerationResult,
    managed_files: list[str],
) -> Path:
    marker = root / DATASET_COMPLETE_NAME
    temporary = root / f".{DATASET_COMPLETE_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        _write_json(
            temporary,
            _completion_marker_payload(result, managed_files),
            overwrite=True,
        )
        _atomic_replace(temporary, marker)
        return marker
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_staged_dataset(
    staging_root: Path,
    root: Path,
    result: DatasetGenerationResult,
    managed_files: list[str],
    *,
    overwrite: bool,
) -> None:
    del overwrite  # overwrite conflicts are checked before staging publication.
    old_managed = _existing_managed_files(root) if root.exists() else []
    new_without_marker = _relative_paths_to_publish(managed_files)
    old_without_marker = _relative_paths_to_publish(old_managed)
    paths_to_backup = sorted(
        set(new_without_marker)
        | set(old_without_marker)
        | ({DATASET_COMPLETE_NAME} if (root / DATASET_COMPLETE_NAME).exists() else set())
    )
    backup_root = root.with_name(f".{root.name}.triqto-backup-{uuid.uuid4().hex}")
    backed_up: list[str] = []
    published: list[str] = []
    marker_published = False

    try:
        if backup_root.exists():
            raise RuntimeError(f"Refusing to reuse existing backup directory: {backup_root}")

        for relative in paths_to_backup:
            _safe_managed_relative_path(root, relative)
            if _backup_existing_file(root, backup_root, relative):
                backed_up.append(relative)

        for relative in sorted(new_without_marker):
            source = staging_root / relative
            if not source.is_file():
                raise FileNotFoundError(
                    f"Staged managed file missing before publication: {relative}"
                )
            _atomic_replace(source, root / relative)
            published.append(relative)

        _validate_persisted_dataset(
            root,
            result,
            managed_files,
            require_marker=False,
        )
        _write_completion_marker(root, result, managed_files)
        marker_published = True
        _validate_persisted_dataset(
            root,
            result,
            managed_files,
            require_marker=True,
        )
    except Exception:
        if marker_published and (root / DATASET_COMPLETE_NAME).is_file():
            (root / DATASET_COMPLETE_NAME).unlink()
        _remove_published_paths(root, published)
        _restore_backed_up_files(root, backup_root, backed_up)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        raise
    else:
        if backup_root.exists():
            shutil.rmtree(backup_root)


def write_dataset(
    result: DatasetGenerationResult,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> DatasetWriteResult:
    """Write a generated Phase 7 dataset transactionally."""
    root = Path(output_root)
    staging_root = root.with_name(f".{root.name}.triqto-staging-{uuid.uuid4().hex}")
    managed_files = _managed_files(result)
    artifact_paths = _planned_paths(result, staging_root)
    final_artifact_paths = _planned_paths(result, root)
    final_known_paths = [root / entry for entry in managed_files]

    if not overwrite:
        for path in _unique_sorted(final_known_paths):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing dataset file: {path}")

    if staging_root.exists():
        shutil.rmtree(staging_root)

    try:
        _write_json(
            staging_root / "generation_config.json",
            config_to_dict(result.config),
            overwrite=False,
        )
        _write_json(
            staging_root / "dataset_summary.json",
            result.summary,
            overwrite=False,
        )
        _write_circuits(result, artifact_paths, overwrite=False)
        _write_probabilities(result, staging_root, overwrite=False)
        _write_statevectors(result, staging_root, overwrite=False)
        _write_counts(result, staging_root, overwrite=False)

        circuit_records, simulation_records = _records_with_artifact_refs(result)
        manifest_writer = ManifestWriter(staging_root / "manifests")
        manifest_writer.write_records(
            "sample_manifest", result.sample_records, overwrite=False
        )
        manifest_writer.write_records(
            "circuit_manifest", circuit_records, overwrite=False
        )
        manifest_writer.write_records(
            "simulation_manifest", simulation_records, overwrite=False
        )
        manifest_writer.write_records(
            "distortion_manifest", result.distortion_records, overwrite=False
        )
        manifest_writer.write_records(
            "metric_manifest",
            _metric_records_for_manifest(result),
            overwrite=False,
        )

        _validate_persisted_dataset(
            staging_root,
            result,
            managed_files,
            require_marker=False,
        )
        _publish_staged_dataset(
            staging_root,
            root,
            result,
            managed_files,
            overwrite=overwrite,
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    final_manifest_paths = {
        name: root / "manifests" / f"{manifest}.parquet"
        for name, manifest in MANIFEST_NAMES.items()
    }
    written_paths = _unique_sorted([root / entry for entry in managed_files])
    for path in written_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Reported written path does not exist: {path}")

    return DatasetWriteResult(
        output_root=root,
        written_paths=written_paths,
        manifest_paths=dict(sorted(final_manifest_paths.items())),
        artifact_paths={
            key: _unique_sorted(paths)
            for key, paths in sorted(final_artifact_paths.items())
        },
        summary_path=root / "dataset_summary.json",
        config_path=root / "generation_config.json",
    )
