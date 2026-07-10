"""Artifact writer for Phase 7 generated datasets."""
from __future__ import annotations

from dataclasses import replace
import shutil
import uuid
import json
from pathlib import Path
from typing import Any

import numpy as np

from triqto.storage import ManifestWriter

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
KNOWN_PHASE7_TOP_LEVEL = {"generation_config.json", "dataset_summary.json", DATASET_COMPLETE_NAME, "manifests", "artifacts"}


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


def _records_with_artifact_refs(result: DatasetGenerationResult) -> tuple[list[Any], list[Any]]:
    circuit_records = []
    for record in result.circuit_records:
        metadata = dict(record.metadata)
        metadata["artifact_ref"] = _relative_ref(f"artifacts/circuits/{record.circuit_id}.qpy")
        circuit_records.append(replace(record, metadata=metadata))

    simulation_records = []
    for record in result.simulation_records:
        metadata = dict(record.metadata)
        if record.simulation_mode == "ideal_statevector":
            statevector_ref = metadata.pop("statevector_ref", None)
            probabilities_ref = metadata.pop("probabilities_ref", None)
            simulation_records.append(
                replace(
                    record,
                    statevector_ref=statevector_ref,
                    probabilities_ref=probabilities_ref,
                    metadata=metadata,
                )
            )
        elif record.simulation_mode == "ideal_shot":
            counts_ref = metadata.pop("counts_ref", None)
            simulation_records.append(replace(record, counts_ref=counts_ref, metadata=metadata))
        else:
            simulation_records.append(record)
    return circuit_records, simulation_records


def _unique_sorted(paths: list[Path]) -> list[Path]:
    return sorted(set(paths), key=lambda path: path.as_posix())


def _planned_paths(result: DatasetGenerationResult, root: Path) -> dict[str, list[Path]]:
    circuits = [root / "artifacts" / "circuits" / f"{record.circuit_id}.qpy" for record in result.circuit_records]
    probabilities = []
    statevectors = []
    for sample in result.samples:
        probabilities.append(root / "artifacts" / "probabilities" / f"{sample.clean_run_id}.json")
        probabilities.append(root / "artifacts" / "probabilities" / f"{sample.distorted_run_id}.json")
        if result.config.store_statevectors:
            statevectors.append(root / "artifacts" / "statevectors" / f"{sample.clean_run_id}.npy")
            statevectors.append(root / "artifacts" / "statevectors" / f"{sample.distorted_run_id}.npy")
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
        raise ValueError(f"Record {record_id} field {field_name} has empty or non-string artifact reference: {reference!r}")
    if Path(reference).as_posix() != reference:
        raise ValueError(f"Record {record_id} field {field_name} is not a normalized POSIX path: {reference}")
    ref_path = Path(reference)
    if ref_path.is_absolute():
        raise ValueError(f"Record {record_id} field {field_name} has absolute artifact reference: {reference}")
    resolved_root = root.resolve()
    resolved_target = (root / ref_path).resolve()
    if resolved_target == resolved_root:
        raise ValueError(f"Record {record_id} field {field_name} references dataset root: {reference}")
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Record {record_id} field {field_name} escapes dataset root: {reference}") from exc
    return resolved_target


def _require_file(root: Path, record_id: str, field_name: str, reference: Any) -> Path:
    path = _safe_reference_path(root, record_id, field_name, reference)
    if not path.exists():
        raise FileNotFoundError(f"Record {record_id} field {field_name} references missing artifact: {reference}")
    if not path.is_file():
        raise ValueError(f"Record {record_id} field {field_name} does not reference a file: {reference}")
    return path


def _require_absent(record_id: str, field_name: str, reference: Any) -> None:
    if reference is not None:
        raise ValueError(f"Record {record_id} field {field_name} must be absent for this simulation mode: {reference}")


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
        _require_file(root_path, record.circuit_id, "metadata.artifact_ref", record.metadata.get("artifact_ref"))
    for record in simulation_records:
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
            if not record.metadata.get("source_run_id"):
                raise ValueError(f"Record {record.run_id} field metadata.source_run_id is required for ideal_shot")
            if record.metadata.get("sampling_source") != "sampled_from_exact_born_probabilities":
                raise ValueError(f"Record {record.run_id} field metadata.sampling_source is invalid for ideal_shot")
        else:
            raise ValueError(f"Record {record.run_id} has unsupported simulation_mode: {record.simulation_mode}")


def validate_dataset_joins(
    sample_records: list[Any],
    circuit_records: list[Any],
    simulation_records: list[Any],
    distortion_records: list[Any],
    metric_records: list[Any],
) -> None:
    """Validate semantic joins among Phase 7 sample, circuit, simulation, distortion, and metric records."""
    circuits = {record.circuit_id: record for record in circuit_records}
    simulations = {record.run_id: record for record in simulation_records}
    distortions = {record.distortion_id: record for record in distortion_records}
    metrics = {record.metric_id: record for record in metric_records}
    for sample in sample_records:
        clean_circuit = circuits[sample.clean_circuit_id]
        distorted_circuit = circuits[sample.distorted_circuit_id]
        if clean_circuit.metadata.get("role") != "clean":
            raise ValueError(f"Sample {sample.sample_id} clean_circuit_id does not point to a clean CircuitRecord")
        if distorted_circuit.metadata.get("role") != "distorted":
            raise ValueError(f"Sample {sample.sample_id} distorted_circuit_id does not point to a distorted CircuitRecord")
        if distorted_circuit.metadata.get("source_clean_circuit_id") != sample.clean_circuit_id:
            raise ValueError(f"Sample {sample.sample_id} distorted circuit source_clean_circuit_id mismatch")
        clean_run = simulations[sample.clean_run_id]
        distorted_run = simulations[sample.distorted_run_id]
        if clean_run.simulation_mode != "ideal_statevector" or clean_run.circuit_id != sample.clean_circuit_id:
            raise ValueError(f"Sample {sample.sample_id} clean_run_id does not point to clean ideal_statevector run")
        if distorted_run.simulation_mode != "ideal_statevector" or distorted_run.circuit_id != sample.distorted_circuit_id:
            raise ValueError(f"Sample {sample.sample_id} distorted_run_id does not point to distorted ideal_statevector run")
        distortion = distortions[sample.distortion_id]
        if distortion.circuit_id != sample.clean_circuit_id:
            raise ValueError(f"Sample {sample.sample_id} distortion circuit_id mismatch")
        if distortion.metadata.get("distorted_circuit_id") != sample.distorted_circuit_id:
            raise ValueError(f"Sample {sample.sample_id} distortion distorted_circuit_id mismatch")
        metric = metrics[sample.metric_id]
        if metric.run_id != sample.distorted_run_id or metric.circuit_id != sample.distorted_circuit_id or metric.distortion_id != sample.distortion_id:
            raise ValueError(f"Sample {sample.sample_id} metric record IDs do not match sample joins")
        if metric.metadata.get("clean_run_id") != sample.clean_run_id:
            raise ValueError(f"Sample {sample.sample_id} metric clean_run_id mismatch")
        if metric.metadata.get("distorted_run_id") != sample.distorted_run_id:
            raise ValueError(f"Sample {sample.sample_id} metric distorted_run_id mismatch")
        if metric.metadata.get("sample_id") != sample.sample_id:
            raise ValueError(f"Sample {sample.sample_id} metric sample_id mismatch")


def _write_circuits(result: DatasetGenerationResult, artifact_paths: dict[str, list[Path]], overwrite: bool) -> list[Path]:
    qpy = _load_qpy_module()
    circuits_by_id = {sample.clean_circuit_id: sample.clean_circuit for sample in result.samples}
    circuits_by_id.update({sample.distorted_circuit_id: sample.distorted_circuit for sample in result.samples})
    written = []
    for path in artifact_paths["circuits"]:
        circuit_id = path.stem
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing circuit artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("wb") as handle:
                qpy.dump(circuits_by_id[circuit_id], handle)
        except Exception as exc:  # pragma: no cover - defensive IO wrapper
            raise RuntimeError(f"Failed to write QPY circuit artifact {path}") from exc
        written.append(path)
    return written


def _write_probabilities(result: DatasetGenerationResult, root: Path, overwrite: bool) -> list[Path]:
    written = []
    seen = set()
    for sample in result.samples:
        for run_id, probabilities in (
            (sample.clean_run_id, sample.clean_result.probabilities),
            (sample.distorted_run_id, sample.distorted_result.probabilities),
        ):
            if run_id in seen:
                continue
            seen.add(run_id)
            written.append(
                _write_json(root / "artifacts" / "probabilities" / f"{run_id}.json", probabilities, overwrite=overwrite)
            )
    return written


def _write_statevectors(result: DatasetGenerationResult, root: Path, overwrite: bool) -> list[Path]:
    if not result.config.store_statevectors:
        return []
    written = []
    seen = set()
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
                raise FileExistsError(f"Refusing to overwrite existing statevector artifact: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(statevector.data))
            written.append(path)
    return written


def _shot_counts_by_run(result: DatasetGenerationResult) -> dict[str, dict[str, int]]:
    counts = {}
    for sample in result.samples:
        if sample.clean_shot_run_id and sample.clean_shot_result is not None:
            counts[sample.clean_shot_run_id] = sample.clean_shot_result.counts
        if sample.distorted_shot_run_id and sample.distorted_shot_result is not None:
            counts[sample.distorted_shot_run_id] = sample.distorted_shot_result.counts
    return counts


def _write_counts(result: DatasetGenerationResult, root: Path, overwrite: bool) -> list[Path]:
    counts_by_run = _shot_counts_by_run(result)
    written = []
    for run_id in sorted(counts_by_run):
        written.append(_write_json(root / "artifacts" / "counts" / f"{run_id}.json", counts_by_run[run_id], overwrite=overwrite))
    return written


def _metric_records_for_manifest(result: DatasetGenerationResult) -> list[Any]:
    """Return metric records with uncomputed empty families encoded as null for Parquet."""
    encoded = []
    for record in result.metric_records:
        metadata = dict(record.metadata)
        metadata["empty_metric_map_storage_encoding"] = "parquet_null_normalized_to_empty_dict"
        encoded.append(
            replace(
                record,
                hilbert_metrics=None,
                parameter_metrics=None,
                topology_metrics=None,
                metadata=metadata,
            )
        )
    return encoded


def _merge_staging_into_existing_root(staging_root: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for child in staging_root.iterdir():
        target = root / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        child.replace(target)
    shutil.rmtree(staging_root)


def write_dataset(result: DatasetGenerationResult, output_root: str | Path, *, overwrite: bool = False) -> DatasetWriteResult:
    """Write a generated Phase 7 dataset to manifests and external artifacts."""
    root = Path(output_root)
    staging_root = root.with_name(f".{root.name}.triqto-staging-{uuid.uuid4().hex}")
    artifact_paths = _planned_paths(result, staging_root)
    final_artifact_paths_for_conflict = _planned_paths(result, root)
    fixed_paths = [
        root / "generation_config.json",
        root / "dataset_summary.json",
        root / DATASET_COMPLETE_NAME,
        *[root / "manifests" / f"{name}.parquet" for name in MANIFEST_NAMES.values()],
        *[path for paths in final_artifact_paths_for_conflict.values() for path in paths],
    ]
    for path in _unique_sorted(fixed_paths):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing dataset file: {path}")

    if staging_root.exists():
        shutil.rmtree(staging_root)
    try:
        written_paths = [
            _write_json(staging_root / "generation_config.json", config_to_dict(result.config), overwrite=overwrite),
            _write_json(staging_root / "dataset_summary.json", result.summary, overwrite=overwrite),
        ]
        written_paths.extend(_write_circuits(result, artifact_paths, overwrite))
        written_paths.extend(_write_probabilities(result, staging_root, overwrite))
        written_paths.extend(_write_statevectors(result, staging_root, overwrite))
        written_paths.extend(_write_counts(result, staging_root, overwrite))

        circuit_records, simulation_records = _records_with_artifact_refs(result)
        manifest_writer = ManifestWriter(staging_root / "manifests")
        manifest_paths = {
            "sample_manifest": manifest_writer.write_records("sample_manifest", result.sample_records, overwrite=overwrite),
            "circuit_manifest": manifest_writer.write_records("circuit_manifest", circuit_records, overwrite=overwrite),
            "simulation_manifest": manifest_writer.write_records("simulation_manifest", simulation_records, overwrite=overwrite),
            "distortion_manifest": manifest_writer.write_records("distortion_manifest", result.distortion_records, overwrite=overwrite),
            "metric_manifest": manifest_writer.write_records("metric_manifest", _metric_records_for_manifest(result), overwrite=overwrite),
        }
        written_paths.extend(manifest_paths.values())
        validate_dataset_joins(result.sample_records, circuit_records, simulation_records, result.distortion_records, result.metric_records)
        verify_dataset_references(staging_root, circuit_records, simulation_records, require_statevectors=result.config.store_statevectors)
        written_paths.append(
            _write_json(
                staging_root / DATASET_COMPLETE_NAME,
                {"dataset_name": result.dataset_name, "scientific_generation_id": result.scientific_generation_id, "complete": True},
                overwrite=overwrite,
            )
        )
        if overwrite and root.exists():
            for child_name in KNOWN_PHASE7_TOP_LEVEL:
                child = root / child_name
                if child.is_dir():
                    shutil.rmtree(child)
                elif child.exists():
                    child.unlink()
        staging_root.replace(root) if not root.exists() else _merge_staging_into_existing_root(staging_root, root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    final_artifact_paths = _planned_paths(result, root)
    final_manifest_paths = {name: root / "manifests" / f"{manifest}.parquet" for name, manifest in MANIFEST_NAMES.items()}
    final_written = [root / path.relative_to(staging_root) for path in written_paths]
    all_written = _unique_sorted(final_written)
    for path in all_written:
        if not path.exists():
            raise FileNotFoundError(f"Reported written path does not exist: {path}")
    return DatasetWriteResult(
        output_root=root,
        written_paths=all_written,
        manifest_paths=dict(sorted(final_manifest_paths.items())),
        artifact_paths={key: _unique_sorted(paths) for key, paths in sorted(final_artifact_paths.items())},
        summary_path=root / "dataset_summary.json",
        config_path=root / "generation_config.json",
    )
