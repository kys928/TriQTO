"""Artifact writer for Phase 7 generated datasets."""
from __future__ import annotations

from dataclasses import replace
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


def _ref_to_path(root: Path, record_id: str, reference: str | None) -> Path | None:
    if reference is None:
        return None
    ref_path = Path(reference)
    if ref_path.is_absolute():
        raise ValueError(f"Record {record_id} has absolute artifact reference: {reference}")
    return root / ref_path


def verify_dataset_references(root: str | Path, circuit_records: list[Any], simulation_records: list[Any]) -> None:
    """Verify all manifest artifact references are relative and point to existing files."""
    root_path = Path(root)
    for record in circuit_records:
        reference = record.metadata.get("artifact_ref")
        path = _ref_to_path(root_path, record.circuit_id, reference)
        if path is None:
            raise ValueError(f"Circuit record {record.circuit_id} is missing artifact_ref")
        if not path.exists():
            raise FileNotFoundError(f"Circuit record {record.circuit_id} references missing artifact: {reference}")
    for record in simulation_records:
        for field_name in ("statevector_ref", "probabilities_ref", "counts_ref"):
            reference = getattr(record, field_name)
            path = _ref_to_path(root_path, record.run_id, reference)
            if reference is not None and path is not None and not path.exists():
                raise FileNotFoundError(
                    f"Simulation record {record.run_id} field {field_name} references missing artifact: {reference}"
                )


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
    return [
        replace(
            record,
            hilbert_metrics=None,
            parameter_metrics=None,
            topology_metrics=None,
        )
        for record in result.metric_records
    ]


def write_dataset(result: DatasetGenerationResult, output_root: str | Path, *, overwrite: bool = False) -> DatasetWriteResult:
    """Write a generated Phase 7 dataset to manifests and external artifacts."""
    root = Path(output_root)
    artifact_paths = _planned_paths(result, root)
    fixed_paths = [
        root / "generation_config.json",
        root / "dataset_summary.json",
        *[root / "manifests" / f"{name}.parquet" for name in MANIFEST_NAMES.values()],
        *[path for paths in artifact_paths.values() for path in paths],
    ]
    for path in _unique_sorted(fixed_paths):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing dataset file: {path}")

    written_paths = [
        _write_json(root / "generation_config.json", config_to_dict(result.config), overwrite=overwrite),
        _write_json(root / "dataset_summary.json", result.summary, overwrite=overwrite),
    ]
    written_paths.extend(_write_circuits(result, artifact_paths, overwrite))
    written_paths.extend(_write_probabilities(result, root, overwrite))
    written_paths.extend(_write_statevectors(result, root, overwrite))
    written_paths.extend(_write_counts(result, root, overwrite))

    circuit_records, simulation_records = _records_with_artifact_refs(result)
    manifest_writer = ManifestWriter(root / "manifests")
    manifest_paths = {
        "sample_manifest": manifest_writer.write_records("sample_manifest", result.sample_records, overwrite=overwrite),
        "circuit_manifest": manifest_writer.write_records("circuit_manifest", circuit_records, overwrite=overwrite),
        "simulation_manifest": manifest_writer.write_records("simulation_manifest", simulation_records, overwrite=overwrite),
        "distortion_manifest": manifest_writer.write_records("distortion_manifest", result.distortion_records, overwrite=overwrite),
        "metric_manifest": manifest_writer.write_records("metric_manifest", _metric_records_for_manifest(result), overwrite=overwrite),
    }
    written_paths.extend(manifest_paths.values())
    verify_dataset_references(root, circuit_records, simulation_records)
    all_written = _unique_sorted(written_paths)
    for path in all_written:
        if not path.exists():
            raise FileNotFoundError(f"Reported written path does not exist: {path}")
    return DatasetWriteResult(
        output_root=root,
        written_paths=all_written,
        manifest_paths=dict(sorted(manifest_paths.items())),
        artifact_paths={key: _unique_sorted(paths) for key, paths in sorted(artifact_paths.items())},
        summary_path=root / "dataset_summary.json",
        config_path=root / "generation_config.json",
    )
