"""NPZ persistence and immutable publication for Phase 8 graph datasets."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np

from triqto.storage.graph_schema import GraphPairRecord, GraphRecord
from triqto.storage.manifest import ManifestReader, ManifestWriter

from .config import save_graph_config
from .constants import (
    GRAPH_ARRAY_NAMES,
    GRAPH_ARTIFACT_SCHEMA_VERSION,
    GRAPH_METADATA_ARRAY_NAME,
    PAIR_ARRAY_NAMES,
    PAIR_ARTIFACT_SCHEMA_VERSION,
    PAIR_METADATA_ARRAY_NAME,
)
from .identities import graph_content_hash, pair_content_hash
from .models import (
    CircuitGraphData,
    GraphConversionResult,
    GraphSamplePair,
    GraphWriteResult,
)
from .source import verify_source_snapshot
from .utils import (
    json_copy,
    resolve_safe_file,
    strict_json_loads,
    write_strict_json,
)
from .validation import (
    validate_graph_data,
    validate_graph_dataset_joins,
    validate_pair_data,
)


def graph_arrays(graph: CircuitGraphData) -> dict[str, np.ndarray]:
    return {name: getattr(graph, name) for name in GRAPH_ARRAY_NAMES}


def _json_bytes_array(payload: Mapping[str, Any]) -> np.ndarray:
    text = write_json_text(payload)
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def write_json_text(payload: Mapping[str, Any]) -> str:
    from .utils import strict_json_dumps

    return strict_json_dumps(json_copy(dict(payload)), indent=None)


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


def _require_metadata_keys(
    metadata: Mapping[str, Any],
    required: set[str],
    artifact_name: str,
) -> None:
    missing = required - set(metadata)
    unexpected = set(metadata) - required
    if missing or unexpected:
        raise ValueError(
            f"{artifact_name} metadata-key mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _graph_metadata(graph: CircuitGraphData) -> dict[str, Any]:
    return {
        "artifact_schema_version": GRAPH_ARTIFACT_SCHEMA_VERSION,
        "graph_id": graph.graph_id,
        "circuit_id": graph.circuit_id,
        "source_run_id": graph.source_run_id,
        "role": graph.role,
        "family": graph.family,
        "graph_schema_version": graph.graph_schema_version,
        "n_qubits": graph.n_qubits,
        "n_clbits": graph.n_clbits,
        "source_sample_ids": list(graph.source_sample_ids),
        "node_feature_names": list(graph.node_feature_names),
        "edge_feature_names": list(graph.edge_feature_names),
        "gate_feature_names": list(graph.gate_feature_names),
        "global_feature_names": list(graph.global_feature_names),
        "exact_probability_available_mask": graph.exact_probability_available_mask,
        "supplemental_counts_available_mask": graph.supplemental_counts_available_mask,
        "hilbert_available_mask": graph.hilbert_available_mask,
        "supplemental_shots": graph.supplemental_shots,
        "scientific_metadata": graph.scientific_metadata,
        "provenance_metadata": graph.provenance_metadata,
        "content_hash": graph_content_hash(graph),
    }


def save_graph_artifact(graph: CircuitGraphData, path: str | Path) -> Path:
    validate_graph_data(graph)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = graph_arrays(graph)
    arrays[GRAPH_METADATA_ARRAY_NAME] = _json_bytes_array(_graph_metadata(graph))
    np.savez_compressed(target, **arrays)
    return target


def _require_exact_npz_names(
    names: set[str],
    expected: set[str],
    artifact_name: str,
) -> None:
    missing = expected - names
    unexpected = names - expected
    if missing or unexpected:
        raise ValueError(
            f"{artifact_name} array-name mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def load_graph_artifact(
    path: str | Path,
    expected_content_hash: str | None = None,
) -> CircuitGraphData:
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        expected_names = set(GRAPH_ARRAY_NAMES) | {GRAPH_METADATA_ARRAY_NAME}
        _require_exact_npz_names(set(payload.files), expected_names, "graph artifact")
        arrays = {name: payload[name].copy() for name in GRAPH_ARRAY_NAMES}
        metadata = _decode_json_bytes(
            payload[GRAPH_METADATA_ARRAY_NAME],
            GRAPH_METADATA_ARRAY_NAME,
        )
    graph_metadata_keys = {
        "artifact_schema_version",
        "graph_id",
        "circuit_id",
        "source_run_id",
        "role",
        "family",
        "graph_schema_version",
        "n_qubits",
        "n_clbits",
        "source_sample_ids",
        "node_feature_names",
        "edge_feature_names",
        "gate_feature_names",
        "global_feature_names",
        "exact_probability_available_mask",
        "supplemental_counts_available_mask",
        "hilbert_available_mask",
        "supplemental_shots",
        "scientific_metadata",
        "provenance_metadata",
        "content_hash",
    }
    _require_metadata_keys(metadata, graph_metadata_keys, "graph artifact")
    if metadata.get("artifact_schema_version") != GRAPH_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported graph artifact schema version")
    for name in (
        "source_sample_ids",
        "node_feature_names",
        "edge_feature_names",
        "gate_feature_names",
        "global_feature_names",
    ):
        if not isinstance(metadata[name], list):
            raise TypeError(f"graph artifact metadata.{name} must be a list")
    for name in ("scientific_metadata", "provenance_metadata"):
        if not isinstance(metadata[name], Mapping):
            raise TypeError(f"graph artifact metadata.{name} must be a mapping")
    graph = CircuitGraphData(
        graph_id=metadata["graph_id"],
        circuit_id=metadata["circuit_id"],
        source_run_id=metadata["source_run_id"],
        role=metadata["role"],
        family=metadata["family"],
        graph_schema_version=metadata["graph_schema_version"],
        n_qubits=metadata["n_qubits"],
        n_clbits=metadata["n_clbits"],
        **arrays,
        source_sample_ids=tuple(metadata["source_sample_ids"]),
        node_feature_names=tuple(metadata["node_feature_names"]),
        edge_feature_names=tuple(metadata["edge_feature_names"]),
        gate_feature_names=tuple(metadata["gate_feature_names"]),
        global_feature_names=tuple(metadata["global_feature_names"]),
        exact_probability_available_mask=metadata["exact_probability_available_mask"],
        supplemental_counts_available_mask=metadata["supplemental_counts_available_mask"],
        hilbert_available_mask=metadata["hilbert_available_mask"],
        supplemental_shots=metadata.get("supplemental_shots"),
        scientific_metadata=dict(metadata["scientific_metadata"]),
        provenance_metadata=dict(metadata["provenance_metadata"]),
    )
    validate_graph_data(graph)
    actual_hash = graph_content_hash(graph)
    stored_hash = metadata.get("content_hash")
    if stored_hash != actual_hash:
        raise ValueError("graph artifact stored content_hash mismatch")
    if expected_content_hash is not None and expected_content_hash != actual_hash:
        raise ValueError("graph artifact content_hash does not match manifest")
    return graph


def _pair_metadata(pair: GraphSamplePair) -> dict[str, Any]:
    return {
        "artifact_schema_version": PAIR_ARTIFACT_SCHEMA_VERSION,
        "graph_pair_id": pair.graph_pair_id,
        "sample_id": pair.sample_id,
        "clean_graph_id": pair.clean_graph_id,
        "distorted_graph_id": pair.distorted_graph_id,
        "distortion_id": pair.distortion_id,
        "metric_id": pair.metric_id,
        "born_zero_shift": pair.born_zero_shift,
        "born_observable_shift_absent": pair.born_observable_shift_absent,
        "marker_only": pair.marker_only,
        "applicability_warning": pair.applicability_warning,
        "identifiability_status": pair.identifiability_status,
        "identifiability_reason": pair.identifiability_reason,
        "diagnosis_supervision_mask": pair.diagnosis_supervision_mask,
        "observable_evidence_fingerprint": pair.observable_evidence_fingerprint,
        "metadata": pair.metadata,
        "content_hash": pair_content_hash(pair),
    }


def save_pair_artifact(pair: GraphSamplePair, path: str | Path) -> Path:
    if not pair.content_hash:
        pair.content_hash = pair_content_hash(pair)
    validate_pair_data(pair)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(pair, name) for name in PAIR_ARRAY_NAMES}
    arrays[PAIR_METADATA_ARRAY_NAME] = _json_bytes_array(_pair_metadata(pair))
    np.savez_compressed(target, **arrays)
    return target


def load_pair_artifact(
    path: str | Path,
    expected_content_hash: str | None = None,
) -> GraphSamplePair:
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        expected_names = set(PAIR_ARRAY_NAMES) | {PAIR_METADATA_ARRAY_NAME}
        _require_exact_npz_names(set(payload.files), expected_names, "pair artifact")
        metadata = _decode_json_bytes(
            payload[PAIR_METADATA_ARRAY_NAME],
            PAIR_METADATA_ARRAY_NAME,
        )
        arrays = {name: payload[name].copy() for name in PAIR_ARRAY_NAMES}
    pair_metadata_keys = {
        "artifact_schema_version",
        "graph_pair_id",
        "sample_id",
        "clean_graph_id",
        "distorted_graph_id",
        "distortion_id",
        "metric_id",
        "born_zero_shift",
        "born_observable_shift_absent",
        "marker_only",
        "applicability_warning",
        "identifiability_status",
        "identifiability_reason",
        "diagnosis_supervision_mask",
        "observable_evidence_fingerprint",
        "metadata",
        "content_hash",
    }
    _require_metadata_keys(metadata, pair_metadata_keys, "pair artifact")
    if metadata.get("artifact_schema_version") != PAIR_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported pair artifact schema version")
    if not isinstance(metadata["metadata"], Mapping):
        raise TypeError("pair artifact metadata.metadata must be a mapping")
    pair = GraphSamplePair(
        graph_pair_id=metadata["graph_pair_id"],
        sample_id=metadata["sample_id"],
        clean_graph_id=metadata["clean_graph_id"],
        distorted_graph_id=metadata["distorted_graph_id"],
        distortion_id=metadata["distortion_id"],
        metric_id=metadata["metric_id"],
        born_metric_names=arrays["born_metric_names"],
        born_metric_values=arrays["born_metric_values"],
        born_metric_positive_infinity_mask=arrays["born_metric_positive_infinity_mask"],
        measurement_setting_ids=arrays["measurement_setting_ids"],
        measurement_basis_codes=arrays["measurement_basis_codes"],
        measurement_outcome_bitstrings=arrays["measurement_outcome_bitstrings"],
        measurement_setting_index=arrays["measurement_setting_index"],
        clean_measurement_probabilities=arrays["clean_measurement_probabilities"],
        distorted_measurement_probabilities=arrays["distorted_measurement_probabilities"],
        born_zero_shift=metadata["born_zero_shift"],
        born_observable_shift_absent=metadata["born_observable_shift_absent"],
        marker_only=metadata["marker_only"],
        applicability_warning=metadata.get("applicability_warning"),
        identifiability_status=metadata["identifiability_status"],
        identifiability_reason=metadata.get("identifiability_reason"),
        diagnosis_supervision_mask=metadata["diagnosis_supervision_mask"],
        observable_evidence_fingerprint=metadata["observable_evidence_fingerprint"],
        metadata=dict(metadata["metadata"]),
        content_hash=metadata.get("content_hash", ""),
    )
    validate_pair_data(pair)
    actual_hash = pair_content_hash(pair)
    if pair.content_hash != actual_hash:
        raise ValueError("pair artifact stored content_hash mismatch")
    if expected_content_hash is not None and expected_content_hash != actual_hash:
        raise ValueError("pair artifact content_hash does not match manifest")
    return pair


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _validate_completion_marker(root: Path, expected: Mapping[str, Any]) -> None:
    from .utils import strict_json_load

    actual = strict_json_load(root / "graph_complete.json")
    if actual != dict(expected):
        raise ValueError("graph_complete.json content does not match graph dataset")
    for reference in actual["managed_files"]:
        resolve_safe_file(root, reference, f"graph_complete managed file {reference}")


def write_graph_dataset(
    result: GraphConversionResult,
    output_root: str | Path,
) -> GraphWriteResult:
    """Publish a validated graph dataset atomically into a fresh output root."""
    if not isinstance(result, GraphConversionResult):
        raise TypeError("result must be GraphConversionResult")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Graph output root already exists: {output}")
    verify_source_snapshot(result.source_root, result.source_snapshot)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected existing staging directory: {staging}")

    try:
        (staging / "manifests").mkdir(parents=True)
        (staging / "artifacts" / "graphs").mkdir(parents=True)
        (staging / "artifacts" / "pairs").mkdir(parents=True)

        managed: list[str] = []
        save_graph_config(result.config, staging / "graph_config.json")
        managed.append("graph_config.json")
        write_strict_json(staging / "graph_summary.json", result.summary)
        managed.append("graph_summary.json")

        for graph in result.graphs:
            reference = f"artifacts/graphs/{graph.graph_id}.npz"
            save_graph_artifact(graph, staging / reference)
            managed.append(reference)
        for pair in result.pairs:
            reference = f"artifacts/pairs/{pair.graph_pair_id}.npz"
            save_pair_artifact(pair, staging / reference)
            managed.append(reference)

        writer = ManifestWriter(staging / "manifests")
        writer.write_records("graph_manifest", result.graph_records, overwrite=False)
        managed.append("manifests/graph_manifest.parquet")
        writer.write_records("graph_pair_manifest", result.graph_pair_records, overwrite=False)
        managed.append("manifests/graph_pair_manifest.parquet")

        reader = ManifestReader(staging / "manifests")
        graph_records = reader.read_typed_records("graph_manifest", GraphRecord)
        pair_records = reader.read_typed_records("graph_pair_manifest", GraphPairRecord)
        loaded_graphs = {
            record.graph_id: load_graph_artifact(
                resolve_safe_file(
                    staging,
                    record.graph_ref,
                    f"GraphRecord {record.graph_id}.graph_ref",
                ),
                record.content_hash,
            )
            for record in graph_records
        }
        loaded_pairs = {
            record.graph_pair_id: load_pair_artifact(
                resolve_safe_file(
                    staging,
                    record.pair_ref,
                    f"GraphPairRecord {record.graph_pair_id}.pair_ref",
                ),
                record.content_hash,
            )
            for record in pair_records
        }
        validate_graph_dataset_joins(
            graph_records,
            pair_records,
            graphs_by_id=loaded_graphs,
            pairs_by_id=loaded_pairs,
            root=staging,
        )

        expected_before_marker = set(managed)
        actual_before_marker = _relative_file_set(staging)
        if actual_before_marker != expected_before_marker:
            raise ValueError(
                "Staging graph dataset contains unexpected or missing files; "
                f"missing={sorted(expected_before_marker - actual_before_marker)}, "
                f"unexpected={sorted(actual_before_marker - expected_before_marker)}"
            )

        managed_files = tuple(sorted([*managed, "graph_complete.json"]))
        if len(set(managed_files)) != len(managed_files):
            raise ValueError("managed graph file inventory contains duplicates")
        completion = {
            "complete": True,
            "source_scientific_generation_id": result.source_scientific_generation_id,
            "graph_conversion_id": result.graph_conversion_id,
            "operational_config_id": result.operational_config_id,
            "graph_schema_id": result.graph_schema_id,
            "graph_count": len(result.graphs),
            "pair_count": len(result.pairs),
            "source_snapshot_hash": result.source_snapshot.aggregate_sha256,
            "managed_files": list(managed_files),
        }
        write_strict_json(staging / "graph_complete.json", completion)
        if _relative_file_set(staging) != set(managed_files):
            raise ValueError("Committed graph file inventory does not match staging files")
        _validate_completion_marker(staging, completion)
        verify_source_snapshot(result.source_root, result.source_snapshot)

        if output.exists():
            raise FileExistsError(f"Graph output root appeared during publication: {output}")
        os.replace(staging, output)

        manifest_paths = tuple(
            sorted(
                (
                    output / "manifests" / "graph_manifest.parquet",
                    output / "manifests" / "graph_pair_manifest.parquet",
                ),
                key=lambda path: path.as_posix(),
            )
        )
        artifact_paths = tuple(
            sorted(
                [output / reference for reference in managed_files if reference.startswith("artifacts/")],
                key=lambda path: path.as_posix(),
            )
        )
        written_paths = tuple(
            sorted(
                [output / reference for reference in managed_files],
                key=lambda path: path.as_posix(),
            )
        )
        return GraphWriteResult(
            output_root=output,
            graph_complete_path=output / "graph_complete.json",
            manifest_paths=manifest_paths,
            artifact_paths=artifact_paths,
            written_paths=written_paths,
            managed_files=managed_files,
            graph_count=len(result.graphs),
            pair_count=len(result.pairs),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "graph_arrays",
    "load_graph_artifact",
    "load_pair_artifact",
    "save_graph_artifact",
    "save_pair_artifact",
    "write_graph_dataset",
]
