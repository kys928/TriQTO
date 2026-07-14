"""Integrity validation for Phase 8 graphs, pairs, and manifests."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np

from triqto.storage.graph_schema import GraphPairRecord, GraphRecord
from triqto.storage.schema import DatasetSampleRecord

from .constants import (
    EDGE_FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
)
from .evidence import (
    validate_born_metric_arrays,
    validate_count_arrays,
    validate_probability_arrays,
)
from .identities import graph_content_hash, graph_id, graph_pair_id, pair_content_hash
from .models import CircuitGraphData, GraphSamplePair
from .utils import require_nonblank, resolve_safe_file

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _array(value: Any, name: str, dtype: Any, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype.hasobject:
        raise TypeError(f"{name} must not use object dtype")
    if value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must use dtype {np.dtype(dtype)}, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    return value


def _unicode(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional ndarray")
    if value.dtype.hasobject or value.dtype.kind != "U":
        raise TypeError(f"{name} must use fixed-width Unicode dtype")
    return value


def _finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")


def _csr(pointer: Any, flat: np.ndarray, gate_count: int, name: str) -> None:
    ptr = _array(pointer, name, np.int64, 1)
    if ptr.shape != (gate_count + 1,):
        raise ValueError(f"{name} length must equal gate_count + 1")
    if int(ptr[0]) != 0 or np.any(np.diff(ptr) < 0):
        raise ValueError(f"{name} must start at zero and be monotonic")
    if int(ptr[-1]) != len(flat):
        raise ValueError(f"{name} final value must equal flattened-array length")


def validate_hash_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must have format sha256:<64 lowercase hex characters>")
    return value


def validate_graph_data(graph: CircuitGraphData) -> None:
    if not isinstance(graph, CircuitGraphData):
        raise TypeError("graph must be CircuitGraphData")
    if graph.graph_schema_version != GRAPH_SCHEMA_VERSION:
        raise ValueError(f"Unsupported graph schema version: {graph.graph_schema_version!r}")
    if graph.role not in {"clean", "distorted"}:
        raise ValueError("graph role must be clean or distorted")
    require_nonblank(graph.circuit_id, "circuit_id")
    require_nonblank(graph.source_run_id, "source_run_id")
    require_nonblank(graph.family, "family")
    for name in ("n_qubits", "n_clbits"):
        value = getattr(graph, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer and not bool")
    if graph.n_qubits <= 0 or graph.n_clbits < 0:
        raise ValueError("n_qubits must be positive and n_clbits nonnegative")
    expected_id = graph_id(graph.circuit_id, graph.source_run_id, graph.role)
    if graph.graph_id != expected_id:
        raise ValueError(f"graph_id mismatch: expected {expected_id}, got {graph.graph_id}")

    schemas = (
        (graph.node_feature_names, NODE_FEATURE_NAMES, "node_feature_names"),
        (graph.edge_feature_names, EDGE_FEATURE_NAMES, "edge_feature_names"),
        (graph.gate_feature_names, GATE_FEATURE_NAMES, "gate_feature_names"),
        (graph.global_feature_names, GLOBAL_FEATURE_NAMES, "global_feature_names"),
    )
    for actual, expected, name in schemas:
        if actual != expected:
            raise ValueError(f"{name} does not match Phase 8 v1 schema")

    node_index = _array(graph.node_index, "node_index", np.int64, 1)
    if not np.array_equal(node_index, np.arange(graph.n_qubits, dtype=np.int64)):
        raise ValueError("node_index must equal [0, ..., n_qubits - 1]")
    node_features = _array(graph.node_features, "node_features", np.float64, 2)
    if node_features.shape != (graph.n_qubits, len(NODE_FEATURE_NAMES)):
        raise ValueError("node_features shape does not match schema")
    _finite(node_features, "node_features")

    edge_index = _array(graph.edge_index, "edge_index", np.int64, 2)
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    edge_count = edge_index.shape[1]
    if edge_count and (int(edge_index.min()) < 0 or int(edge_index.max()) >= graph.n_qubits):
        raise ValueError("edge_index contains out-of-range qubit index")
    event_index = _array(graph.edge_event_index, "edge_event_index", np.int64, 1)
    if event_index.shape != (edge_count,):
        raise ValueError("edge_event_index length must equal edge count")
    edge_features = _array(graph.edge_features, "edge_features", np.float64, 2)
    if edge_features.shape != (edge_count, len(EDGE_FEATURE_NAMES)):
        raise ValueError("edge_features shape does not match schema")
    _finite(edge_features, "edge_features")

    gate_names = _unicode(graph.gate_names, "gate_names")
    gate_count = len(gate_names)
    gate_features = _array(graph.gate_features, "gate_features", np.float64, 2)
    if gate_features.shape != (gate_count, len(GATE_FEATURE_NAMES)):
        raise ValueError("gate_features shape does not match schema")
    _finite(gate_features, "gate_features")
    if event_index.size and (int(event_index.min()) < 0 or int(event_index.max()) >= gate_count):
        raise ValueError("edge_event_index contains out-of-range gate event")

    qindices = _array(graph.gate_qubit_indices, "gate_qubit_indices", np.int64, 1)
    cindices = _array(graph.gate_clbit_indices, "gate_clbit_indices", np.int64, 1)
    pvalues = _array(graph.gate_parameter_values, "gate_parameter_values", np.float64, 1)
    _csr(graph.gate_qubit_ptr, qindices, gate_count, "gate_qubit_ptr")
    _csr(graph.gate_clbit_ptr, cindices, gate_count, "gate_clbit_ptr")
    _csr(graph.gate_parameter_ptr, pvalues, gate_count, "gate_parameter_ptr")
    if qindices.size and (int(qindices.min()) < 0 or int(qindices.max()) >= graph.n_qubits):
        raise ValueError("gate_qubit_indices contains out-of-range index")
    if cindices.size and (
        graph.n_clbits == 0 or int(cindices.min()) < 0 or int(cindices.max()) >= graph.n_clbits
    ):
        raise ValueError("gate_clbit_indices contains out-of-range index")

    psin = _array(graph.gate_parameter_sin, "gate_parameter_sin", np.float64, 1)
    pcos = _array(graph.gate_parameter_cos, "gate_parameter_cos", np.float64, 1)
    pmask = _array(graph.gate_parameter_angle_mask, "gate_parameter_angle_mask", np.bool_, 1)
    if not (len(pvalues) == len(psin) == len(pcos) == len(pmask)):
        raise ValueError("gate parameter arrays must have matching lengths")
    for array, name in ((pvalues, "gate_parameter_values"), (psin, "gate_parameter_sin"), (pcos, "gate_parameter_cos")):
        _finite(array, name)
    if np.any(~pmask & ((psin != 0.0) | (pcos != 0.0))):
        raise ValueError("non-angular gate parameter phasor placeholders must be zero")

    names = _unicode(graph.parameter_names, "parameter_names")
    values = _array(graph.parameter_values, "parameter_values", np.float64, 1)
    sine = _array(graph.parameter_sin, "parameter_sin", np.float64, 1)
    cosine = _array(graph.parameter_cos, "parameter_cos", np.float64, 1)
    if not (len(names) == len(values) == len(sine) == len(cosine)):
        raise ValueError("global parameter arrays must have matching lengths")
    normalized_names = [str(value) for value in names.tolist()]
    if normalized_names != sorted(normalized_names) or len(set(normalized_names)) != len(normalized_names):
        raise ValueError("parameter_names must be sorted and unique")
    if any(not name for name in normalized_names):
        raise ValueError("parameter_names must be nonblank")
    for array, name in ((values, "parameter_values"), (sine, "parameter_sin"), (cosine, "parameter_cos")):
        _finite(array, name)

    basis_codes = _array(graph.measurement_basis_codes, "measurement_basis_codes", np.int64, 1)
    if basis_codes.shape != (graph.n_qubits,) or np.any((basis_codes < 0) | (basis_codes > 2)):
        raise ValueError("measurement_basis_codes must contain one Z/X/Y code per qubit")
    validate_probability_arrays(graph.outcome_bitstrings, graph.exact_probabilities, graph.n_qubits)
    global_features = _array(graph.global_features, "global_features", np.float64, 1)
    if global_features.shape != (0,):
        raise ValueError("Phase 8 v1 global_features must be empty")

    for name in ("exact_probability_available_mask", "supplemental_counts_available_mask", "hilbert_available_mask"):
        if not isinstance(getattr(graph, name), bool):
            raise TypeError(f"{name} must be bool")
    if graph.exact_probability_available_mask is not True or graph.hilbert_available_mask is not False:
        raise ValueError("exact probability must be available and Hilbert feature mask false")
    if graph.supplemental_counts_available_mask:
        if graph.supplemental_shots is None:
            raise ValueError("supplemental_shots is required when counts are available")
        validate_count_arrays(
            graph.count_outcome_bitstrings,
            graph.supplemental_counts,
            graph.n_qubits,
            graph.supplemental_shots,
        )
    else:
        if graph.supplemental_shots is not None:
            raise ValueError("supplemental_shots must be absent when counts are unavailable")
        if len(_unicode(graph.count_outcome_bitstrings, "count_outcome_bitstrings")):
            raise ValueError("count_outcome_bitstrings must be empty when unavailable")
        if len(_array(graph.supplemental_counts, "supplemental_counts", np.int64, 1)):
            raise ValueError("supplemental_counts must be empty when unavailable")

    if not isinstance(graph.source_sample_ids, tuple):
        raise TypeError("source_sample_ids must be a tuple")
    if graph.source_sample_ids != tuple(sorted(graph.source_sample_ids)):
        raise ValueError("source_sample_ids must be sorted")
    if len(set(graph.source_sample_ids)) != len(graph.source_sample_ids):
        raise ValueError("source_sample_ids must be unique")
    if any(not isinstance(value, str) or not value for value in graph.source_sample_ids):
        raise ValueError("source_sample_ids entries must be nonblank strings")
    if not isinstance(graph.scientific_metadata, Mapping) or not isinstance(graph.provenance_metadata, Mapping):
        raise TypeError("scientific_metadata and provenance_metadata must be mappings")


def validate_pair_data(pair: GraphSamplePair) -> None:
    if not isinstance(pair, GraphSamplePair):
        raise TypeError("pair must be GraphSamplePair")
    expected_id = graph_pair_id(pair.sample_id, pair.clean_graph_id, pair.distorted_graph_id)
    if pair.graph_pair_id != expected_id:
        raise ValueError(f"graph_pair_id mismatch: expected {expected_id}, got {pair.graph_pair_id}")
    for name in ("born_zero_shift", "born_observable_shift_absent", "marker_only", "diagnosis_supervision_mask", "action_supervision_mask", "born_target_mask"):
        if not isinstance(getattr(pair, name), bool):
            raise TypeError(f"{name} must be bool")
    if pair.identifiability_status not in {"identifiable", "conditionally_identifiable", "unidentifiable"}:
        raise ValueError("invalid identifiability_status")
    if not isinstance(pair.identifiability_reason, str) or not pair.identifiability_reason:
        raise ValueError("identifiability_reason must be nonblank")
    if pair.identifiability_status == "unidentifiable" and (pair.diagnosis_supervision_mask or pair.action_supervision_mask):
        raise ValueError("unidentifiable pairs must mask diagnosis/action supervision")
    if pair.applicability_warning is not None and not isinstance(pair.applicability_warning, str):
        raise TypeError("applicability_warning must be a string or None")
    if not isinstance(pair.metadata, Mapping):
        raise TypeError("pair metadata must be a mapping")
    validate_born_metric_arrays(
        pair.born_metric_names,
        pair.born_metric_values,
        pair.born_metric_positive_infinity_mask,
    )
    expected_hash = pair_content_hash(pair)
    if pair.content_hash and pair.content_hash != expected_hash:
        raise ValueError("pair content_hash does not match pair content")


def _index_unique(records: Sequence[Any], field_name: str, record_type: type) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, record_type):
            raise TypeError(f"Expected {record_type.__name__}, got {type(record).__name__}")
        record.validate()
        identifier = getattr(record, field_name)
        if identifier in result:
            raise ValueError(f"Duplicate {record_type.__name__} {field_name} {identifier}")
        result[identifier] = record
    return result


def validate_graph_dataset_joins(
    graph_records: Sequence[GraphRecord],
    pair_records: Sequence[GraphPairRecord],
    *,
    source_samples: Sequence[DatasetSampleRecord] | None = None,
    graphs_by_id: Mapping[str, CircuitGraphData] | None = None,
    pairs_by_id: Mapping[str, GraphSamplePair] | None = None,
    root: str | Path | None = None,
) -> None:
    graphs = _index_unique(graph_records, "graph_id", GraphRecord)
    pairs = _index_unique(pair_records, "graph_pair_id", GraphPairRecord)
    for record in graphs.values():
        expected = graph_id(record.circuit_id, record.source_run_id, record.role)
        if record.graph_id != expected:
            raise ValueError(f"GraphRecord {record.graph_id} identity mismatch; expected {expected}")
    for record in pairs.values():
        expected = graph_pair_id(record.sample_id, record.clean_graph_id, record.distorted_graph_id)
        if record.graph_pair_id != expected:
            raise ValueError(f"GraphPairRecord {record.graph_pair_id} identity mismatch; expected {expected}")
    samples = (
        _index_unique(source_samples, "sample_id", DatasetSampleRecord)
        if source_samples is not None
        else {}
    )

    for record in pairs.values():
        clean = graphs.get(record.clean_graph_id)
        distorted = graphs.get(record.distorted_graph_id)
        if clean is None:
            raise ValueError(f"GraphPairRecord {record.graph_pair_id} references missing clean GraphRecord {record.clean_graph_id}")
        if distorted is None:
            raise ValueError(f"GraphPairRecord {record.graph_pair_id} references missing distorted GraphRecord {record.distorted_graph_id}")
        if clean.role != "clean" or distorted.role != "distorted":
            raise ValueError(f"GraphPairRecord {record.graph_pair_id} graph roles are invalid")
        if samples:
            sample = samples.get(record.sample_id)
            if sample is None:
                raise ValueError(f"GraphPairRecord {record.graph_pair_id} references missing DatasetSampleRecord {record.sample_id}")
            expected_fields = {
                "distortion_id": sample.distortion_id,
                "metric_id": sample.metric_id,
            }
            for name, expected in expected_fields.items():
                if getattr(record, name) != expected:
                    raise ValueError(f"GraphPairRecord {record.graph_pair_id} {name} mismatch")
            graph_links = (
                (clean, sample.clean_circuit_id, sample.clean_run_id),
                (distorted, sample.distorted_circuit_id, sample.distorted_run_id),
            )
            for graph_record, circuit_id, run_id in graph_links:
                if graph_record.circuit_id != circuit_id or graph_record.source_run_id != run_id:
                    raise ValueError(f"GraphPairRecord {record.graph_pair_id} source graph join mismatch")
                if graph_record.family != sample.family:
                    raise ValueError(f"GraphPairRecord {record.graph_pair_id} family mismatch")

        if graphs_by_id is not None:
            for graph_record in (clean, distorted):
                graph = graphs_by_id.get(graph_record.graph_id)
                if graph is None:
                    raise ValueError(f"GraphRecord {graph_record.graph_id} has no loaded graph artifact")
                validate_graph_data(graph)
                checks = {
                    "content_hash": graph_content_hash(graph),
                    "circuit_id": graph.circuit_id,
                    "source_run_id": graph.source_run_id,
                    "role": graph.role,
                    "family": graph.family,
                    "graph_schema_version": graph.graph_schema_version,
                    "n_nodes": graph.n_qubits,
                    "n_edges": graph.edge_index.shape[1],
                    "n_gate_events": graph.gate_features.shape[0],
                    "node_feature_dim": graph.node_features.shape[1],
                    "edge_feature_dim": graph.edge_features.shape[1],
                    "gate_feature_dim": graph.gate_features.shape[1],
                }
                for name, expected in checks.items():
                    if getattr(graph_record, name) != expected:
                        raise ValueError(f"GraphRecord {graph_record.graph_id} {name} mismatch")
        if pairs_by_id is not None:
            pair = pairs_by_id.get(record.graph_pair_id)
            if pair is None:
                raise ValueError(f"GraphPairRecord {record.graph_pair_id} has no loaded pair artifact")
            validate_pair_data(pair)
            fields = ("content_hash", "sample_id", "clean_graph_id", "distorted_graph_id", "distortion_id", "metric_id")
            for name in fields:
                if getattr(pair, name) != getattr(record, name):
                    raise ValueError(f"GraphPairRecord {record.graph_pair_id} {name} mismatch")
        if root is not None:
            resolve_safe_file(root, clean.graph_ref, f"GraphRecord {clean.graph_id}.graph_ref")
            resolve_safe_file(root, distorted.graph_ref, f"GraphRecord {distorted.graph_id}.graph_ref")
            resolve_safe_file(root, record.pair_ref, f"GraphPairRecord {record.graph_pair_id}.pair_ref")


__all__ = [
    "validate_graph_data",
    "validate_graph_dataset_joins",
    "validate_hash_string",
    "validate_pair_data",
]
