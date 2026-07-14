"""Deterministic scientific identities and logical content hashes for Phase 8."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

import numpy as np

from triqto.core.ids import canonical_json, make_deterministic_id

from .config import GraphConversionConfig, graph_config_to_dict
from .constants import (
    ANGLE_SLOT_MAPPING_VERSION,
    EDGE_FEATURE_NAMES,
    EDGE_REPRESENTATION_VERSION,
    GATE_FEATURE_NAMES,
    GATE_VOCAB_VERSION,
    GLOBAL_FEATURE_NAMES,
    GRAPH_CORE_ARRAY_NAMES,
    GRAPH_SCHEMA_VERSION,
    LOGICAL_LAYER_ALGORITHM_VERSION,
    NODE_FEATURE_NAMES,
    PAIR_ARTIFACT_SCHEMA_VERSION,
)
from .models import CircuitGraphData, GraphSamplePair
from .utils import json_copy, require_nonblank


def graph_schema_id() -> str:
    return make_deterministic_id(
        "graphschema",
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "node_feature_names": NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "gate_feature_names": GATE_FEATURE_NAMES,
            "global_feature_names": GLOBAL_FEATURE_NAMES,
            "gate_vocab_version": GATE_VOCAB_VERSION,
            "angle_slot_mapping_version": ANGLE_SLOT_MAPPING_VERSION,
            "edge_representation_version": EDGE_REPRESENTATION_VERSION,
            "logical_layer_algorithm_version": LOGICAL_LAYER_ALGORITHM_VERSION,
            "pair_artifact_schema_version": PAIR_ARTIFACT_SCHEMA_VERSION,
        },
    )


def graph_id(circuit_id: str, source_run_id: str, role: str) -> str:
    circuit = require_nonblank(circuit_id, "circuit_id")
    run = require_nonblank(source_run_id, "source_run_id")
    if role not in {"clean", "distorted"}:
        raise ValueError("role must be clean or distorted")
    return make_deterministic_id(
        "graph",
        {
            "circuit_id": circuit,
            "source_run_id": run,
            "role": role,
            "graph_schema_id": graph_schema_id(),
        },
    )


def graph_pair_id(
    sample_id: str,
    clean_graph_id: str,
    distorted_graph_id: str,
) -> str:
    return make_deterministic_id(
        "graphpair",
        {
            "sample_id": require_nonblank(sample_id, "sample_id"),
            "clean_graph_id": require_nonblank(clean_graph_id, "clean_graph_id"),
            "distorted_graph_id": require_nonblank(
                distorted_graph_id,
                "distorted_graph_id",
            ),
            "graph_schema_id": graph_schema_id(),
        },
    )


def graph_conversion_id(source_scientific_generation_id: str) -> str:
    return make_deterministic_id(
        "graphconv",
        {
            "source_scientific_generation_id": require_nonblank(
                source_scientific_generation_id,
                "source_scientific_generation_id",
            ),
            "graph_schema_id": graph_schema_id(),
        },
    )


def graph_operational_config_id(config: GraphConversionConfig) -> str:
    return make_deterministic_id("graphconfig", graph_config_to_dict(config))


def _update_array_hash(hasher: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(canonical_json(list(contiguous.shape)).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(contiguous.tobytes(order="C"))
    hasher.update(b"\0")


def hash_logical_content(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(canonical_json(json_copy(dict(metadata))).encode("utf-8"))
    hasher.update(b"\0")
    for name in sorted(arrays):
        _update_array_hash(hasher, name, arrays[name])
    return f"sha256:{hasher.hexdigest()}"


def graph_scientific_arrays(graph: CircuitGraphData) -> dict[str, np.ndarray]:
    return {name: getattr(graph, name) for name in GRAPH_CORE_ARRAY_NAMES}


def graph_scientific_hash_metadata(graph: CircuitGraphData) -> dict[str, Any]:
    return {
        "graph_schema_id": graph_schema_id(),
        "graph_schema_version": graph.graph_schema_version,
        "n_qubits": graph.n_qubits,
        "n_clbits": graph.n_clbits,
        "node_feature_names": list(graph.node_feature_names),
        "edge_feature_names": list(graph.edge_feature_names),
        "gate_feature_names": list(graph.gate_feature_names),
        "global_feature_names": list(graph.global_feature_names),
        "logical_layer_algorithm_version": LOGICAL_LAYER_ALGORITHM_VERSION,
        "scientific_metadata": graph.scientific_metadata,
    }


def graph_content_hash(graph: CircuitGraphData) -> str:
    """Hash represented scientific content, excluding IDs and provenance paths."""
    return hash_logical_content(
        graph_scientific_arrays(graph),
        graph_scientific_hash_metadata(graph),
    )


def pair_content_hash(pair: GraphSamplePair) -> str:
    arrays = {
        "born_metric_names": pair.born_metric_names,
        "born_metric_values": pair.born_metric_values,
        "born_metric_positive_infinity_mask": (
            pair.born_metric_positive_infinity_mask
        ),
    }
    metadata = {
        "pair_artifact_schema_version": PAIR_ARTIFACT_SCHEMA_VERSION,
        "sample_id": pair.sample_id,
        "clean_graph_id": pair.clean_graph_id,
        "distorted_graph_id": pair.distorted_graph_id,
        "distortion_id": pair.distortion_id,
        "metric_id": pair.metric_id,
        "born_zero_shift": pair.born_zero_shift,
        "born_observable_shift_absent": pair.born_observable_shift_absent,
        "marker_only": pair.marker_only,
        "identifiability_status": pair.identifiability_status,
        "identifiability_reason": pair.identifiability_reason,
        "diagnosis_supervision_mask": pair.diagnosis_supervision_mask,
        "action_supervision_mask": pair.action_supervision_mask,
        "born_target_mask": pair.born_target_mask,
        "applicability_warning": pair.applicability_warning,
        "metadata": pair.metadata,
    }
    return hash_logical_content(arrays, metadata)


__all__ = [
    "graph_content_hash",
    "graph_conversion_id",
    "graph_id",
    "graph_operational_config_id",
    "graph_pair_id",
    "graph_schema_id",
    "hash_logical_content",
    "pair_content_hash",
]
