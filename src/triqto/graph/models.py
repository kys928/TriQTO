"""Framework-neutral in-memory records for Phase 8 graph conversion."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from triqto.data_generation import DatasetGenerationConfig
from triqto.storage.schema import (
    CircuitRecord,
    DatasetSampleRecord,
    DistortionRecord,
    MetricRecord,
    SimulationRecord,
)

from .config import GraphConversionConfig
from .constants import (
    EDGE_FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)


@dataclass(frozen=True, slots=True)
class SourceFileEntry:
    reference: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    entries: tuple[SourceFileEntry, ...]
    aggregate_sha256: str


@dataclass(slots=True)
class CircuitGraphData:
    graph_id: str
    circuit_id: str
    source_run_id: str
    role: str
    family: str
    graph_schema_version: str
    n_qubits: int
    n_clbits: int
    node_index: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_event_index: np.ndarray
    edge_features: np.ndarray
    gate_names: np.ndarray
    gate_features: np.ndarray
    gate_qubit_ptr: np.ndarray
    gate_qubit_indices: np.ndarray
    gate_clbit_ptr: np.ndarray
    gate_clbit_indices: np.ndarray
    gate_parameter_ptr: np.ndarray
    gate_parameter_values: np.ndarray
    gate_parameter_sin: np.ndarray
    gate_parameter_cos: np.ndarray
    gate_parameter_angle_mask: np.ndarray
    parameter_names: np.ndarray
    parameter_values: np.ndarray
    parameter_sin: np.ndarray
    parameter_cos: np.ndarray
    outcome_bitstrings: np.ndarray
    exact_probabilities: np.ndarray
    global_features: np.ndarray
    count_outcome_bitstrings: np.ndarray
    supplemental_counts: np.ndarray
    source_sample_ids: tuple[str, ...] = ()
    node_feature_names: tuple[str, ...] = NODE_FEATURE_NAMES
    edge_feature_names: tuple[str, ...] = EDGE_FEATURE_NAMES
    gate_feature_names: tuple[str, ...] = GATE_FEATURE_NAMES
    global_feature_names: tuple[str, ...] = GLOBAL_FEATURE_NAMES
    exact_probability_available_mask: bool = True
    supplemental_counts_available_mask: bool = False
    hilbert_available_mask: bool = False
    supplemental_shots: int | None = None
    scientific_metadata: dict[str, Any] = field(default_factory=dict)
    provenance_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphSamplePair:
    graph_pair_id: str
    sample_id: str
    clean_graph_id: str
    distorted_graph_id: str
    distortion_id: str
    metric_id: str
    born_metric_names: np.ndarray
    born_metric_values: np.ndarray
    born_metric_positive_infinity_mask: np.ndarray
    born_zero_shift: bool
    born_observable_shift_absent: bool
    marker_only: bool
    applicability_warning: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(slots=True)
class CompletedPhase7Dataset:
    source_root: Path
    generation_config: DatasetGenerationConfig
    generation_config_payload: dict[str, Any]
    source_scientific_generation_id: str
    source_config_id: str
    samples: list[DatasetSampleRecord]
    circuits: list[CircuitRecord]
    simulations: list[SimulationRecord]
    distortions: list[DistortionRecord]
    metrics: list[MetricRecord]
    circuits_by_id: dict[str, QuantumCircuit]
    probabilities_by_run_id: dict[str, dict[str, float]]
    counts_by_exact_run_id: dict[str, dict[str, int]]
    shot_records_by_exact_run_id: dict[str, SimulationRecord]
    statevector_storage_enabled: bool
    completion_marker: dict[str, Any]
    managed_files: tuple[str, ...]
    source_snapshot: SourceFileSnapshot


@dataclass(slots=True)
class GraphConversionResult:
    source_root: Path
    config: GraphConversionConfig
    source_scientific_generation_id: str
    graph_conversion_id: str
    operational_config_id: str
    graph_schema_id: str
    graphs: list[CircuitGraphData]
    pairs: list[GraphSamplePair]
    graph_records: list[Any]
    graph_pair_records: list[Any]
    source_snapshot: SourceFileSnapshot
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GraphWriteResult:
    output_root: Path
    graph_complete_path: Path
    manifest_paths: tuple[Path, ...]
    artifact_paths: tuple[Path, ...]
    written_paths: tuple[Path, ...]
    managed_files: tuple[str, ...]
    graph_count: int
    pair_count: int


__all__ = [
    "CircuitGraphData",
    "CompletedPhase7Dataset",
    "GraphConversionResult",
    "GraphSamplePair",
    "GraphWriteResult",
    "SourceFileEntry",
    "SourceFileSnapshot",
]
