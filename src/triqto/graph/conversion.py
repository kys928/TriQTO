"""Dataset-level orchestration for deterministic Phase 8 graph conversion."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from triqto.storage.graph_schema import GraphPairRecord, GraphRecord

from .circuit_graph import circuit_to_graph
from .config import GraphConversionConfig
from .evidence import decode_born_metric_arrays
from .identities import (
    graph_content_hash,
    graph_conversion_id,
    graph_operational_config_id,
    graph_pair_id,
    graph_schema_id,
    pair_content_hash,
)
from .models import GraphConversionResult, GraphSamplePair
from .source import load_completed_phase7_dataset, verify_source_snapshot
from .utils import require_mapping
from .validation import (
    validate_graph_data,
    validate_graph_dataset_joins,
    validate_pair_data,
)


def _index_unique(records: list[Any], field_name: str, record_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        identifier = getattr(record, field_name, None)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{record_name}.{field_name} must be nonblank")
        if identifier in result:
            raise ValueError(f"Duplicate {record_name} {field_name} {identifier}")
        result[identifier] = record
    return result


def _marker_only(sample_metadata: Mapping[str, Any], distortion_metadata: Mapping[str, Any]) -> bool:
    sample_value = sample_metadata.get("marker_only", False)
    distortion_value = distortion_metadata.get("marker_only")
    if not isinstance(sample_value, bool):
        raise TypeError("DatasetSampleRecord metadata.marker_only must be bool")
    # Parquet can materialize an absent key in heterogeneous metadata as null.
    # The sample record remains authoritative when the distortion did not declare it.
    if distortion_value is None:
        return sample_value
    if not isinstance(distortion_value, bool):
        raise TypeError("DistortionRecord metadata.marker_only must be bool or null")
    if sample_value != distortion_value:
        raise ValueError("Sample and distortion marker_only metadata disagree")
    return sample_value


def _required_bool(metadata: Mapping[str, Any], name: str, record_id: str) -> bool:
    value = metadata.get(name)
    if not isinstance(value, bool):
        raise TypeError(f"Sample {record_id} metadata.{name} must be bool")
    return value


def convert_completed_dataset_to_graphs(
    source_root: str | Path,
    config: GraphConversionConfig | None = None,
) -> GraphConversionResult:
    """Convert every Phase 7 sample into reusable graph and pair records."""
    conversion_config = config or GraphConversionConfig()
    dataset = load_completed_phase7_dataset(source_root)

    circuit_records = _index_unique(dataset.circuits, "circuit_id", "CircuitRecord")
    simulation_records = _index_unique(dataset.simulations, "run_id", "SimulationRecord")
    distortion_records = _index_unique(
        dataset.distortions,
        "distortion_id",
        "DistortionRecord",
    )
    metric_records = _index_unique(dataset.metrics, "metric_id", "MetricRecord")

    usage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for sample in dataset.samples:
        usage[(sample.clean_circuit_id, sample.clean_run_id, "clean")].add(sample.sample_id)
        usage[(sample.distorted_circuit_id, sample.distorted_run_id, "distorted")].add(
            sample.sample_id
        )

    graph_cache: dict[tuple[str, str, str], Any] = {}
    graphs = []
    pairs: list[GraphSamplePair] = []

    for sample in sorted(dataset.samples, key=lambda item: item.sample_id):
        sample_metadata = require_mapping(
            sample.metadata,
            f"DatasetSampleRecord {sample.sample_id}.metadata",
        )
        graph_keys = (
            (sample.clean_circuit_id, sample.clean_run_id, "clean"),
            (sample.distorted_circuit_id, sample.distorted_run_id, "distorted"),
        )
        sample_graphs = []
        for circuit_id, run_id, role in graph_keys:
            key = (circuit_id, run_id, role)
            if key not in graph_cache:
                circuit_record = circuit_records[circuit_id]
                simulation_record = simulation_records[run_id]
                if simulation_record.simulation_mode != "ideal_statevector":
                    raise ValueError(
                        f"Graph source run {run_id} must be ideal_statevector"
                    )
                if simulation_record.circuit_id != circuit_id:
                    raise ValueError(
                        f"Graph source run {run_id} circuit_id mismatch"
                    )
                shot_record = dataset.shot_records_by_exact_run_id.get(run_id)
                counts = None
                shots = None
                source_counts_ref = None
                source_shot_run_id = None
                if conversion_config.include_supplemental_counts and shot_record is not None:
                    counts = dataset.counts_by_exact_run_id[run_id]
                    shots = shot_record.shots
                    source_counts_ref = shot_record.counts_ref
                    source_shot_run_id = shot_record.run_id

                provenance = {
                    "source_circuit_ref": circuit_record.metadata.get("artifact_ref"),
                    "source_probability_ref": simulation_record.probabilities_ref,
                    "source_statevector_ref": simulation_record.statevector_ref,
                    "source_counts_ref": source_counts_ref,
                    "source_shot_run_id": source_shot_run_id,
                    "source_sample_ids": sorted(usage[key]),
                }
                graph = circuit_to_graph(
                    dataset.circuits_by_id[circuit_id],
                    circuit_id=circuit_id,
                    source_run_id=run_id,
                    role=role,
                    family=sample.family,
                    parameter_bindings=sample.parameter_bindings,
                    exact_probabilities=dataset.probabilities_by_run_id[run_id],
                    source_sample_ids=sorted(usage[key]),
                    supplemental_counts=counts,
                    supplemental_shots=shots,
                    scientific_metadata={
                        "exact_probability_source": "ideal_statevector",
                    },
                    provenance_metadata=provenance,
                    config=conversion_config,
                )
                validate_graph_data(graph)
                graph_cache[key] = graph
                graphs.append(graph)
            sample_graphs.append(graph_cache[key])

        clean_graph, distorted_graph = sample_graphs
        metric_record = metric_records[sample.metric_id]
        distortion_record = distortion_records[sample.distortion_id]
        metric_metadata = require_mapping(
            metric_record.metadata,
            f"MetricRecord {metric_record.metric_id}.metadata",
        )
        distortion_metadata = require_mapping(
            distortion_record.metadata,
            f"DistortionRecord {distortion_record.distortion_id}.metadata",
        )
        metric_names, metric_values, infinity_mask = decode_born_metric_arrays(
            metric_record.born_metrics
        )
        warning = metric_metadata.get("applicability_warning")
        if warning is not None and not isinstance(warning, str):
            raise TypeError(
                f"MetricRecord {metric_record.metric_id} applicability_warning "
                "must be string or None"
            )
        pair = GraphSamplePair(
            graph_pair_id=graph_pair_id(
                sample.sample_id,
                clean_graph.graph_id,
                distorted_graph.graph_id,
            ),
            sample_id=sample.sample_id,
            clean_graph_id=clean_graph.graph_id,
            distorted_graph_id=distorted_graph.graph_id,
            distortion_id=sample.distortion_id,
            metric_id=sample.metric_id,
            born_metric_names=metric_names,
            born_metric_values=metric_values,
            born_metric_positive_infinity_mask=infinity_mask,
            born_zero_shift=_required_bool(
                sample_metadata,
                "born_zero_shift",
                sample.sample_id,
            ),
            born_observable_shift_absent=_required_bool(
                sample_metadata,
                "born_observable_shift_absent",
                sample.sample_id,
            ),
            marker_only=_marker_only(sample_metadata, distortion_metadata),
            applicability_warning=warning,
            metadata={
                "distortion_type": distortion_record.distortion_type,
                "phase": 8,
            },
        )
        pair.content_hash = pair_content_hash(pair)
        validate_pair_data(pair)
        pairs.append(pair)

    graphs = sorted(graphs, key=lambda item: item.graph_id)
    pairs = sorted(pairs, key=lambda item: item.graph_pair_id)
    graph_records = []
    for graph in graphs:
        content_hash = graph_content_hash(graph)
        graph_records.append(
            GraphRecord(
                graph_id=graph.graph_id,
                circuit_id=graph.circuit_id,
                source_run_id=graph.source_run_id,
                role=graph.role,
                family=graph.family,
                graph_schema_version=graph.graph_schema_version,
                graph_ref=f"artifacts/graphs/{graph.graph_id}.npz",
                content_hash=content_hash,
                n_nodes=graph.n_qubits,
                n_edges=graph.edge_index.shape[1],
                n_gate_events=graph.gate_features.shape[0],
                node_feature_dim=graph.node_features.shape[1],
                edge_feature_dim=graph.edge_features.shape[1],
                gate_feature_dim=graph.gate_features.shape[1],
                metadata={
                    "source_sample_ids": list(graph.source_sample_ids),
                    "supplemental_counts_available": (
                        graph.supplemental_counts_available_mask
                    ),
                    "hilbert_available": False,
                    "phase": 8,
                },
            )
        )
    pair_records = [
        GraphPairRecord(
            graph_pair_id=pair.graph_pair_id,
            sample_id=pair.sample_id,
            clean_graph_id=pair.clean_graph_id,
            distorted_graph_id=pair.distorted_graph_id,
            distortion_id=pair.distortion_id,
            metric_id=pair.metric_id,
            pair_ref=f"artifacts/pairs/{pair.graph_pair_id}.npz",
            content_hash=pair.content_hash,
            metadata={
                "marker_only": pair.marker_only,
                "born_zero_shift": pair.born_zero_shift,
                "phase": 8,
            },
        )
        for pair in pairs
    ]

    graphs_by_id = {graph.graph_id: graph for graph in graphs}
    pairs_by_id = {pair.graph_pair_id: pair for pair in pairs}
    validate_graph_dataset_joins(
        graph_records,
        pair_records,
        source_samples=dataset.samples,
        graphs_by_id=graphs_by_id,
        pairs_by_id=pairs_by_id,
    )

    schema_id = graph_schema_id()
    conversion_id = graph_conversion_id(dataset.source_scientific_generation_id)
    operational_id = graph_operational_config_id(conversion_config)
    family_counts = Counter(sample.family for sample in dataset.samples)
    distortion_counts = Counter(
        distortion_records[sample.distortion_id].distortion_type
        for sample in dataset.samples
    )
    qubit_counts = Counter(str(sample.n_qubits) for sample in dataset.samples)
    verify_source_snapshot(dataset.source_root, dataset.source_snapshot)
    summary = {
        "source_scientific_generation_id": dataset.source_scientific_generation_id,
        "graph_conversion_id": conversion_id,
        "operational_config_id": operational_id,
        "graph_schema_id": schema_id,
        "source_sample_count": len(dataset.samples),
        "graph_count": len(graphs),
        "pair_count": len(pairs),
        "clean_graph_count": sum(graph.role == "clean" for graph in graphs),
        "distorted_graph_count": sum(
            graph.role == "distorted" for graph in graphs
        ),
        "family_counts": dict(sorted(family_counts.items())),
        "distortion_counts": dict(sorted(distortion_counts.items())),
        "marker_only_pair_count": sum(pair.marker_only for pair in pairs),
        "born_zero_shift_pair_count": sum(pair.born_zero_shift for pair in pairs),
        "variable_qubit_count_distribution": dict(sorted(qubit_counts.items())),
        "total_nodes": sum(graph.n_qubits for graph in graphs),
        "total_directed_edges": sum(graph.edge_index.shape[1] for graph in graphs),
        "total_gate_events": sum(graph.gate_features.shape[0] for graph in graphs),
        "multi_qubit_event_count": sum(
            int(graph.scientific_metadata.get("multi_qubit_event_count", 0))
            for graph in graphs
        ),
        "supplemental_count_graph_count": sum(
            graph.supplemental_counts_available_mask for graph in graphs
        ),
        "source_managed_file_count": len(dataset.source_snapshot.entries),
        "source_snapshot_hash": dataset.source_snapshot.aggregate_sha256,
        "source_immutability_verified": True,
        "schema_versions": {
            "graph": conversion_config.schema_version,
        },
    }
    return GraphConversionResult(
        source_root=dataset.source_root,
        config=conversion_config,
        source_scientific_generation_id=dataset.source_scientific_generation_id,
        graph_conversion_id=conversion_id,
        operational_config_id=operational_id,
        graph_schema_id=schema_id,
        graphs=graphs,
        pairs=pairs,
        graph_records=graph_records,
        graph_pair_records=pair_records,
        source_snapshot=dataset.source_snapshot,
        summary=summary,
    )


__all__ = ["convert_completed_dataset_to_graphs"]
