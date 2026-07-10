from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from triqto.graph import (
    GraphSamplePair,
    circuit_to_graph,
    graph_content_hash,
    graph_pair_id,
    load_graph_artifact,
    load_pair_artifact,
    pair_content_hash,
    save_graph_artifact,
    save_pair_artifact,
)
from triqto.storage import GraphPairRecord, GraphRecord


def make_graph():
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    return circuit_to_graph(
        circuit,
        circuit_id="circuit_a",
        source_run_id="run_a",
        role="clean",
        family="unit",
        parameter_bindings={},
        exact_probabilities={"00": 0.5, "11": 0.5},
        source_sample_ids=["sample_a"],
    )


def make_pair(graph_id_value: str):
    pair = GraphSamplePair(
        graph_pair_id=graph_pair_id("sample_a", graph_id_value, "graph_distorted"),
        sample_id="sample_a",
        clean_graph_id=graph_id_value,
        distorted_graph_id="graph_distorted",
        distortion_id="distortion_a",
        metric_id="metric_a",
        born_metric_names=np.asarray(["total_variation"], dtype="<U15"),
        born_metric_values=np.asarray([0.2], dtype=np.float64),
        born_metric_positive_infinity_mask=np.asarray([False], dtype=np.bool_),
        born_zero_shift=False,
        born_observable_shift_absent=False,
        marker_only=False,
        applicability_warning=None,
        metadata={"phase": 8},
    )
    pair.content_hash = pair_content_hash(pair)
    return pair


def test_graph_artifact_roundtrip_and_hash(tmp_path):
    graph = make_graph()
    path = tmp_path / "graph.npz"
    save_graph_artifact(graph, path)
    loaded = load_graph_artifact(path, graph_content_hash(graph))
    assert loaded.graph_id == graph.graph_id
    assert np.array_equal(loaded.edge_index, graph.edge_index)
    with np.load(path, allow_pickle=False) as payload:
        assert all(not payload[name].dtype.hasobject for name in payload.files)


def test_graph_artifact_missing_array_rejected(tmp_path):
    graph = make_graph()
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, node_index=graph.node_index)
    with pytest.raises(ValueError, match="array-name mismatch"):
        load_graph_artifact(path)


def test_pair_artifact_roundtrip_and_hash(tmp_path):
    graph = make_graph()
    pair = make_pair(graph.graph_id)
    path = tmp_path / "pair.npz"
    save_pair_artifact(pair, path)
    loaded = load_pair_artifact(path, pair.content_hash)
    assert loaded.graph_pair_id == pair.graph_pair_id
    assert loaded.content_hash == pair.content_hash


def test_graph_and_pair_manifest_records_are_strict():
    graph = make_graph()
    content_hash = graph_content_hash(graph)
    GraphRecord(
        graph_id=graph.graph_id,
        circuit_id=graph.circuit_id,
        source_run_id=graph.source_run_id,
        role="clean",
        family="unit",
        graph_schema_version=graph.graph_schema_version,
        graph_ref=f"artifacts/graphs/{graph.graph_id}.npz",
        content_hash=content_hash,
        n_nodes=2,
        n_edges=2,
        n_gate_events=3,
        node_feature_dim=graph.node_features.shape[1],
        edge_feature_dim=graph.edge_features.shape[1],
        gate_feature_dim=graph.gate_features.shape[1],
        metadata={},
    ).validate()
    pair = make_pair(graph.graph_id)
    GraphPairRecord(
        graph_pair_id=pair.graph_pair_id,
        sample_id=pair.sample_id,
        clean_graph_id=pair.clean_graph_id,
        distorted_graph_id=pair.distorted_graph_id,
        distortion_id=pair.distortion_id,
        metric_id=pair.metric_id,
        pair_ref=f"artifacts/pairs/{pair.graph_pair_id}.npz",
        content_hash=pair.content_hash,
        metadata={},
    ).validate()
    with pytest.raises(ValueError):
        GraphPairRecord(
            graph_pair_id=pair.graph_pair_id,
            sample_id=pair.sample_id,
            clean_graph_id=pair.clean_graph_id,
            distorted_graph_id=pair.distorted_graph_id,
            distortion_id=pair.distortion_id,
            metric_id=pair.metric_id,
            pair_ref="../escape.npz",
            content_hash=pair.content_hash,
            metadata={},
        ).validate()
