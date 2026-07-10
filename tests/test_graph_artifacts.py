import json
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate

from triqto.graph import (
    GraphConversionConfig, circuit_to_graph, content_hash, graph_arrays,
    graph_config_from_dict, graph_config_to_dict, load_graph_artifact,
    save_graph_artifact, validate_graph_data,
)
from triqto.storage import GraphPairRecord, GraphRecord


def probs(n):
    return {format(i, f"0{n}b"): 1.0/(2**n) for i in range(2**n)}


def test_config_strict_roundtrip_and_rejections(tmp_path):
    cfg = GraphConversionConfig()
    assert graph_config_from_dict(graph_config_to_dict(cfg)) == cfg
    with pytest.raises(ValueError): graph_config_from_dict({**graph_config_to_dict(cfg), "split":"train"})
    with pytest.raises(TypeError): GraphConversionConfig(max_gate_events=True)
    with pytest.raises(ValueError): GraphConversionConfig(max_gate_events=0)
    with pytest.raises(TypeError): GraphConversionConfig(include_supplemental_counts=1)


def test_basic_multigraph_measurement_reset_barrier_layers_and_params(tmp_path):
    qc = QuantumCircuit(4, 2)
    qc.h(0); qc.x(2); qc.cx(0, 1); qc.cx(0, 1); qc.swap(1, 2); qc.reset(3); qc.barrier(); qc.measure(0, 0); qc.measure(1, 1)
    g = circuit_to_graph(qc, circuit_id="c1", sample_id="s1", role="clean", family="unit", parameter_bindings={"b":2.0,"a":1.0}, exact_probabilities=probs(4), metadata={})
    assert g.n_qubits == 4
    assert g.node_features.shape[0] == 4
    assert g.edge_index.shape == (2, 6)
    assert np.array_equal(g.edge_event_index, np.array([2,2,3,3,4,4]))
    assert g.parameter_names.tolist() == ["a", "b"]
    assert g.gate_names.tolist()[6:] == ["barrier", "measure", "measure"]
    validate_graph_data(g)
    p = tmp_path / "g.npz"
    save_graph_artifact(g, p)
    ch = content_hash(graph_arrays(g), load_graph_artifact(p).metadata | {"dummy":"ignored"})
    assert load_graph_artifact(p).graph_id == g.graph_id


def test_angular_and_non_angular_parameters_global_phase_and_no_mutation():
    qc = QuantumCircuit(1)
    qc.global_phase = 0.5
    qc.rz(math.pi/7, 0)
    qc.append(RXGate(math.pi/5).power(2), [0])
    before = qc.copy()
    g = circuit_to_graph(qc, circuit_id="c2", sample_id="s2", role="distorted", family="unit", parameter_bindings={}, exact_probabilities={"0":1.0,"1":0.0}, metadata={})
    assert g.gate_parameter_angle_mask[0]
    assert g.metadata["global_phase_excluded_from_features"] is True
    assert g.global_features.shape == (0,)
    assert qc == before


def test_probability_validation_rejects_bad_width_and_negative():
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError): circuit_to_graph(qc, circuit_id="c", sample_id="s", role="clean", family="f", parameter_bindings={}, exact_probabilities={"0":1.0}, metadata={})
    with pytest.raises(ValueError): circuit_to_graph(qc, circuit_id="c", sample_id="s", role="clean", family="f", parameter_bindings={}, exact_probabilities={"00":1.1,"11":-0.1}, metadata={})


def test_records_validate_and_refs_are_safe():
    GraphRecord("g","s","c","clean","f","v","artifacts/graphs/g.npz","sha256:x",1,0,0,1,0,0,{}).validate()
    GraphPairRecord("p","s","g1","g2","d","m","artifacts/pairs/p.npz",{}).validate()
    with pytest.raises(ValueError): GraphRecord("g","s","c","other","f","v","x","sha256:x",1,0,0,1,0,0,{}).validate()
    with pytest.raises(ValueError): GraphPairRecord("p","s","g1","g2","d","m","../p.npz",{}).validate()
