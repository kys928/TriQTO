from __future__ import annotations

import json
from pathlib import Path

import pytest

from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    generate_dataset,
    write_dataset,
)
from triqto.graph import (
    GraphConversionConfig,
    convert_completed_dataset_to_graphs,
    load_graph_artifact,
    load_pair_artifact,
    write_graph_dataset,
)
from triqto.storage import GraphPairRecord, GraphRecord, ManifestReader


def phase7_source(root: Path) -> Path:
    config = DatasetGenerationConfig(
        dataset_name="persist-source",
        base_seed=19,
        circuit_specs=[
            CircuitGenerationSpec(
                family="hardware_efficient_ansatz",
                n_qubits=2,
                generator_kwargs={
                    "layers": 1,
                    "entanglement": "none",
                    "measure": True,
                },
                repetitions=1,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            )
        ],
        ideal_shots=8,
        store_statevectors=False,
        max_samples=4,
    )
    write_dataset(generate_dataset(config), root)
    return root


def test_immutable_graph_dataset_roundtrip_and_actual_config(tmp_path):
    source = phase7_source(tmp_path / "source")
    config = GraphConversionConfig(
        max_gate_events=777,
        max_probability_outcomes=123,
        include_supplemental_counts=False,
    )
    result = convert_completed_dataset_to_graphs(source, config)
    output = tmp_path / "graphs"
    written = write_graph_dataset(result, output)
    assert json.loads((output / "graph_config.json").read_text()) == {
        "schema_version": config.schema_version,
        "max_gate_events": 777,
        "max_probability_outcomes": 123,
        "include_supplemental_counts": False,
        "reject_conditioned_operations": True,
    }
    complete = json.loads((output / "graph_complete.json").read_text())
    assert complete["managed_files"] == sorted(complete["managed_files"])
    assert set(complete["managed_files"]) == {
        path.relative_to(output).as_posix() for path in written.written_paths
    }
    reader = ManifestReader(output / "manifests")
    graph_records = reader.read_typed_records("graph_manifest", GraphRecord)
    pair_records = reader.read_typed_records("graph_pair_manifest", GraphPairRecord)
    for record in graph_records:
        load_graph_artifact(output / record.graph_ref, record.content_hash)
    for record in pair_records:
        load_pair_artifact(output / record.pair_ref, record.content_hash)
    with pytest.raises(FileExistsError):
        write_graph_dataset(result, output)


def test_failed_publication_cleans_staging_and_leaves_final_absent(tmp_path, monkeypatch):
    source = phase7_source(tmp_path / "source")
    result = convert_completed_dataset_to_graphs(source)
    output = tmp_path / "graphs"
    import triqto.graph.artifacts as artifacts

    def fail_pair(*args, **kwargs):
        raise RuntimeError("injected pair persistence failure")

    monkeypatch.setattr(artifacts, "save_pair_artifact", fail_pair)
    with pytest.raises(RuntimeError, match="injected"):
        write_graph_dataset(result, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".graphs.staging-*"))


def test_logical_reproduction_across_output_roots(tmp_path):
    source = phase7_source(tmp_path / "source")
    first = convert_completed_dataset_to_graphs(source)
    second = convert_completed_dataset_to_graphs(source)
    assert [graph.graph_id for graph in first.graphs] == [
        graph.graph_id for graph in second.graphs
    ]
    assert [record.content_hash for record in first.graph_records] == [
        record.content_hash for record in second.graph_records
    ]
    write_graph_dataset(first, tmp_path / "out-a")
    write_graph_dataset(second, tmp_path / "out-b")
    first_complete = json.loads((tmp_path / "out-a" / "graph_complete.json").read_text())
    second_complete = json.loads((tmp_path / "out-b" / "graph_complete.json").read_text())
    assert first_complete == second_complete
