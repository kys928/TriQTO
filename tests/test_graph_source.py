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
    graph_content_hash,
    load_completed_phase7_dataset,
)


def make_phase7_source(
    root: Path,
    *,
    store_statevectors: bool = False,
    ideal_shots: int | None = 8,
) -> Path:
    config = DatasetGenerationConfig(
        dataset_name="phase8-source",
        base_seed=17,
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
                kwargs={"strength": 0.3, "qubits": [0]},
            ),
            DistortionSpec(
                name="readout_bitflip_marker",
                kwargs={"probability": 0.1, "qubits": [0]},
            ),
        ],
        ideal_shots=ideal_shots,
        store_statevectors=store_statevectors,
        max_samples=10,
    )
    write_dataset(generate_dataset(config), root)
    return root


def file_bytes(root: Path) -> dict[str, bytes]:
    marker = json.loads((root / "dataset_complete.json").read_text())
    return {ref: (root / ref).read_bytes() for ref in marker["managed_files"]}


def test_completed_source_load_and_byte_immutability(tmp_path):
    root = make_phase7_source(tmp_path / "source")
    before = file_bytes(root)
    loaded = load_completed_phase7_dataset(root)
    after = file_bytes(root)
    assert before == after
    assert len(loaded.samples) == 2
    assert loaded.counts_by_exact_run_id
    assert loaded.source_snapshot.aggregate_sha256.startswith("sha256:")


def test_shared_clean_graph_has_complete_provenance_without_one_sample_owner(tmp_path):
    root = make_phase7_source(tmp_path / "source")
    result = convert_completed_dataset_to_graphs(root)
    assert len(result.pairs) == 2
    assert len(result.graphs) == 3
    clean = [graph for graph in result.graphs if graph.role == "clean"]
    assert len(clean) == 1
    assert clean[0].source_sample_ids == tuple(
        sorted(pair.sample_id for pair in result.pairs)
    )
    graph_record = next(
        record for record in result.graph_records if record.graph_id == clean[0].graph_id
    )
    assert not hasattr(graph_record, "sample_id")
    assert graph_record.metadata["source_sample_ids"] == list(clean[0].source_sample_ids)


def test_supplemental_count_toggle_does_not_change_ids_or_hashes(tmp_path):
    root = make_phase7_source(tmp_path / "source")
    included = convert_completed_dataset_to_graphs(
        root,
        GraphConversionConfig(include_supplemental_counts=True),
    )
    excluded = convert_completed_dataset_to_graphs(
        root,
        GraphConversionConfig(include_supplemental_counts=False),
    )
    assert [graph.graph_id for graph in included.graphs] == [
        graph.graph_id for graph in excluded.graphs
    ]
    assert [pair.graph_pair_id for pair in included.pairs] == [
        pair.graph_pair_id for pair in excluded.pairs
    ]
    assert [graph_content_hash(graph) for graph in included.graphs] == [
        graph_content_hash(graph) for graph in excluded.graphs
    ]
    assert all(graph.supplemental_counts_available_mask for graph in included.graphs)
    assert not any(graph.supplemental_counts_available_mask for graph in excluded.graphs)


def test_marker_only_pair_uses_metric_applicability_warning(tmp_path):
    root = make_phase7_source(tmp_path / "source")
    result = convert_completed_dataset_to_graphs(root)
    marker_pair = next(pair for pair in result.pairs if pair.marker_only)
    assert marker_pair.born_zero_shift is True
    assert marker_pair.applicability_warning


def test_source_loader_rejects_completion_identity_mismatch(tmp_path):
    root = make_phase7_source(tmp_path / "source")
    marker_path = root / "dataset_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["scientific_generation_id"] = "generation_wrong"
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="scientific_generation_id"):
        load_completed_phase7_dataset(root)


def test_source_loader_requires_managed_files(tmp_path):
    root = make_phase7_source(tmp_path / "source")
    marker_path = root / "dataset_complete.json"
    marker = json.loads(marker_path.read_text())
    del marker["managed_files"]
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="managed_files"):
        load_completed_phase7_dataset(root)


def test_statevectors_are_not_loaded(tmp_path, monkeypatch):
    root = make_phase7_source(
        tmp_path / "source",
        store_statevectors=True,
        ideal_shots=None,
    )
    import numpy as np

    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Phase 8 must not load statevector NPY artifacts")
        ),
    )
    loaded = load_completed_phase7_dataset(root)
    assert loaded.statevector_storage_enabled is True
