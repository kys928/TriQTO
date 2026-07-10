from __future__ import annotations

import json
from pathlib import Path

import pytest

from triqto.actions import load_action_engine_sources, load_completed_graph_dataset
from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    generate_dataset,
    write_dataset,
)
from triqto.graph import convert_completed_dataset_to_graphs, write_graph_dataset


def build_sources(tmp_path: Path) -> tuple[Path, Path]:
    phase7_root = tmp_path / "phase7"
    graph_root = tmp_path / "phase8"
    config = DatasetGenerationConfig(
        dataset_name="source-integrity",
        base_seed=31,
        circuit_specs=[
            CircuitGenerationSpec(
                family="bell",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=1,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="phase_rz_drift",
                kwargs={"strength": 0.2, "qubits": [0]},
            )
        ],
        store_statevectors=False,
        max_samples=2,
    )
    write_dataset(generate_dataset(config), phase7_root)
    write_graph_dataset(
        convert_completed_dataset_to_graphs(phase7_root),
        graph_root,
    )
    return phase7_root, graph_root


def test_valid_sources_cross_validate(tmp_path):
    phase7_root, graph_root = build_sources(tmp_path)
    sources = load_action_engine_sources(phase7_root, graph_root)
    assert sources.graph.completion_marker["source_scientific_generation_id"] == (
        sources.phase7.source_scientific_generation_id
    )


def test_missing_graph_completion_marker_is_rejected(tmp_path):
    _, graph_root = build_sources(tmp_path)
    (graph_root / "graph_complete.json").unlink()
    with pytest.raises(FileNotFoundError, match="completion marker"):
        load_completed_graph_dataset(graph_root)


def test_unmanaged_graph_file_is_rejected(tmp_path):
    _, graph_root = build_sources(tmp_path)
    (graph_root / "unmanaged.txt").write_text("not managed")
    with pytest.raises(ValueError, match="unmanaged"):
        load_completed_graph_dataset(graph_root)


def test_graph_conversion_identity_mismatch_is_rejected(tmp_path):
    _, graph_root = build_sources(tmp_path)
    marker_path = graph_root / "graph_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["graph_conversion_id"] = "graphconv_corrupt"
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="graph_conversion_id"):
        load_completed_graph_dataset(graph_root)


def test_phase7_and_graph_snapshot_mismatch_is_rejected(tmp_path):
    phase7_root, graph_root = build_sources(tmp_path)
    marker_path = graph_root / "graph_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["source_snapshot_hash"] = "sha256:" + "0" * 64
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    summary_path = graph_root / "graph_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source_snapshot_hash"] = marker["source_snapshot_hash"]
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="snapshot"):
        load_action_engine_sources(phase7_root, graph_root)
