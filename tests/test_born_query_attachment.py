from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from triqto.model import TriQTOModelConfig
from triqto.phase15_6.born_query_attachment import attach_born_query_coordinates
from triqto.training import (
    build_model_ready_example,
    load_model_ready_artifact,
    load_model_ready_dataset,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    artifact = root / "artifacts" / "items" / "aa" / "born.npz"
    artifact.parent.mkdir(parents=True)
    np.savez_compressed(
        artifact,
        schema_version=np.asarray("triqto.phase12.model_preprocessing.v1"),
        view_item_id=np.asarray("born_item"),
        training_view_id=np.asarray("born_view"),
        task=np.asarray("born_prediction"),
        split=np.asarray("train"),
        split_group_id=np.asarray("group_0"),
        entity_id=np.asarray("entity_0"),
        x_graph_node_features=np.zeros((2, 13), dtype=np.float32),
        x_graph_edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        x_graph_edge_features=np.zeros((2, 10), dtype=np.float32),
        x_graph_edge_event_index=np.asarray([0, 0], dtype=np.int64),
        x_graph_gate_features=np.zeros((1, 16), dtype=np.float32),
        x_graph_gate_qubit_ptr=np.asarray([0, 2], dtype=np.int64),
        x_graph_gate_qubit_indices=np.asarray([0, 1], dtype=np.int64),
        x_graph_parameter_sin=np.asarray([np.sin(0.2)], dtype=np.float32),
        x_graph_parameter_cos=np.asarray([np.cos(0.2)], dtype=np.float32),
        x_backend_available_mask=np.asarray(False, dtype=np.bool_),
        x_backend_features=np.zeros(16, dtype=np.float32),
        x_topology_available_mask=np.asarray(False, dtype=np.bool_),
        x_topology_source_available_mask=np.asarray(False, dtype=np.bool_),
        x_topology_materialized_mask=np.asarray(False, dtype=np.bool_),
        y_born_target_outcome_bitstrings=np.asarray(["00", "11"]),
        y_born_target_probabilities=np.asarray([0.6, 0.4], dtype=np.float32),
    )
    row = {
        "view_item_id": "born_item",
        "training_view_id": "born_view",
        "training_view_dataset_id": "dataset_query_test",
        "task": "born_prediction",
        "split": "train",
        "split_group_id": "group_0",
        "entity_id": "entity_0",
        "artifact_ref": artifact.relative_to(root).as_posix(),
        "content_hash": _sha(artifact),
        "source_artifact_ref": "source/born.npz",
        "source_content_hash": "0" * 64,
        "hilbert_available_mask": False,
        "topology_available_mask": False,
        "has_action_candidates": False,
        "deployable_candidate_count": 0,
        "should_act": None,
        "repair_count": 0,
    }
    manifest = root / "manifests" / "processed_item_manifest.parquet"
    manifest.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), manifest)
    contract = {
        "schema_version": "triqto.phase12.model_preprocessing.v1",
        "model_inputs": "arrays beginning with x_ only",
        "model_targets": "arrays beginning with y_ only",
        "topology_attachment": {
            "schema_version": "triqto.phase11_phase12.topology_attachment.v1",
            "lambda_top": 0.0,
            "head_policy": {
                "joint_multitask.action_ranking": False,
                "joint_multitask.born_prediction": False,
            },
        },
    }
    (root / "manifests" / "model_input_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (root / "manifests" / "should_act_class_weights.json").write_text(
        json.dumps({"negative": 0.65, "positive": 2.0}), encoding="utf-8"
    )
    (root / "preprocessed_complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "schema_version": "triqto.phase12.model_preprocessing.v1",
                "lambda_top": 0.0,
                "accepted_count": 1,
                "processed_item_manifest_sha256": _sha(manifest),
            }
        ),
        encoding="utf-8",
    )
    return root, artifact


def _config() -> TriQTOModelConfig:
    return TriQTOModelConfig(
        hidden_dim=16,
        graph_message_passing_layers=1,
        residual_mlp_layers=1,
        backend_input_dim=16,
        topology_input_dim=121,
        hilbert_deformation_dim=8,
        topology_prediction_dim=8,
        dropout=0.0,
        initialization_seed=17,
    )


def test_born_query_attachment_preserves_source_and_enables_prediction(tmp_path: Path) -> None:
    source, source_artifact = _source(tmp_path)
    source_artifact_hash = _sha(source_artifact)
    source_manifest_hash = _sha(source / "manifests" / "processed_item_manifest.parquet")

    result = attach_born_query_coordinates(
        source_root=source,
        output_parent=tmp_path / "outputs",
    )
    assert result["status"] == "complete"
    assert result["rewritten_rows"] == 1
    assert result["target_probabilities_copied"] is False
    assert _sha(source_artifact) == source_artifact_hash
    assert _sha(source / "manifests" / "processed_item_manifest.parquet") == source_manifest_hash

    dataset = load_model_ready_dataset(result["output_root"])
    artifact = load_model_ready_artifact(dataset, dataset.records[0])
    assert "x_born_query_outcome_bitstrings" in artifact.inputs
    assert "x_born_input_probabilities" not in artifact.inputs
    assert "y_born_target_probabilities" in artifact.targets
    example = build_model_ready_example(artifact, _config())
    assert example.model_batch.born is None
    assert example.model_batch.born_queries is not None
    assert example.model_batch.born_queries.outcome_bits.shape == (2, 2)
