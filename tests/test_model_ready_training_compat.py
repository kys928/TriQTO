from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from triqto.model import TriQTOModel, TriQTOModelConfig
from triqto.model.constants import HEAD_ORDER, STREAM_ORDER
from triqto.training import (
    CANONICAL_TOPOLOGY_INPUT_DIM,
    build_model_ready_example,
    compute_model_ready_action_losses,
    load_model_ready_artifact,
    load_model_ready_dataset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _arrays(*, should_act: bool, ranking: bool) -> dict[str, np.ndarray]:
    node = np.zeros((2, 13), dtype=np.float32)
    gate = np.zeros((1, 16), dtype=np.float32)
    edge = np.zeros((2, 10), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray("triqto.phase12.model_preprocessing.v1"),
        "view_item_id": np.asarray("item_joint"),
        "training_view_id": np.asarray("view_joint"),
        "task": np.asarray("joint_multitask"),
        "split": np.asarray("train"),
        "split_group_id": np.asarray("group_0"),
        "entity_id": np.asarray("entity_0"),
        "x_graph_node_features": node,
        "x_graph_edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "x_graph_edge_features": edge,
        "x_graph_edge_event_index": np.asarray([0, 0], dtype=np.int64),
        "x_graph_gate_features": gate,
        "x_graph_gate_qubit_ptr": np.asarray([0, 2], dtype=np.int64),
        "x_graph_gate_qubit_indices": np.asarray([0, 1], dtype=np.int64),
        "x_graph_parameter_sin": np.asarray([np.sin(0.2)], dtype=np.float32),
        "x_graph_parameter_cos": np.asarray([np.cos(0.2)], dtype=np.float32),
        "x_born_input_outcome_bitstrings": np.asarray(["00", "11"]),
        "x_born_input_probabilities": np.asarray([0.5, 0.5], dtype=np.float32),
        "x_born_input_sqrt_probabilities": np.sqrt(
            np.asarray([0.5, 0.5], dtype=np.float32)
        ),
        "x_backend_available_mask": np.asarray(False, dtype=np.bool_),
        "x_backend_features": np.zeros(16, dtype=np.float32),
        "x_action_candidate_ids": np.asarray(["candidate_0", "candidate_1"]),
        "x_action_candidate_feature_names": np.asarray(
            ["edit_count", "risk_score", "depth_delta", "gate_delta", "is_no_op"]
        ),
        "x_action_candidate_features": np.asarray(
            [[1.0, 0.1, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "x_action_candidate_count": np.asarray(2, dtype=np.int64),
        "x_action_edit_ptr": np.asarray([0, 1, 1], dtype=np.int64),
        "x_action_edit_magnitudes": np.asarray([0.2], dtype=np.float32),
        "x_action_edit_types": np.asarray(["append_rx"]),
        "x_action_edit_qubit_ptr": np.asarray([0, 1], dtype=np.int64),
        "x_action_edit_qubits": np.asarray([0], dtype=np.int64),
        "x_topology_available_mask": np.asarray(True, dtype=np.bool_),
        "x_topology_source_available_mask": np.asarray(True, dtype=np.bool_),
        "x_topology_materialized_mask": np.asarray(True, dtype=np.bool_),
        "x_topology_manifold_available_mask": np.asarray(
            [True, False, True], dtype=np.bool_
        ),
        "x_topology_features": np.concatenate(
            (
                np.linspace(-1.0, 0.0, 55, dtype=np.float32),
                np.linspace(0.0, 1.0, 55, dtype=np.float32),
            )
        ),
        "x_topology_feature_names": np.asarray(
            [f"parameter_feature_{index}" for index in range(55)]
            + [f"born_feature_{index}" for index in range(55)]
        ),
        "x_topology_feature_mask": np.ones(110, dtype=np.bool_),
        "x_topology_positive_infinity_mask": np.zeros(110, dtype=np.bool_),
        "x_topology_negative_infinity_mask": np.zeros(110, dtype=np.bool_),
        "x_topology_alignment_features": np.linspace(
            -0.5, 0.5, 11, dtype=np.float32
        ),
        "x_topology_alignment_feature_names": np.asarray(
            [f"alignment_{index}" for index in range(11)]
        ),
        "x_topology_alignment_feature_mask": np.ones(11, dtype=np.bool_),
        "x_topology_alignment_positive_infinity_mask": np.zeros(11, dtype=np.bool_),
        "x_topology_alignment_negative_infinity_mask": np.zeros(11, dtype=np.bool_),
        "x_topology_parameter_features": np.linspace(
            -1.0, 0.0, 55, dtype=np.float32
        ),
        "x_topology_parameter_feature_names": np.asarray(
            [f"feature_{index}" for index in range(55)]
        ),
        "x_topology_parameter_feature_mask": np.ones(55, dtype=np.bool_),
        "x_topology_parameter_positive_infinity_mask": np.zeros(55, dtype=np.bool_),
        "x_topology_parameter_negative_infinity_mask": np.zeros(55, dtype=np.bool_),
        "x_topology_born_features": np.linspace(
            0.0, 1.0, 55, dtype=np.float32
        ),
        "x_topology_born_feature_names": np.asarray(
            [f"feature_{index}" for index in range(55)]
        ),
        "x_topology_born_feature_mask": np.ones(55, dtype=np.bool_),
        "x_topology_born_positive_infinity_mask": np.zeros(55, dtype=np.bool_),
        "x_topology_born_negative_infinity_mask": np.zeros(55, dtype=np.bool_),
        "y_should_act": np.asarray(should_act, dtype=np.bool_),
        "y_should_act_weight": np.asarray(2.0 if should_act else 0.65, dtype=np.float32),
        "y_ranking_loss_mask": np.asarray(ranking, dtype=np.bool_),
        "y_candidate_reward": np.asarray([0.2, 0.0], dtype=np.float32),
        "y_candidate_rank": np.asarray([1, 2], dtype=np.int64),
        "y_candidate_selected_mask": np.asarray(
            [ranking, False], dtype=np.bool_
        ),
        "y_candidate_listwise_distribution": np.asarray(
            [1.0, 0.0] if ranking else [0.0, 0.0], dtype=np.float32
        ),
        "y_candidate_eligible_mask": np.asarray([True, True], dtype=np.bool_),
        "y_born_target_outcome_bitstrings": np.asarray(["00", "11"]),
        "y_born_target_probabilities": np.asarray([0.6, 0.4], dtype=np.float32),
    }
    return arrays


def _dataset_root(tmp_path: Path, *, should_act: bool, ranking: bool) -> Path:
    root = tmp_path / "model_ready"
    artifact = root / "artifacts" / "items" / "aa" / "item_joint.npz"
    artifact.parent.mkdir(parents=True)
    np.savez_compressed(artifact, **_arrays(should_act=should_act, ranking=ranking))
    row = {
        "view_item_id": "item_joint",
        "training_view_id": "view_joint",
        "training_view_dataset_id": "dataset_test",
        "task": "joint_multitask",
        "split": "train",
        "split_group_id": "group_0",
        "entity_id": "entity_0",
        "artifact_ref": artifact.relative_to(root).as_posix(),
        "content_hash": _sha256(artifact),
        "source_artifact_ref": "source/item_joint.npz",
        "source_content_hash": "0" * 64,
        "hilbert_available_mask": False,
        "topology_available_mask": True,
        "has_action_candidates": True,
        "deployable_candidate_count": 2,
        "should_act": should_act,
        "repair_count": 0,
        "topology_attachment_status": "attached",
        "topology_feature_dim": 110,
        "topology_alignment_feature_dim": 11,
        "topology_parameter_feature_dim": 55,
        "topology_born_feature_dim": 55,
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
                "joint_multitask.diagnosis": True,
                "joint_multitask.action_ranking": False,
                "joint_multitask.born_prediction": False,
                "joint_multitask.topology_audit": True,
                "hardware_masked": False,
            },
        },
    }
    (root / "manifests" / "model_input_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (root / "manifests" / "should_act_class_weights.json").write_text(
        json.dumps({"negative": 0.65, "positive": 2.0}), encoding="utf-8"
    )
    completion = {
        "complete": True,
        "schema_version": "triqto.phase11_phase12.topology_attachment.v1",
        "lambda_top": 0.0,
        "processed_item_manifest_sha256": _sha256(manifest),
        "counts": {"published_model_items": 1},
    }
    (root / "preprocessed_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    topology_completion = {
        **completion,
        "scientific_boundaries": {
            "cross_split_groups_audit_only": True,
            "topology_supervised_target_present": False,
            "action_head_topology_enabled": False,
            "born_prediction_head_topology_enabled": False,
            "hardware_attachment_requested": False,
        },
    }
    (root / "topology_attachment_complete.json").write_text(
        json.dumps(topology_completion), encoding="utf-8"
    )
    return root


def _config() -> TriQTOModelConfig:
    return TriQTOModelConfig(
        hidden_dim=32,
        graph_message_passing_layers=1,
        residual_mlp_layers=1,
        backend_input_dim=16,
        topology_input_dim=CANONICAL_TOPOLOGY_INPUT_DIM,
        hilbert_deformation_dim=8,
        topology_prediction_dim=8,
        dropout=0.0,
        initialization_seed=17,
    )


def test_model_ready_loader_forward_gate_and_topology_policy(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path, should_act=True, ranking=True)
    dataset = load_model_ready_dataset(root)
    artifact = load_model_ready_artifact(dataset, dataset.records[0])
    assert artifact.inputs and all(name.startswith("x_") for name in artifact.inputs)
    assert artifact.targets and all(name.startswith("y_") for name in artifact.targets)

    example = build_model_ready_example(artifact, _config())
    assert example.model_batch.topology is not None
    assert example.model_batch.topology.features.shape == (1, 121)
    expected_topology = np.concatenate(
        (
            artifact.inputs["x_topology_features"],
            artifact.inputs["x_topology_alignment_features"],
        )
    )
    assert np.array_equal(
        example.model_batch.topology.features.numpy().reshape(-1),
        expected_topology,
    )
    assert example.topology_ablation_inputs["x_topology_parameter_features"].size == 55
    assert example.topology_ablation_inputs["x_topology_born_features"].size == 55

    model = TriQTOModel(_config())
    output = model(example.model_batch)
    assert output.action_ranking.should_act_logit.shape == (1,)
    topology_stream = STREAM_ORDER.index("topology")
    assert not output.effective_head_stream_mask[
        0, HEAD_ORDER.index("action_ranking"), topology_stream
    ]
    assert not output.effective_head_stream_mask[
        0, HEAD_ORDER.index("born_prediction"), topology_stream
    ]
    assert output.effective_head_stream_mask[
        0, HEAD_ORDER.index("diagnosis"), topology_stream
    ]

    losses = compute_model_ready_action_losses(output, example.action_targets)
    assert losses["topology"].item() == 0.0
    assert losses["action_should_act"].item() > 0.0
    assert losses["action_rank_distribution"].item() > 0.0
    losses["total"].backward()
    gate_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "action_ranking_head.should_act" in name
    ]
    assert gate_gradients
    assert any(gradient is not None for gradient in gate_gradients)


def test_no_action_masks_candidate_ranking_loss(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path, should_act=False, ranking=False)
    dataset = load_model_ready_dataset(root)
    artifact = load_model_ready_artifact(dataset, dataset.records[0])
    example = build_model_ready_example(artifact, _config())
    output = TriQTOModel(_config())(example.model_batch)
    losses = compute_model_ready_action_losses(output, example.action_targets)
    assert losses["action_should_act"].item() > 0.0
    assert losses["action_rank_distribution"].item() == 0.0
    assert losses["action_reward"].item() == 0.0
