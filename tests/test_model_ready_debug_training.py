from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from triqto.model import TriQTOModelConfig
from triqto.training import (
    CurriculumStageConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    load_training_checkpoint,
    run_model_ready_debug_training,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(
    *,
    item_id: str,
    entity_id: str,
    split_group_id: str,
    split: str,
    should_act: bool,
) -> dict[str, np.ndarray]:
    ranking = should_act
    return {
        "schema_version": np.asarray("triqto.phase12.model_preprocessing.v1"),
        "view_item_id": np.asarray(item_id),
        "training_view_id": np.asarray(f"view_{item_id}"),
        "task": np.asarray("action_ranking"),
        "split": np.asarray(split),
        "split_group_id": np.asarray(split_group_id),
        "entity_id": np.asarray(entity_id),
        "x_graph_node_features": np.zeros((2, 13), dtype=np.float32),
        "x_graph_edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "x_graph_edge_features": np.zeros((2, 10), dtype=np.float32),
        "x_graph_edge_event_index": np.asarray([0, 0], dtype=np.int64),
        "x_graph_gate_features": np.zeros((1, 16), dtype=np.float32),
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
        "x_topology_available_mask": np.asarray(False, dtype=np.bool_),
        "x_topology_source_available_mask": np.asarray(False, dtype=np.bool_),
        "x_topology_materialized_mask": np.asarray(False, dtype=np.bool_),
        "y_should_act": np.asarray(should_act, dtype=np.bool_),
        "y_should_act_weight": np.asarray(
            2.0 if should_act else 0.65, dtype=np.float32
        ),
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
    }


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "model_ready"
    rows = []
    definitions = [
        ("train", True),
        ("train", False),
        ("train", True),
        ("train", False),
        ("validation", True),
        ("validation", False),
    ]
    for index, (split, should_act) in enumerate(definitions):
        item_id = f"item_{index}"
        entity_id = f"entity_{index}"
        group_id = f"group_{index}"
        artifact = root / "artifacts" / "items" / f"{index:02d}" / f"{item_id}.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            **_arrays(
                item_id=item_id,
                entity_id=entity_id,
                split_group_id=group_id,
                split=split,
                should_act=should_act,
            ),
        )
        rows.append(
            {
                "view_item_id": item_id,
                "training_view_id": f"view_{item_id}",
                "training_view_dataset_id": "dataset_debug",
                "task": "action_ranking",
                "split": split,
                "split_group_id": group_id,
                "entity_id": entity_id,
                "artifact_ref": artifact.relative_to(root).as_posix(),
                "content_hash": _sha256(artifact),
                "source_artifact_ref": f"source/{item_id}.npz",
                "source_content_hash": "0" * 64,
                "hilbert_available_mask": False,
                "topology_available_mask": False,
                "has_action_candidates": True,
                "deployable_candidate_count": 2,
                "should_act": should_act,
                "repair_count": 0,
            }
        )
    manifest = root / "manifests" / "processed_item_manifest.parquet"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), manifest)
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
        "counts": {"published_model_items": len(rows)},
    }
    (root / "preprocessed_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (root / "topology_attachment_complete.json").write_text(
        json.dumps(
            {
                **completion,
                "scientific_boundaries": {
                    "cross_split_groups_audit_only": True,
                    "topology_supervised_target_present": False,
                    "action_head_topology_enabled": False,
                    "born_prediction_head_topology_enabled": False,
                    "hardware_attachment_requested": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _model_config() -> TriQTOModelConfig:
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


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        run_name="model_ready_debug_test",
        seed=23,
        stages=(
            CurriculumStageConfig(
                name="action_debug", epochs=1, tasks=("action_ranking",)
            ),
        ),
        batch_size=2,
        optimizer=OptimizerConfig(learning_rate=1.0e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(
            name="constant", warmup_steps=0, minimum_learning_rate_ratio=1.0
        ),
        loss=LossConfig(
            action_selection_weight=1.0,
            action_rank_distribution_weight=0.5,
            action_reward_weight=0.25,
            uncertainty_weighting=False,
            privileged_oracle_loss_weight=0.0,
            topology_weight=0.0,
        ),
        deterministic_algorithms=True,
        device="cpu",
        normalize_action_features=False,
        normalize_topology_features=False,
        normalize_backend_features=False,
        max_items=100,
    )


def test_model_ready_debug_runner_publishes_metrics_and_safe_checkpoints(
    tmp_path: Path,
) -> None:
    source = _dataset_root(tmp_path)
    result = run_model_ready_debug_training(
        model_ready_root=source,
        output_root=tmp_path / "runs",
        model_config=_model_config(),
        training_config=_training_config(),
        task="action_ranking",
        train_items=4,
        validation_items=2,
    )
    assert result["status"] == "complete"
    assert result["test_rows_used"] is False
    assert result["lambda_top"] == 0.0
    assert result["global_step"] == 2
    assert result["epoch_metrics"][0]["train"]["item_count"] == 4
    assert result["epoch_metrics"][0]["validation"]["item_count"] == 2
    output = Path(result["output_root"])
    assert (output / "model_ready_debug_complete.json").is_file()
    final_path = output / result["final_checkpoint"]["artifact_ref"]
    metadata = load_training_checkpoint(final_path)
    assert metadata["training_run_id"] == result["run_id"]
    assert metadata["epoch_completed"] == 0

    repeated = run_model_ready_debug_training(
        model_ready_root=source,
        output_root=tmp_path / "runs",
        model_config=_model_config(),
        training_config=_training_config(),
        task="action_ranking",
        train_items=4,
        validation_items=2,
    )
    assert repeated["status"] == "already_complete"
