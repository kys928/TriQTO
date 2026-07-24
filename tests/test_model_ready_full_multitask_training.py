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
    run_model_ready_full_training,
)

_TASKS = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "joint_multitask",
    "hardware_masked",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topology_arrays() -> dict[str, np.ndarray]:
    parameter_names = np.asarray([f"p_{index}" for index in range(55)])
    born_names = np.asarray([f"b_{index}" for index in range(55)])
    combined_names = np.concatenate((parameter_names, born_names))
    parameter = np.linspace(-1.0, 0.0, 55, dtype=np.float32)
    born = np.linspace(0.0, 1.0, 55, dtype=np.float32)
    combined = np.concatenate((parameter, born))
    alignment = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    result: dict[str, np.ndarray] = {
        "x_topology_available_mask": np.asarray(True, dtype=np.bool_),
        "x_topology_source_available_mask": np.asarray(True, dtype=np.bool_),
        "x_topology_materialized_mask": np.asarray(True, dtype=np.bool_),
        "x_topology_manifold_available_mask": np.asarray(
            [True, False, True], dtype=np.bool_
        ),
        "x_topology_features": combined,
        "x_topology_feature_names": combined_names,
        "x_topology_feature_mask": np.ones(110, dtype=np.bool_),
        "x_topology_positive_infinity_mask": np.zeros(110, dtype=np.bool_),
        "x_topology_negative_infinity_mask": np.zeros(110, dtype=np.bool_),
        "x_topology_alignment_features": alignment,
        "x_topology_alignment_feature_names": np.asarray(
            [f"a_{index}" for index in range(11)]
        ),
        "x_topology_alignment_feature_mask": np.ones(11, dtype=np.bool_),
        "x_topology_alignment_positive_infinity_mask": np.zeros(
            11, dtype=np.bool_
        ),
        "x_topology_alignment_negative_infinity_mask": np.zeros(
            11, dtype=np.bool_
        ),
        "x_topology_parameter_features": parameter,
        "x_topology_parameter_feature_names": parameter_names,
        "x_topology_parameter_feature_mask": np.ones(55, dtype=np.bool_),
        "x_topology_parameter_positive_infinity_mask": np.zeros(
            55, dtype=np.bool_
        ),
        "x_topology_parameter_negative_infinity_mask": np.zeros(
            55, dtype=np.bool_
        ),
        "x_topology_born_features": born,
        "x_topology_born_feature_names": born_names,
        "x_topology_born_feature_mask": np.ones(55, dtype=np.bool_),
        "x_topology_born_positive_infinity_mask": np.zeros(
            55, dtype=np.bool_
        ),
        "x_topology_born_negative_infinity_mask": np.zeros(
            55, dtype=np.bool_
        ),
    }
    return result


def _arrays(
    *,
    item_id: str,
    entity_id: str,
    group_id: str,
    task: str,
    split: str,
    should_act: bool,
    variant: int,
) -> dict[str, np.ndarray]:
    born_target = (
        np.asarray([0.65, 0.35], dtype=np.float32)
        if variant % 2 == 0
        else np.asarray([0.4, 0.6], dtype=np.float32)
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray("triqto.phase12.model_preprocessing.v1"),
        "view_item_id": np.asarray(item_id),
        "training_view_id": np.asarray(f"view_{item_id}"),
        "task": np.asarray(task),
        "split": np.asarray(split),
        "split_group_id": np.asarray(group_id),
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
        "x_topology_available_mask": np.asarray(False, dtype=np.bool_),
        "x_topology_source_available_mask": np.asarray(False, dtype=np.bool_),
        "x_topology_materialized_mask": np.asarray(False, dtype=np.bool_),
    }
    diagnosis_active = task in {"diagnosis", "joint_multitask", "hardware_masked"}
    action_active = task in {"action_ranking", "joint_multitask", "hardware_masked"}
    born_active = task in {"born_prediction", "joint_multitask", "hardware_masked"}
    if diagnosis_active:
        arrays.update(
            {
                "y_diagnosis_distortion_type": np.asarray("phase_rz_drift"),
                "y_diagnosis_strength": np.asarray([0.2], dtype=np.float32),
                "y_diagnosis_strength_available_mask": np.asarray(
                    [True], dtype=np.bool_
                ),
                "y_diagnosis_affected_qubit_mask": np.asarray(
                    [True, False], dtype=np.bool_
                ),
            }
        )
    if action_active:
        arrays.update(
            {
                "x_action_candidate_ids": np.asarray(["edit", "no_op"]),
                "x_action_candidate_feature_names": np.asarray(
                    [
                        "edit_count",
                        "risk_score",
                        "depth_delta",
                        "gate_delta",
                        "is_no_op",
                    ]
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
                "y_should_act": np.asarray(should_act, dtype=np.bool_),
                "y_should_act_weight": np.asarray(
                    2.0 if should_act else 0.65, dtype=np.float32
                ),
                "y_ranking_loss_mask": np.asarray(should_act, dtype=np.bool_),
                "y_candidate_reward": np.asarray([0.2, 0.0], dtype=np.float32),
                "y_candidate_rank": np.asarray([1, 2], dtype=np.int64),
                "y_candidate_selected_mask": np.asarray(
                    [should_act, False], dtype=np.bool_
                ),
                "y_candidate_listwise_distribution": np.asarray(
                    [1.0, 0.0] if should_act else [0.0, 0.0],
                    dtype=np.float32,
                ),
                "y_candidate_eligible_mask": np.asarray(
                    [True, True], dtype=np.bool_
                ),
            }
        )
    if born_active:
        arrays.update(
            {
                "y_born_target_outcome_bitstrings": np.asarray(["00", "11"]),
                "y_born_target_probabilities": born_target,
            }
        )
    if task == "joint_multitask":
        arrays.update(_topology_arrays())
        arrays.update(
            {
                "y_joint_head_names": np.asarray(
                    ["diagnosis", "action_ranking", "born_prediction"]
                ),
                "y_joint_head_target_available_mask": np.asarray(
                    [True, True, True], dtype=np.bool_
                ),
            }
        )
    if task == "hardware_masked":
        arrays.update(
            {
                "y_hardware_head_names": np.asarray(
                    ["diagnosis", "action_ranking", "born_prediction"]
                ),
                "y_hardware_head_target_available_mask": np.asarray(
                    [True, True, True], dtype=np.bool_
                ),
            }
        )
    return arrays


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "model_ready"
    rows: list[dict[str, object]] = []
    index = 0
    for task in _TASKS:
        for split in ("train", "validation"):
            for local in range(2):
                item_id = f"item_{index:03d}"
                entity_id = f"entity_{task}_{split}_{local}"
                group_id = f"group_{task}_{split}_{local}"
                should_act = local == 0
                artifact = (
                    root
                    / "artifacts"
                    / "items"
                    / f"{index:02d}"
                    / f"{item_id}.npz"
                )
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    artifact,
                    **_arrays(
                        item_id=item_id,
                        entity_id=entity_id,
                        group_id=group_id,
                        task=task,
                        split=split,
                        should_act=should_act,
                        variant=index,
                    ),
                )
                action_active = task in {
                    "action_ranking",
                    "joint_multitask",
                    "hardware_masked",
                }
                rows.append(
                    {
                        "view_item_id": item_id,
                        "training_view_id": f"view_{item_id}",
                        "training_view_dataset_id": "dataset_full_test",
                        "task": task,
                        "split": split,
                        "split_group_id": group_id,
                        "entity_id": entity_id,
                        "artifact_ref": artifact.relative_to(root).as_posix(),
                        "content_hash": _sha256(artifact),
                        "source_artifact_ref": f"source/{item_id}.npz",
                        "source_content_hash": "0" * 64,
                        "hilbert_available_mask": False,
                        "topology_available_mask": task == "joint_multitask",
                        "has_action_candidates": action_active,
                        "deployable_candidate_count": 2 if action_active else 0,
                        "should_act": should_act if action_active else None,
                        "repair_count": 0,
                        "topology_attachment_status": (
                            "attached" if task == "joint_multitask" else "not_attached"
                        ),
                    }
                )
                index += 1
    manifest = root / "manifests" / "processed_item_manifest.parquet"
    manifest.parent.mkdir(parents=True)
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
        run_name="full_multitask_test",
        seed=23,
        stages=(
            CurriculumStageConfig(
                name="foundation",
                epochs=1,
                tasks=("diagnosis", "action_ranking", "born_prediction"),
            ),
            CurriculumStageConfig(
                name="joint", epochs=1, tasks=("joint_multitask",)
            ),
            CurriculumStageConfig(
                name="hardware", epochs=1, tasks=("hardware_masked",)
            ),
        ),
        batch_size=2,
        optimizer=OptimizerConfig(learning_rate=1.0e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(
            name="constant", warmup_steps=0, minimum_learning_rate_ratio=1.0
        ),
        loss=LossConfig(
            uncertainty_weighting=False,
            privileged_oracle_loss_weight=0.0,
            topology_weight=0.0,
        ),
        deterministic_algorithms=True,
        device="cpu",
        num_workers=0,
        early_stopping_patience=0,
        normalize_action_features=False,
        normalize_topology_features=False,
        normalize_backend_features=False,
        max_items=100,
        max_hilbert_amplitudes_per_batch=1,
    )


def test_full_multitask_runner_vectorizes_all_tasks_and_restores_checkpoint(
    tmp_path: Path,
) -> None:
    source = _dataset_root(tmp_path)
    result = run_model_ready_full_training(
        model_ready_root=source,
        output_root=tmp_path / "runs",
        model_config=_model_config(),
        training_config=_training_config(),
        train_limit_per_task=2,
        validation_limit_per_task=2,
        progress_every_batches=0,
    )
    assert result.status == "complete"
    assert result.summary["test_rows_used"] is False
    assert result.summary["lambda_top"] == 0.0
    assert result.summary["global_step"] == 5
    assert result.summary["executed_epochs"] == [0, 1, 2]
    output = result.output_root
    assert (output / "model_ready_full_complete.json").is_file()
    assert not (output / "model_ready_full_incomplete.json").exists()
    for epoch in result.summary["epoch_metrics"]:
        assert epoch["validation"]["losses"]["topology"] == 0.0
        assert epoch["validation"]["gradient"]["clipped_step_fraction"] == 0.0
    final = output / result.summary["final_checkpoint"]["artifact_ref"]
    metadata = load_training_checkpoint(final)
    assert metadata["training_run_id"] == result.summary["run_id"]
    assert metadata["epoch_completed"] == 2
    repeated = run_model_ready_full_training(
        model_ready_root=source,
        output_root=tmp_path / "runs",
        model_config=_model_config(),
        training_config=_training_config(),
        train_limit_per_task=2,
        validation_limit_per_task=2,
        progress_every_batches=0,
    )
    assert repeated.status == "already_complete"
