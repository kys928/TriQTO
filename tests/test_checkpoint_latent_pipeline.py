from __future__ import annotations

from pathlib import Path
import pytest
import torch

from triqto.model import TriQTOModel, TriQTOModelConfig
from triqto.training import CurriculumStageConfig, DeterministicLRScheduler, LossConfig, OptimizerConfig, SchedulerConfig, TrainingConfig, TrainingDataSpec, build_optimizer, save_training_checkpoint, training_operational_config_id, training_recipe_id, training_run_id, training_schema_id
from triqto.training.latent_extraction import LatentExtractionConfig, restore_checkpoint_for_latents


def _model_config() -> TriQTOModelConfig:
    return TriQTOModelConfig(hidden_dim=16, graph_message_passing_layers=1, residual_mlp_layers=1, topology_input_dim=4, hilbert_deformation_dim=4, topology_prediction_dim=4, dropout=0.0, initialization_seed=77)


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        run_name="latent_checkpoint_test",
        stages=(CurriculumStageConfig(name="diagnosis", epochs=1, tasks=("diagnosis",)),),
        batch_size=1,
        optimizer=OptimizerConfig(name="sgd", learning_rate=1e-3, weight_decay=0.0, momentum=0.0),
        scheduler=SchedulerConfig(name="constant", minimum_learning_rate_ratio=1.0),
        loss=LossConfig(geometry_weight=0.0, uncertainty_weighting=False),
        device="cpu", topology_loss_weight=0.0,
    )


def _spec() -> TrainingDataSpec:
    result = TrainingDataSpec(
        training_view_dataset_id="viewdataset_latent_test",
        distortion_labels=("phase_like", "amplitude_like", "entanglement_like", "lattice_layout_like", "noise_readout_like", "mixed_uncertain"),
        distortion_mapping=(("rx_overrotation", "amplitude_like"),),
        action_edit_types=("no_op", "rx", "ry", "rz", "rzz", "layout", "routing", "diagnostic_basis"),
        action_edit_mapping=(("append_rx", "rx"),),
        action_feature_names=("edit_count", "risk_score", "depth_delta", "gate_delta", "is_no_op"),
        action_feature_mean=(0.0,) * 5,
        action_feature_std=(1.0,) * 5,
        topology_feature_names=(), topology_feature_mean=(), topology_feature_std=(),
        backend_feature_names=tuple(f"backend_feature_{index}" for index in range(16)),
        backend_feature_mean=(0.0,) * 16, backend_feature_std=(1.0,) * 16,
        topology_input_dim=4, normalize_action_features=True,
        normalize_topology_features=True, normalize_backend_features=True,
        adapter_version="latent-test-adapter",
    )
    result.validate()
    return result


def _checkpoint(tmp_path: Path, *, global_step: int) -> tuple[Path, str]:
    cfg, model_cfg, spec = _training_config(), _model_config(), _spec()
    model = TriQTOModel(model_cfg)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = DeterministicLRScheduler(optimizer, cfg.scheduler, total_steps=1)
    if global_step > 0:
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    recipe = training_recipe_id(spec.training_view_dataset_id, model.architecture_id, "modelconfig_latent_test", cfg, spec.content_hash)
    operational = training_operational_config_id(cfg)
    run_id = training_run_id(recipe, operational)
    path = tmp_path / f"checkpoint-{global_step}.npz"
    save_training_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, training_config=cfg, model_config=model_cfg, data_spec=spec, training_schema_id=training_schema_id(), training_recipe_id=recipe, operational_config_id=operational, training_run_id=run_id, epoch_completed=0, stage_index=0, global_step=global_step, best_validation_loss=1.0, best_epoch=0, kind="epoch")
    return path, run_id


def test_restore_checkpoint_for_latents_requires_trained_step_and_is_deterministic(tmp_path: Path) -> None:
    path, run_id = _checkpoint(tmp_path, global_step=1)
    first_model, first, first_spec = restore_checkpoint_for_latents(path, expected_training_run_id=run_id)
    second_model, second, second_spec = restore_checkpoint_for_latents(path, expected_training_run_id=run_id)
    assert first["checkpoint_id"] == second["checkpoint_id"]
    assert first["content_hash"] == second["content_hash"]
    assert first_spec == second_spec
    for name, tensor in first_model.state_dict().items():
        assert torch.equal(tensor, second_model.state_dict()[name])


def test_restore_checkpoint_for_latents_rejects_zero_step(tmp_path: Path) -> None:
    path, run_id = _checkpoint(tmp_path, global_step=0)
    with pytest.raises(ValueError, match="zero-step"):
        restore_checkpoint_for_latents(path, expected_training_run_id=run_id)


def test_latent_extraction_config_rejects_unknown_head() -> None:
    assert LatentExtractionConfig(split="validation").split == "validation"
    with pytest.raises(ValueError, match="head"):
        LatentExtractionConfig(head="not_a_head")
