from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from triqto.model import (
    ActionCandidateTensorBatch,
    GraphTensorBatch,
    OutcomeQueryTensorBatch,
    ParameterTensorBatch,
    TriQTOBatch,
    TriQTOModel,
    TriQTOModelConfig,
)
from triqto.model.constants import HEAD_ORDER, STREAM_ORDER
from triqto.training import (
    CurriculumStageConfig,
    DeterministicLRScheduler,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    build_epoch_plan,
    build_optimizer,
    collate_training_examples,
    compute_supervised_losses,
    deterministic_budget_batches,
    load_training_checkpoint,
    save_training_checkpoint,
    training_config_from_dict,
    training_config_to_dict,
    training_operational_config_id,
    training_recipe_id,
    training_run_id,
    training_schema_id,
    run_training,
)
from triqto.training.losses import _distribution_losses
from triqto.training.trainer import _run_epoch
from triqto.training.models import (
    ActionTargets,
    BornTargets,
    DiagnosisTargets,
    GeometryTargets,
    TrainingDataSpec,
    TrainingExample,
    TrainingTargets,
)


def model_config() -> TriQTOModelConfig:
    return TriQTOModelConfig(
        hidden_dim=32,
        graph_message_passing_layers=1,
        residual_mlp_layers=1,
        backend_input_dim=4,
        topology_input_dim=8,
        hilbert_deformation_dim=6,
        topology_prediction_dim=6,
        dropout=0.0,
        initialization_seed=1401,
    )


def train_config(**overrides) -> TrainingConfig:
    values = {
        "stages": (
            CurriculumStageConfig(
                name="born_only",
                epochs=2,
                tasks=("born_prediction",),
            ),
        ),
        "batch_size": 2,
        "optimizer": OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
        "scheduler": SchedulerConfig(name="constant", minimum_learning_rate_ratio=1.0),
        "loss": LossConfig(geometry_weight=0.1, uncertainty_weighting=False),
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingConfig(**values)


def data_spec() -> TrainingDataSpec:
    result = TrainingDataSpec(
        training_view_dataset_id="viewdataset_test",
        distortion_labels=(
            "phase_like",
            "amplitude_like",
            "entanglement_like",
            "lattice_layout_like",
            "noise_readout_like",
            "mixed_uncertain",
        ),
        distortion_mapping=(("rx_overrotation", "amplitude_like"),),
        action_edit_types=("no_op", "rx", "ry", "rz", "rzz", "layout", "routing", "diagnostic_basis"),
        action_edit_mapping=(("append_rx", "rx"),),
        action_feature_names=("edit_count", "risk_score", "depth_delta", "gate_delta", "is_no_op"),
        action_feature_mean=(0.0, 0.0, 0.0, 0.0, 0.0),
        action_feature_std=(1.0, 1.0, 1.0, 1.0, 1.0),
        topology_feature_names=(),
        topology_feature_mean=(),
        topology_feature_std=(),
        topology_input_dim=8,
        normalize_action_features=True,
        normalize_topology_features=True,
        adapter_version="test-adapter",
    )
    result.validate()
    return result


def example(index: int, probabilities: tuple[float, float]) -> TrainingExample:
    cfg = model_config()
    graph = GraphTensorBatch(
        node_features=torch.tensor(
            [[0.1 + index] * cfg.node_input_dim, [0.2 + index] * cfg.node_input_dim],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_features=torch.zeros((2, cfg.edge_input_dim), dtype=torch.float32),
        edge_event_index=torch.tensor([0, 0], dtype=torch.long),
        gate_features=torch.zeros((1, cfg.gate_input_dim), dtype=torch.float32),
        gate_qubit_ptr=torch.tensor([0, 2], dtype=torch.long),
        gate_qubit_indices=torch.tensor([0, 1], dtype=torch.long),
        node_batch=torch.tensor([0, 0], dtype=torch.long),
        gate_batch=torch.tensor([0], dtype=torch.long),
        graph_count=1,
    )
    values = torch.tensor([0.1 + 0.05 * index], dtype=torch.float32)
    parameter = ParameterTensorBatch(
        values=values,
        sin=torch.sin(values),
        cos=torch.cos(values),
        batch_index=torch.tensor([0], dtype=torch.long),
        available_mask=torch.tensor([True]),
    )
    queries = OutcomeQueryTensorBatch(
        outcome_bits=torch.tensor([[0, 0], [1, 1]], dtype=torch.float32),
        outcome_bit_mask=torch.ones((2, 2), dtype=torch.bool),
        batch_index=torch.tensor([0, 0], dtype=torch.long),
        available_mask=torch.tensor([True]),
        measurement_basis_codes=torch.zeros((2, 2), dtype=torch.long),
        measurement_setting_index=torch.zeros(2, dtype=torch.long),
    )
    head_active = torch.zeros((1, len(HEAD_ORDER)), dtype=torch.bool)
    head_stream = torch.zeros((1, len(HEAD_ORDER), len(STREAM_ORDER)), dtype=torch.bool)
    born_head = HEAD_ORDER.index("born_prediction")
    uncertainty_head = HEAD_ORDER.index("uncertainty")
    for head in (born_head, uncertainty_head):
        head_active[0, head] = True
        for stream in ("circuit_graph", "parameter", "phasor"):
            head_stream[0, head, STREAM_ORDER.index(stream)] = True
    model_batch = TriQTOBatch(
        graph=graph,
        parameter=parameter,
        born_queries=queries,
        head_stream_mask=head_stream,
        head_active_mask=head_active,
    )
    empty_action = ActionTargets(
        rank=torch.zeros(0, dtype=torch.long),
        reward=torch.zeros(0),
        selected_mask=torch.zeros(0, dtype=torch.bool),
        candidate_target_mask=torch.zeros(0, dtype=torch.bool),
        privileged_oracle_mask=torch.zeros(0, dtype=torch.bool),
        candidate_batch=torch.zeros(0, dtype=torch.long),
    )
    target = torch.tensor(probabilities, dtype=torch.float32)
    targets = TrainingTargets(
        diagnosis=DiagnosisTargets(
            class_index=torch.zeros(1, dtype=torch.long),
            class_mask=torch.zeros(1, dtype=torch.bool),
            strength=torch.zeros(1),
            strength_mask=torch.zeros(1, dtype=torch.bool),
            affected_qubit=torch.zeros(2),
            affected_qubit_mask=torch.zeros(2, dtype=torch.bool),
        ),
        action=empty_action,
        born_prediction=BornTargets(
            probabilities=target,
            outcome_batch=torch.tensor([0, 0]),
            row_mask=torch.ones(2, dtype=torch.bool),
        ),
        hilbert_to_born=BornTargets(
            probabilities=torch.zeros(0),
            outcome_batch=torch.zeros(0, dtype=torch.long),
            row_mask=torch.zeros(0, dtype=torch.bool),
        ),
        geometry=GeometryTargets(
            target_distance=torch.zeros((1, 1)),
            pair_mask=torch.zeros((1, 1), dtype=torch.bool),
        ),
    )
    return TrainingExample(
        view_item_id=f"item_{index}",
        entity_id=f"entity_{index}",
        task="born_prediction",
        split="train",
        split_group_id=f"group_{index}",
        model_batch=model_batch,
        targets=targets,
        n_qubits=2,
        born_distribution=(("00", probabilities[0]), ("11", probabilities[1])),
        hilbert_state=None,
        privileged_target_available=False,
    )


def test_training_config_is_strict_and_topology_stays_zero() -> None:
    config = train_config()
    assert training_config_from_dict(training_config_to_dict(config)) == config
    assert len(build_epoch_plan(config)) == 2
    with pytest.raises(ValueError, match="topology"):
        train_config(topology_loss_weight=0.1)
    with pytest.raises(ValueError, match="topology"):
        train_config(loss=LossConfig(topology_weight=0.1))
    with pytest.raises(ValueError, match="not trainable"):
        CurriculumStageConfig(name="bad", epochs=1, tasks=("topology_audit",))


def test_collation_preserves_all_items_and_builds_geometry() -> None:
    examples = [example(0, (0.8, 0.2)), example(1, (0.3, 0.7))]
    batch = collate_training_examples(examples)
    assert batch.graph_count == 2
    assert batch.model_batch.graph.node_features.shape[0] == 4
    assert batch.model_batch.graph.edge_index.max().item() == 3
    assert batch.targets.born_prediction.probabilities.shape == (4,)
    assert batch.targets.geometry.pair_mask[0, 1]
    batches = deterministic_budget_batches(
        examples,
        train_config(batch_size=1),
        epoch_seed=99,
        shuffle=True,
    )
    assert sorted(item.view_item_id for group in batches for item in group) == ["item_0", "item_1"]


def test_forward_loss_and_gradient_are_finite() -> None:
    config = train_config()
    batch = collate_training_examples(
        [example(0, (0.8, 0.2)), example(1, (0.3, 0.7))]
    )
    model = TriQTOModel(model_config())
    output = model(batch.model_batch)
    losses = compute_supervised_losses(output, batch, config.loss)
    assert losses["topology"].item() == 0.0
    assert losses["total"].isfinite()
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_safe_checkpoint_roundtrip_restores_model_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    config = train_config()
    model_cfg = model_config()
    model = TriQTOModel(model_cfg)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = DeterministicLRScheduler(optimizer, config.scheduler, total_steps=4)
    batch = collate_training_examples([example(0, (0.8, 0.2))])
    loss = compute_supervised_losses(model(batch.model_batch), batch, config.loss)["total"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    recipe = training_recipe_id(
        "viewdataset_test",
        model.architecture_id,
        "modelconfig_test",
        config,
        data_spec().content_hash,
    )
    operational = training_operational_config_id(config)
    run_id = training_run_id(recipe, operational)
    path = tmp_path / "checkpoint.npz"
    random.seed(55)
    np.random.seed(55)
    torch.manual_seed(55)
    payload = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=config,
        model_config=model_cfg,
        data_spec=data_spec(),
        training_schema_id=training_schema_id(),
        training_recipe_id=recipe,
        operational_config_id=operational,
        training_run_id=run_id,
        epoch_completed=0,
        stage_index=0,
        global_step=1,
        best_validation_loss=1.0,
        best_epoch=0,
        kind="epoch",
    )
    with np.load(path, allow_pickle=False) as artifact:
        assert all(artifact[name].dtype.kind != "O" for name in artifact.files)
    restored_model = TriQTOModel(model_cfg)
    restored_optimizer = build_optimizer(restored_model, config.optimizer)
    restored_scheduler = DeterministicLRScheduler(
        restored_optimizer, config.scheduler, total_steps=4
    )
    metadata = load_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        restore_rng=True,
        expected_training_run_id=run_id,
    )
    assert metadata["checkpoint_id"] == payload["checkpoint_id"]
    assert restored_scheduler.step_index == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])
    corrupted = tmp_path / "corrupted.npz"
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    tensor_name = next(
        name
        for name, array in arrays.items()
        if name.startswith("tensor_") and array.size and array.dtype.kind == "f"
    )
    before = arrays[tensor_name].copy()
    arrays[tensor_name].reshape(-1)[0] += 1.0
    assert not np.array_equal(before, arrays[tensor_name])
    np.savez_compressed(corrupted, **arrays)
    with pytest.raises(ValueError, match="content hash"):
        load_training_checkpoint(corrupted)


def test_repository_phase14_configs_load() -> None:
    from triqto.training import load_training_config

    root = Path(__file__).resolve().parents[1]
    for name in ("phase14_base.yaml", "phase14_small_debug.yaml"):
        loaded = load_training_config(root / "configs" / "train" / name)
        assert loaded.topology_loss_weight == 0.0
        assert loaded.num_workers == 0



def test_distribution_losses_average_complete_measurement_setting_distances() -> None:
    predicted = torch.tensor(
        [0.5, 0.5, 0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        dtype=torch.float64,
    )
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    measurement_setting_index = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
    row_mask = torch.ones(6, dtype=torch.bool)
    kl, hellinger = _distribution_losses(
        predicted,
        target,
        row_mask,
        measurement_setting_index,
        distribution_count=2,
    )
    assert torch.allclose(kl, torch.log(torch.tensor(2.0, dtype=torch.float64)))
    expected_hellinger = torch.sqrt(
        torch.tensor(1.0 - 2.0 ** -0.5, dtype=torch.float64)
    )
    assert torch.allclose(hellinger, expected_hellinger)


def test_partial_gradient_accumulation_matches_equivalent_item_batches() -> None:
    optimizer_config = OptimizerConfig(
        name="sgd",
        learning_rate=0.01,
        weight_decay=0.0,
        momentum=0.0,
    )
    loss_config = LossConfig(geometry_weight=0.0, uncertainty_weighting=False)
    accumulated_config = train_config(
        gradient_accumulation_steps=2,
        optimizer=optimizer_config,
        loss=loss_config,
    )
    direct_config = train_config(
        gradient_accumulation_steps=1,
        optimizer=optimizer_config,
        loss=loss_config,
    )
    accumulated_model = TriQTOModel(model_config())
    direct_model = TriQTOModel(model_config())
    accumulated_optimizer = build_optimizer(accumulated_model, optimizer_config)
    direct_optimizer = build_optimizer(direct_model, optimizer_config)
    accumulated_scheduler = DeterministicLRScheduler(
        accumulated_optimizer,
        accumulated_config.scheduler,
        total_steps=2,
    )
    direct_scheduler = DeterministicLRScheduler(
        direct_optimizer,
        direct_config.scheduler,
        total_steps=2,
    )
    _, accumulated_steps, _, _, _ = _run_epoch(
        model=accumulated_model,
        batches=[
            [example(0, (0.8, 0.2))],
            [example(1, (0.3, 0.7))],
            [example(2, (0.6, 0.4))],
        ],
        config=accumulated_config,
        device=torch.device("cpu"),
        optimizer=accumulated_optimizer,
        scheduler=accumulated_scheduler,
        global_step=0,
    )
    _, direct_steps, _, _, _ = _run_epoch(
        model=direct_model,
        batches=[
            [example(0, (0.8, 0.2)), example(1, (0.3, 0.7))],
            [example(2, (0.6, 0.4))],
        ],
        config=direct_config,
        device=torch.device("cpu"),
        optimizer=direct_optimizer,
        scheduler=direct_scheduler,
        global_step=0,
    )
    assert accumulated_steps == direct_steps == 2
    for name, value in accumulated_model.state_dict().items():
        assert torch.allclose(value, direct_model.state_dict()[name], atol=2e-6, rtol=2e-6)


def test_training_output_rejects_phase12_and_phase7_overlap(tmp_path: Path) -> None:
    phase12 = tmp_path / "phase12"
    phase7 = tmp_path / "phase7"
    unrelated = tmp_path / "unrelated"
    phase12.mkdir()
    phase7.mkdir()
    unrelated.mkdir()
    with pytest.raises(ValueError, match="Phase 12 training-view source"):
        run_training(
            training_view_root=phase12,
            output_root=phase12 / "phase14",
            training_config=train_config(),
            model_config=model_config(),
        )
    with pytest.raises(ValueError, match="Phase 7 statevector source"):
        run_training(
            training_view_root=unrelated,
            output_root=phase7 / "phase14",
            training_config=train_config(),
            model_config=model_config(),
            phase7_root=phase7,
        )
