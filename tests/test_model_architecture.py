from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from triqto.model import (
    ActionCandidateTensorBatch,
    BornTensorBatch,
    DenseFeatureBatch,
    GraphTensorBatch,
    HilbertTensorBatch,
    OutcomeQueryTensorBatch,
    ParameterTensorBatch,
    TriQTOBatch,
    TriQTOModel,
    TriQTOModelConfig,
    architecture_manifest,
    model_architecture_id,
    model_config_id,
    load_model_config,
    model_config_from_dict,
    model_config_to_dict,
    state_dict_signature,
)
from triqto.model.encoders import HilbertEncoder
from triqto.model.losses import MultiTaskLossWeights, apply_phase13_topology_weight
from triqto.model.tensor_ops import segment_sum


@pytest.fixture(scope="module", autouse=True)
def _limit_torch_threads_for_contract_tests():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def config() -> TriQTOModelConfig:
    return TriQTOModelConfig(
        hidden_dim=32,
        graph_message_passing_layers=2,
        residual_mlp_layers=1,
        backend_input_dim=4,
        topology_input_dim=6,
        hilbert_deformation_dim=8,
        topology_prediction_dim=7,
        dropout=0.0,
        initialization_seed=17,
    )


def build_batch(*, born_probabilities=(0.5, 0.5, 0.25, 0.75), hardware_second=True) -> TriQTOBatch:
    cfg = config()
    graph = GraphTensorBatch(
        node_features=torch.tensor(
            [
                [float(i + j) / 10 for j in range(13)]
                for i in range(5)
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]], dtype=torch.long),
        edge_features=torch.tensor(
            [[float(i + j) / 20 for j in range(10)] for i in range(4)],
            dtype=torch.float32,
        ),
        edge_event_index=torch.tensor([1, 1, 3, 3], dtype=torch.long),
        gate_features=torch.tensor(
            [[float(i + j) / 30 for j in range(16)] for i in range(4)],
            dtype=torch.float32,
        ),
        gate_qubit_ptr=torch.tensor([0, 1, 3, 4, 6], dtype=torch.long),
        gate_qubit_indices=torch.tensor([0, 0, 1, 2, 3, 4], dtype=torch.long),
        node_batch=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        gate_batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        graph_count=2,
    )
    values = torch.tensor([0.2, 0.4, 0.1], dtype=torch.float32)
    parameter = ParameterTensorBatch(
        values=values,
        sin=torch.sin(values),
        cos=torch.cos(values),
        batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
        available_mask=torch.tensor([True, True]),
    )
    bits = torch.tensor(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]],
        dtype=torch.float32,
    )
    bit_mask = torch.tensor(
        [[True, True, False], [True, True, False], [True, True, True], [True, True, True]],
    )
    born = BornTensorBatch(
        outcome_bits=bits,
        outcome_bit_mask=bit_mask,
        probabilities=torch.tensor(born_probabilities, dtype=torch.float32),
        batch_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        available_mask=torch.tensor([True, True]),
        measurement_basis_codes=torch.zeros_like(bits, dtype=torch.long),
        measurement_setting_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    inv_sqrt_two = 2.0 ** -0.5
    hilbert = HilbertTensorBatch(
        amplitudes_real_imag=torch.tensor(
            [[inv_sqrt_two, 0], [0, 0], [0, 0], [inv_sqrt_two, 0]],
            dtype=torch.float32,
        ),
        basis_bits=torch.tensor(
            [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
            dtype=torch.float32,
        ),
        basis_bit_mask=torch.tensor([[True, True, False]] * 4),
        batch_index=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        available_mask=torch.tensor([True, False]),
    )
    backend = DenseFeatureBatch(
        features=torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]], dtype=torch.float32),
        available_mask=torch.tensor([True, False]),
    )
    topology = DenseFeatureBatch(
        features=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]],
            dtype=torch.float32,
        ),
        available_mask=torch.tensor([True, True]),
    )
    actions = ActionCandidateTensorBatch(
        candidate_features=torch.tensor(
            [[1, 0.1, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0.2, 1, 1, 0]],
            dtype=torch.float32,
        ),
        candidate_batch=torch.tensor([0, 0, 1], dtype=torch.long),
        candidate_available_mask=torch.tensor([True, True, True]),
        edit_type_ids=torch.tensor([1, 3], dtype=torch.long),
        edit_magnitudes=torch.tensor([0.2, -0.1], dtype=torch.float32),
        edit_qubit_positions=torch.tensor([0.0, 1.0], dtype=torch.float32),
        edit_candidate_index=torch.tensor([0, 2], dtype=torch.long),
    )
    queries = OutcomeQueryTensorBatch(
        outcome_bits=bits,
        outcome_bit_mask=bit_mask,
        batch_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        available_mask=torch.tensor([True, True]),
        measurement_basis_codes=torch.zeros_like(bits, dtype=torch.long),
        measurement_setting_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    hardware = torch.tensor([False, hardware_second])
    return TriQTOBatch(
        graph=graph,
        parameter=parameter,
        born=born,
        hilbert=hilbert,
        backend=backend,
        topology=topology,
        actions=actions,
        born_queries=queries,
        hardware_mode_mask=hardware,
        topology_hilbert_dependent_mask=torch.tensor([False, False]),
        head_stream_mask=torch.ones(2, 6, 7, dtype=torch.bool),
    )


def test_config_is_strict_and_identity_is_stable() -> None:
    cfg = config()
    assert model_config_from_dict(model_config_to_dict(cfg)) == cfg
    assert model_architecture_id(cfg) == model_architecture_id(cfg)
    changed_seed = TriQTOModelConfig(**{
        **model_config_to_dict(cfg),
        "model_name": "same_architecture_different_label",
        "initialization_seed": cfg.initialization_seed + 1,
    })
    assert model_architecture_id(changed_seed) == model_architecture_id(cfg)
    assert model_config_id(changed_seed) != model_config_id(cfg)
    with pytest.raises(ValueError, match="Unknown model config fields"):
        model_config_from_dict({**model_config_to_dict(cfg), "mystery": 1})
    with pytest.raises(TypeError, match="hidden_dim"):
        TriQTOModelConfig(hidden_dim=True)
    with pytest.raises(ValueError, match="even"):
        TriQTOModelConfig(hidden_dim=31)
    with pytest.raises(ValueError, match="topology_loss_weight"):
        TriQTOModelConfig(topology_loss_weight=0.1)


def test_variable_size_forward_and_segment_probabilities() -> None:
    cfg = config()
    model = TriQTOModel(cfg).eval()
    output = model(build_batch())
    assert output.graph_embedding.shape == (2, 32)
    assert output.node_embeddings.shape == (5, 32)
    assert output.stream_embeddings.shape == (2, 7, 32)
    assert output.head_latents.shape == (2, 6, 32)
    assert output.distortion.class_logits.shape == (2, 6)
    assert output.distortion.affected_qubit_logits.shape == (5,)
    action_sums = segment_sum(
        output.action_ranking.candidate_probabilities,
        output.action_ranking.candidate_batch,
        2,
    )
    assert torch.allclose(action_sums, torch.ones(2), atol=1e-6)
    born_sums = segment_sum(
        output.born_prediction.probabilities,
        output.born_prediction.measurement_setting_index,
        2,
    )
    assert torch.allclose(born_sums, torch.ones(2), atol=1e-6)
    assert output.topology.supervised_target_available_mask.tolist() == [False, False]


def test_born_prediction_cannot_observe_born_input() -> None:
    model = TriQTOModel(config()).eval()
    first = model(build_batch(born_probabilities=(0.5, 0.5, 0.25, 0.75)))
    second = model(build_batch(born_probabilities=(0.9, 0.1, 0.8, 0.2)))
    assert torch.allclose(
        first.born_prediction.probabilities,
        second.born_prediction.probabilities,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.allclose(
        first.distortion.class_logits,
        second.distortion.class_logits,
    )
    born_head_index = 2
    born_stream_index = 4
    assert not bool(first.effective_head_stream_mask[:, born_head_index, born_stream_index].any())


def test_hilbert_encoder_is_global_phase_invariant() -> None:
    cfg = config()
    encoder = HilbertEncoder(cfg).eval()
    batch = build_batch(hardware_second=False).hilbert
    assert batch is not None
    reference = torch.zeros(2, cfg.hidden_dim)
    first, mask = encoder(batch, 2, reference)
    angle = torch.tensor(0.73)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]],
        dtype=torch.float32,
    )
    rotated = copy.deepcopy(batch)
    rotated.amplitudes_real_imag = batch.amplitudes_real_imag @ rotation.T
    second, second_mask = encoder(rotated, 2, reference)
    assert torch.equal(mask, second_mask)
    assert torch.allclose(first, second, atol=2e-6, rtol=2e-6)


def test_hardware_mode_rejects_hilbert_and_hilbert_dependent_topology() -> None:
    batch = build_batch()
    assert batch.hilbert is not None
    batch.hilbert.available_mask = torch.tensor([True, True])
    batch.hilbert.batch_index = torch.tensor([0, 0, 0, 1])
    with pytest.raises(ValueError, match="Hardware-mode rows cannot expose Hilbert"):
        batch.validate(config())
    batch = build_batch()
    batch.topology_hilbert_dependent_mask = torch.tensor([False, True])
    with pytest.raises(ValueError, match="topology computed with Hilbert"):
        batch.validate(config())


def test_unavailable_dense_rows_must_be_zero() -> None:
    batch = build_batch()
    assert batch.backend is not None
    batch.backend.features[1, 0] = 1.0
    with pytest.raises(ValueError, match="unavailable rows"):
        batch.validate(config())


def test_deterministic_initialization_preserves_global_rng() -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = TriQTOModel(config())
    after = torch.random.get_rng_state().clone()
    second = TriQTOModel(config())
    assert torch.equal(before, after)
    assert state_dict_signature(first) == state_dict_signature(second)


def test_architecture_manifest_is_explicitly_untrained() -> None:
    manifest = architecture_manifest(TriQTOModel(config()))
    assert manifest["trained"] is False
    assert manifest["optimizer_state_present"] is False
    assert manifest["training_checkpoint"] is False
    assert manifest["topology_loss_weight"] == 0.0
    assert manifest["parameter_count"] > 0


def test_topology_loss_is_present_but_inactive() -> None:
    raw = torch.tensor(3.0, requires_grad=True)
    weighted = apply_phase13_topology_weight(raw)
    assert weighted.item() == 0.0
    with pytest.raises(ValueError, match="exactly 0.0"):
        apply_phase13_topology_weight(raw, 0.2)
    with pytest.raises(ValueError, match="exactly 0.0"):
        MultiTaskLossWeights(topology=0.1)


def test_inactive_heads_may_have_no_streams_and_are_zeroed() -> None:
    cfg = config()
    model = TriQTOModel(cfg).eval()
    batch = build_batch()
    head_active = torch.zeros(2, 6, dtype=torch.bool)
    head_active[:, 0] = True  # diagnosis only
    stream_mask = torch.zeros(2, 6, 7, dtype=torch.bool)
    stream_mask[:, 0, 0] = True  # graph
    stream_mask[:, 0, 4] = True  # Born evidence
    batch.head_active_mask = head_active
    batch.head_stream_mask = stream_mask
    output = model(batch)
    assert torch.all(output.head_latents[:, 1:, :] == 0)
    assert torch.all(output.fusion_weights[:, 1:, :] == 0)
    assert torch.all(output.action_ranking.candidate_probabilities == 0)
    assert not bool(output.action_ranking.graph_available_mask.any())
    assert torch.all(output.born_prediction.probabilities == 0)
    assert not bool(output.born_prediction.graph_available_mask.any())
    assert torch.all(output.hilbert_deformation.mean == 0)
    assert torch.all(output.uncertainty.log_variance == 0)
    assert torch.all(output.topology.feature_prediction == 0)


def test_active_head_without_permitted_stream_is_rejected() -> None:
    model = TriQTOModel(config()).eval()
    batch = build_batch()
    batch.head_active_mask = torch.zeros(2, 6, dtype=torch.bool)
    batch.head_active_mask[:, 2] = True
    batch.head_stream_mask = torch.zeros(2, 6, 7, dtype=torch.bool)
    with pytest.raises(ValueError, match="active head has no permitted"):
        model(batch)


def test_topology_prediction_head_cannot_copy_topology_input() -> None:
    model = TriQTOModel(config()).eval()
    output = model(build_batch())
    topology_head_index = 5
    topology_stream_index = 6
    assert not bool(
        output.effective_head_stream_mask[
            :, topology_head_index, topology_stream_index
        ].any()
    )


@pytest.mark.parametrize(
    "config_name",
    ("triqto_small_debug.yaml", "triqto_base.yaml", "triqto_full.yaml"),
)
def test_repository_model_configs_are_strict(config_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    loaded = load_model_config(root / "configs" / "model" / config_name)
    assert loaded.topology_loss_weight == 0.0
    assert loaded.hidden_dim % 2 == 0


def test_parameter_phasors_must_match_values() -> None:
    batch = build_batch()
    assert batch.parameter is not None
    batch.parameter.sin[0] = 0.0
    with pytest.raises(ValueError, match="must equal sin"):
        batch.validate(config())


def test_masked_basis_bits_and_duplicate_rows_are_rejected() -> None:
    batch = build_batch()
    assert batch.born is not None
    batch.born.outcome_bits[0, 2] = 1.0
    with pytest.raises(ValueError, match="masked bit positions"):
        batch.validate(config())
    batch = build_batch()
    assert batch.born_queries is not None
    batch.born_queries.outcome_bits[1] = batch.born_queries.outcome_bits[0]
    batch.born_queries.outcome_bit_mask[1] = batch.born_queries.outcome_bit_mask[0]
    with pytest.raises(ValueError, match="unique within each measurement setting"):
        batch.validate(config())


def test_measurement_setting_groups_cannot_span_graphs() -> None:
    batch = build_batch()
    assert batch.born_queries is not None
    batch.born_queries.measurement_setting_index[2:] = 0
    with pytest.raises(ValueError, match="must not span graphs"):
        batch.validate(config())


def test_masked_action_candidate_features_are_zero() -> None:
    batch = build_batch()
    assert batch.actions is not None
    batch.actions.candidate_available_mask[1] = False
    with pytest.raises(ValueError, match="features must be exactly zero"):
        batch.validate(config())
