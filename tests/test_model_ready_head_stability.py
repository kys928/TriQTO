from __future__ import annotations

import torch

from triqto.model import (
    ActionCandidateTensorBatch,
    TriQTOModel,
    TriQTOModelConfig,
)
from triqto.model.heads.distortion_head import DistortionHead


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


def test_diagnosis_strength_scale_has_stable_positive_floor() -> None:
    head = DistortionHead(_config())
    with torch.no_grad():
        head.strength.weight.zero_()
        head.strength.bias.zero_()
        head.strength.bias[1] = -1000.0

    graph = torch.zeros((1, 16), requires_grad=True)
    nodes = torch.zeros((2, 16), requires_grad=True)
    output = head(
        graph,
        nodes,
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([True], dtype=torch.bool),
    )
    scale = output.strength_log_scale.exp()
    assert torch.isfinite(scale).all()
    assert float(scale.detach().min()) >= 0.05 - 1.0e-6

    target = torch.ones_like(output.strength_mean)
    error = output.strength_mean - target
    loss = (
        0.5 * torch.exp(-2.0 * output.strength_log_scale) * error.square()
        + output.strength_log_scale
    ).mean()
    loss.backward()
    assert graph.grad is not None
    assert torch.isfinite(graph.grad).all()


def test_full_model_reward_head_starts_at_zero_baseline() -> None:
    config = _config()
    model = TriQTOModel(config)
    head = model.action_ranking_head
    actions = ActionCandidateTensorBatch(
        candidate_features=torch.zeros(
            (2, config.action_candidate_feature_dim), dtype=torch.float32
        ),
        candidate_batch=torch.tensor([0, 0], dtype=torch.long),
        candidate_available_mask=torch.tensor([True, True], dtype=torch.bool),
        edit_type_ids=torch.zeros(0, dtype=torch.long),
        edit_magnitudes=torch.zeros(0, dtype=torch.float32),
        edit_qubit_positions=torch.zeros(0, dtype=torch.float32),
        edit_candidate_index=torch.zeros(0, dtype=torch.long),
    )
    output = head(
        torch.randn((1, config.hidden_dim)),
        actions,
        torch.tensor([True], dtype=torch.bool),
    )
    assert torch.equal(output.predicted_rewards, torch.zeros(2))
    assert torch.equal(head.reward.weight, torch.zeros_like(head.reward.weight))
    assert torch.equal(head.reward.bias, torch.zeros_like(head.reward.bias))
