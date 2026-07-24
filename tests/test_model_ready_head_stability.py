from __future__ import annotations

import math

import torch
from torch import nn

from triqto.model import (
    ActionCandidateTensorBatch,
    TriQTOModel,
    TriQTOModelConfig,
)
from triqto.model.heads.distortion_head import DistortionHead
from triqto.training.model_ready.multitask_losses import _student_t_strength_nll


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

    output = head(
        torch.zeros((1, 16)),
        torch.zeros((2, 16)),
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([True], dtype=torch.bool),
    )
    scale = output.strength_log_scale.exp()
    assert torch.isfinite(scale).all()
    assert float(scale.detach().min()) >= 0.05 - 1.0e-6


def test_student_t_strength_likelihood_has_bounded_small_scale_influence() -> None:
    error = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    log_scale = torch.tensor(
        [math.log(0.05)], dtype=torch.float32, requires_grad=True
    )
    loss = _student_t_strength_nll(error, log_scale).mean()
    error_grad, scale_grad = torch.autograd.grad(loss, (error, log_scale))

    assert torch.isfinite(loss)
    assert torch.isfinite(error_grad).all()
    assert torch.isfinite(scale_grad).all()
    # The previous Gaussian objective produced a mean gradient of 400 here.
    assert float(error_grad.abs().max()) < 5.0
    assert float(scale_grad.abs().max()) <= 3.01


def test_full_model_heads_start_at_neutral_output_baselines() -> None:
    config = _config()
    model = TriQTOModel(config)

    diagnosis = model.distortion_head
    assert torch.equal(
        diagnosis.classifier.weight,
        torch.zeros_like(diagnosis.classifier.weight),
    )
    assert torch.equal(
        diagnosis.classifier.bias,
        torch.zeros_like(diagnosis.classifier.bias),
    )
    assert torch.equal(
        diagnosis.strength.weight,
        torch.zeros_like(diagnosis.strength.weight),
    )
    final_node = diagnosis.node_classifier[-1]
    assert isinstance(final_node, nn.Linear)
    assert torch.equal(final_node.weight, torch.zeros_like(final_node.weight))
    assert torch.equal(final_node.bias, torch.zeros_like(final_node.bias))

    diagnosis_output = diagnosis(
        torch.randn((1, config.hidden_dim)),
        torch.randn((2, config.hidden_dim)),
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([True], dtype=torch.bool),
    )
    assert torch.equal(
        diagnosis_output.class_logits,
        torch.zeros_like(diagnosis_output.class_logits),
    )
    assert torch.equal(
        diagnosis_output.strength_mean,
        torch.zeros_like(diagnosis_output.strength_mean),
    )
    assert torch.allclose(
        diagnosis_output.strength_log_scale.exp(),
        torch.full((1,), 0.5),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert torch.equal(
        diagnosis_output.affected_qubit_logits,
        torch.zeros_like(diagnosis_output.affected_qubit_logits),
    )

    action = model.action_ranking_head
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
    action_output = action(
        torch.randn((1, config.hidden_dim)),
        actions,
        torch.tensor([True], dtype=torch.bool),
    )
    assert torch.equal(action_output.should_act_logit, torch.zeros(1))
    assert torch.equal(
        action_output.should_act_probability,
        torch.full((1,), 0.5),
    )
    assert torch.equal(action_output.candidate_scores, torch.zeros(2))
    assert torch.equal(
        action_output.candidate_probabilities,
        torch.full((2,), 0.5),
    )
    assert torch.equal(action_output.predicted_rewards, torch.zeros(2))
