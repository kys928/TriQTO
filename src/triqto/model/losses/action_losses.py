"""Listwise action-ranking losses for future Phase 14 use."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from triqto.model.tensor_ops import segment_softmax

from .task_losses import masked_mse_loss


@dataclass(slots=True)
class ActionLosses:
    listwise_selection: Tensor
    reward_regression: Tensor

    @property
    def total(self) -> Tensor:
        return self.listwise_selection + self.reward_regression


def action_ranking_losses(
    *,
    candidate_scores: Tensor,
    candidate_batch: Tensor,
    candidate_available_mask: Tensor,
    selected_target_mask: Tensor,
    predicted_rewards: Tensor,
    reward_targets: Tensor,
    reward_target_mask: Tensor,
    graph_count: int,
) -> ActionLosses:
    if selected_target_mask.dtype != torch.bool or selected_target_mask.shape != candidate_scores.shape:
        raise ValueError("selected_target_mask must be bool with candidate score shape")
    probabilities = segment_softmax(
        candidate_scores,
        candidate_batch,
        graph_count,
        candidate_available_mask,
    )
    selected_probability = probabilities[selected_target_mask]
    if selected_probability.numel() == 0:
        listwise = candidate_scores.sum() * 0.0
    else:
        listwise = -torch.log(selected_probability.clamp_min(1e-12)).mean()
    reward = masked_mse_loss(
        predicted_rewards,
        reward_targets,
        reward_target_mask,
    )
    return ActionLosses(listwise_selection=listwise, reward_regression=reward)


__all__ = ["ActionLosses", "action_ranking_losses"]
