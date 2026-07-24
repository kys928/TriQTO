"""Two-stage action losses for immutable model-ready targets."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from triqto.model.outputs import TriQTOModelOutput

from .types import ModelReadyActionTargets


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def compute_model_ready_action_losses(
    output: TriQTOModelOutput,
    targets: ModelReadyActionTargets,
    *,
    should_act_weight: float = 1.0,
    ranking_weight: float = 1.0,
    reward_weight: float = 0.25,
) -> dict[str, Tensor]:
    """Compute weighted gate loss and conditionally active candidate losses.

    ``y_should_act_weight`` supplies the per-example class-balancing factor.
    Candidate ranking and reward regression contribute only for graphs whose
    ``y_ranking_loss_mask`` is true. The topology objective remains structurally
    present and exactly zero.
    """
    if not isinstance(targets, ModelReadyActionTargets):
        raise TypeError("targets must be ModelReadyActionTargets")
    reference = output.graph_embedding
    graph_count = reference.shape[0]
    action = output.action_ranking

    for name, value in (
        ("should_act", targets.should_act),
        ("should_act_weight", targets.should_act_weight),
        ("should_act_mask", targets.should_act_mask),
        ("ranking_loss_mask", targets.ranking_loss_mask),
    ):
        if value.shape != (graph_count,):
            raise ValueError(f"{name} must have shape [{graph_count}]")
    if targets.should_act_mask.dtype != torch.bool:
        raise TypeError("should_act_mask must be bool")
    if targets.ranking_loss_mask.dtype != torch.bool:
        raise TypeError("ranking_loss_mask must be bool")
    if action.should_act_logit.shape != (graph_count,):
        raise ValueError("should_act_logit shape does not match graph count")
    if action.should_act_available_mask.shape != (graph_count,):
        raise ValueError("should_act_available_mask shape does not match graph count")
    if bool((targets.should_act_weight <= 0).any()) or not bool(
        torch.isfinite(targets.should_act_weight).all()
    ):
        raise ValueError("should_act weights must be finite and positive")

    gate_mask = targets.should_act_mask & action.should_act_available_mask
    if bool(gate_mask.any()):
        gate_terms = F.binary_cross_entropy_with_logits(
            action.should_act_logit,
            targets.should_act.to(action.should_act_logit.dtype),
            reduction="none",
        )
        gate_weights = targets.should_act_weight.to(gate_terms.dtype)
        gate_loss = (
            gate_terms[gate_mask] * gate_weights[gate_mask]
        ).sum() / gate_weights[gate_mask].sum().clamp_min(1.0e-12)
    else:
        gate_loss = _zero(reference)

    candidate_count = action.candidate_scores.numel()
    candidate_tensors = (
        targets.candidate_reward,
        targets.candidate_rank,
        targets.candidate_selected_mask,
        targets.candidate_listwise_distribution,
        targets.candidate_target_mask,
        targets.candidate_batch,
    )
    if any(value.shape != (candidate_count,) for value in candidate_tensors):
        raise ValueError("candidate target shapes do not match model output")
    if targets.candidate_target_mask.dtype != torch.bool:
        raise TypeError("candidate_target_mask must be bool")
    if targets.candidate_selected_mask.dtype != torch.bool:
        raise TypeError("candidate_selected_mask must be bool")
    if targets.candidate_batch.dtype != torch.long:
        raise TypeError("candidate_batch must be int64")
    if candidate_count and (
        int(targets.candidate_batch.min()) < 0
        or int(targets.candidate_batch.max()) >= graph_count
    ):
        raise ValueError("candidate_batch contains an out-of-range graph index")
    if candidate_count and not torch.equal(
        targets.candidate_batch, action.candidate_batch
    ):
        raise ValueError("target/model candidate_batch mappings differ")

    rank_numerator = _zero(reference)
    reward_numerator = _zero(reference)
    active_rank_graphs = 0
    for graph in range(graph_count):
        if not bool(targets.ranking_loss_mask[graph]):
            continue
        if not bool(targets.should_act[graph] > 0.5):
            raise ValueError("ranking loss is active for a no-action target")
        graph_candidates = targets.candidate_batch == graph
        candidate_mask = (
            graph_candidates
            & targets.candidate_target_mask
            & action.candidate_available_mask
        )
        if not bool(candidate_mask.any()):
            raise ValueError("ranking-active graph has no eligible deployable candidates")
        distribution = targets.candidate_listwise_distribution[candidate_mask].to(
            reference.dtype
        )
        if bool((distribution < 0).any()) or not bool(torch.isfinite(distribution).all()):
            raise ValueError("candidate listwise distribution is invalid")
        mass = distribution.sum()
        if not bool(mass > 0):
            raise ValueError("ranking-active listwise distribution has zero mass")
        distribution = distribution / mass
        predicted = action.candidate_probabilities[candidate_mask].clamp_min(1.0e-12)
        rank_numerator = rank_numerator + (
            -distribution * torch.log(predicted)
        ).sum()
        reward_error = (
            action.predicted_rewards[candidate_mask]
            - targets.candidate_reward[candidate_mask].to(reference.dtype)
        ).square()
        reward_numerator = reward_numerator + reward_error.mean()
        active_rank_graphs += 1

    if active_rank_graphs:
        ranking_loss = rank_numerator / active_rank_graphs
        candidate_reward_loss = reward_numerator / active_rank_graphs
    else:
        ranking_loss = _zero(reference)
        candidate_reward_loss = _zero(reference)

    topology = _zero(reference)
    total = (
        float(should_act_weight) * gate_loss
        + float(ranking_weight) * ranking_loss
        + float(reward_weight) * candidate_reward_loss
        + topology
    )
    result = {
        "action_should_act": gate_loss,
        "action_rank_distribution": ranking_loss,
        "action_reward": candidate_reward_loss,
        "topology": topology,
        "total": total,
    }
    for name, value in result.items():
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise FloatingPointError(f"loss component {name} is not a finite scalar")
    return result


__all__ = ["compute_model_ready_action_losses"]
