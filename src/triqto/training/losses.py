"""Phase 14 supervised objective over strict Phase 13 outputs."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from triqto.model.constants import HEAD_ORDER
from triqto.model.outputs import TriQTOModelOutput
from triqto.model.tensor_ops import segment_softmax, segment_sum

from .config import LossConfig
from .models import SupervisedBatch


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("loss mask must be bool")
    if values.shape != mask.shape:
        raise ValueError("loss value and mask shapes must match")
    selected = values[mask]
    return selected.mean() if selected.numel() else _zero(values)


def _distribution_losses(
    predicted: Tensor,
    target: Tensor,
    row_mask: Tensor,
    outcome_batch: Tensor,
    graph_count: int,
) -> tuple[Tensor, Tensor]:
    """Average complete KL and Hellinger distances across active graphs."""
    if predicted.shape != target.shape or row_mask.shape != target.shape:
        raise ValueError("Born prediction, target, and mask shapes must match")
    if outcome_batch.shape != target.shape or outcome_batch.dtype != torch.long:
        raise ValueError("outcome_batch must be int64 with the distribution shape")
    if isinstance(graph_count, bool) or not isinstance(graph_count, int) or graph_count <= 0:
        raise ValueError("graph_count must be a positive integer")
    if outcome_batch.numel() and (
        int(outcome_batch.min()) < 0 or int(outcome_batch.max()) >= graph_count
    ):
        raise ValueError("outcome_batch contains a graph index out of range")
    if not bool(row_mask.any()):
        zero = _zero(predicted)
        return zero, zero

    epsilon = torch.finfo(predicted.dtype).tiny
    p = predicted.clamp_min(epsilon)
    q = target.clamp_min(0.0)
    kl_terms = torch.where(
        row_mask & (q > 0),
        q * (torch.log(q.clamp_min(epsilon)) - torch.log(p)),
        torch.zeros_like(q),
    )
    hellinger_sq_terms = torch.where(
        row_mask,
        0.5 * (torch.sqrt(p) - torch.sqrt(q)).square(),
        torch.zeros_like(q),
    )
    per_graph_kl = segment_sum(kl_terms, outcome_batch, graph_count)
    per_graph_hellinger_sq = segment_sum(
        hellinger_sq_terms,
        outcome_batch,
        graph_count,
    )
    active_graphs = segment_sum(
        row_mask.to(predicted.dtype),
        outcome_batch,
        graph_count,
    ) > 0
    return (
        per_graph_kl[active_graphs].mean(),
        torch.sqrt(
            per_graph_hellinger_sq[active_graphs].clamp_min(0.0)
        ).mean(),
    )


def _uncertainty_weight(loss: Tensor, log_variance: Tensor, active_mask: Tensor) -> Tensor:
    selected = log_variance[active_mask]
    if selected.numel() == 0:
        return loss
    scale = selected.mean()
    return torch.exp(-scale) * loss + scale


def compute_supervised_losses(
    output: TriQTOModelOutput,
    batch: SupervisedBatch,
    config: LossConfig,
    *,
    auxiliary_hilbert_output: TriQTOModelOutput | None = None,
) -> dict[str, Tensor]:
    """Return transparent scalar components and their exact total."""
    reference = output.graph_embedding
    diagnosis_target = batch.targets.diagnosis
    if bool(diagnosis_target.class_mask.any()):
        diagnosis_type = F.cross_entropy(
            output.distortion.class_logits[diagnosis_target.class_mask],
            diagnosis_target.class_index[diagnosis_target.class_mask],
        )
    else:
        diagnosis_type = _zero(reference)
    if bool(diagnosis_target.strength_mask.any()):
        indices = diagnosis_target.strength_mask
        error = output.distortion.strength_mean[indices] - diagnosis_target.strength[indices]
        log_scale = output.distortion.strength_log_scale[indices]
        diagnosis_strength = (0.5 * torch.exp(-2.0 * log_scale) * error.square() + log_scale).mean()
    else:
        diagnosis_strength = _zero(reference)
    if bool(diagnosis_target.affected_qubit_mask.any()):
        values = F.binary_cross_entropy_with_logits(
            output.distortion.affected_qubit_logits,
            diagnosis_target.affected_qubit.to(reference.dtype),
            reduction="none",
        )
        diagnosis_affected = _masked_mean(values, diagnosis_target.affected_qubit_mask)
    else:
        diagnosis_affected = _zero(reference)
    diagnosis = (
        config.diagnosis_type_weight * diagnosis_type
        + config.diagnosis_strength_weight * diagnosis_strength
        + config.diagnosis_affected_qubit_weight * diagnosis_affected
    )

    action_target = batch.targets.action
    action_mask = action_target.candidate_target_mask & output.action_ranking.candidate_available_mask
    if bool(action_mask.any()):
        selected = action_target.selected_mask & action_mask
        if int(selected.sum()) != int(output.action_ranking.graph_available_mask.sum()):
            raise ValueError("Each active action-ranking graph must have exactly one selected target")
        selected_probability = output.action_ranking.candidate_probabilities[selected]
        selected_weight = torch.where(
            action_target.privileged_oracle_mask[selected],
            torch.full_like(selected_probability, config.privileged_oracle_loss_weight),
            torch.ones_like(selected_probability),
        )
        action_selection = (
            -torch.log(selected_probability.clamp_min(1e-12)) * selected_weight
        ).sum() / selected_weight.sum().clamp_min(1e-12)

        target_logits = -action_target.rank.to(reference.dtype)
        target_distribution = segment_softmax(
            target_logits,
            action_target.candidate_batch,
            batch.graph_count,
            action_mask,
        )
        predicted = output.action_ranking.candidate_probabilities.clamp_min(1e-12)
        rank_terms = -target_distribution * torch.log(predicted)
        candidate_weight = torch.where(
            action_target.privileged_oracle_mask,
            torch.full_like(predicted, config.privileged_oracle_loss_weight),
            torch.ones_like(predicted),
        )
        action_rank = (rank_terms[action_mask] * candidate_weight[action_mask]).sum() / (
            target_distribution[action_mask] * candidate_weight[action_mask]
        ).sum().clamp_min(1e-12)
        reward_error = (
            output.action_ranking.predicted_rewards - action_target.reward
        ).square()
        action_reward = (
            reward_error[action_mask] * candidate_weight[action_mask]
        ).sum() / candidate_weight[action_mask].sum().clamp_min(1e-12)
    else:
        action_selection = action_rank = action_reward = _zero(reference)
    action = (
        config.action_selection_weight * action_selection
        + config.action_rank_distribution_weight * action_rank
        + config.action_reward_weight * action_reward
    )

    born_target = batch.targets.born_prediction
    if born_target.probabilities.numel():
        born_kl, born_hellinger = _distribution_losses(
            output.born_prediction.probabilities,
            born_target.probabilities,
            born_target.row_mask,
            born_target.outcome_batch,
            batch.graph_count,
        )
    else:
        born_kl = born_hellinger = _zero(reference)
    born = config.born_kl_weight * born_kl + config.born_hellinger_weight * born_hellinger

    hilbert_target = batch.targets.hilbert_to_born
    if bool(hilbert_target.row_mask.any()):
        if auxiliary_hilbert_output is None:
            raise ValueError("Hilbert-to-Born targets require an auxiliary forward output")
        hilbert_kl, hilbert_hellinger = _distribution_losses(
            auxiliary_hilbert_output.born_prediction.probabilities,
            hilbert_target.probabilities,
            hilbert_target.row_mask,
            hilbert_target.outcome_batch,
            batch.graph_count,
        )
        hilbert_to_born = config.hilbert_to_born_weight * (hilbert_kl + hilbert_hellinger)
    else:
        hilbert_kl = hilbert_hellinger = hilbert_to_born = _zero(reference)

    geometry_target = batch.targets.geometry
    if bool(geometry_target.pair_mask.any()):
        active = batch.model_batch.resolved_head_active_mask().to(reference.dtype).unsqueeze(2)
        latent = (output.head_latents * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        latent = F.normalize(latent, p=2, dim=1)
        predicted_distance = torch.cdist(latent, latent, p=2) / 2.0
        geometry = _masked_mean(
            (predicted_distance - geometry_target.target_distance).square(),
            geometry_target.pair_mask,
        ) * config.geometry_weight
    else:
        geometry = _zero(reference)

    if config.uncertainty_weighting:
        active = batch.model_batch.resolved_head_active_mask()
        uncertainty = output.uncertainty.log_variance
        diagnosis = _uncertainty_weight(diagnosis, uncertainty[:, 0], active[:, HEAD_ORDER.index("diagnosis")])
        action = _uncertainty_weight(action, uncertainty[:, 1], active[:, HEAD_ORDER.index("action_ranking")])
        born = _uncertainty_weight(born, uncertainty[:, 2], active[:, HEAD_ORDER.index("born_prediction")])
        if bool(hilbert_target.row_mask.any()):
            hilbert_to_born = _uncertainty_weight(
                hilbert_to_born,
                uncertainty[:, 3],
                batch.auxiliary_hilbert_to_born_batch.resolved_head_active_mask()[:, HEAD_ORDER.index("born_prediction")],
            )

    topology = _zero(reference)
    total = diagnosis + action + born + hilbert_to_born + geometry + topology
    components = {
        "diagnosis_type": diagnosis_type,
        "diagnosis_strength": diagnosis_strength,
        "diagnosis_affected_qubit": diagnosis_affected,
        "diagnosis": diagnosis,
        "action_selection": action_selection,
        "action_rank_distribution": action_rank,
        "action_reward": action_reward,
        "action": action,
        "born_kl": born_kl,
        "born_hellinger": born_hellinger,
        "born_prediction": born,
        "hilbert_to_born_kl": hilbert_kl,
        "hilbert_to_born_hellinger": hilbert_hellinger,
        "hilbert_to_born": hilbert_to_born,
        "geometry": geometry,
        "topology": topology,
        "total": total,
    }
    for name, value in components.items():
        if value.ndim != 0 or not torch.isfinite(value):
            raise FloatingPointError(f"Loss component {name} is not a finite scalar")
    return components


__all__ = ["compute_supervised_losses"]
