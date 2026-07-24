"""Full model-ready multi-task objective with transparent masked components."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from triqto.model.constants import HEAD_ORDER
from triqto.model.outputs import TriQTOModelOutput
from triqto.model.tensor_ops import segment_sum
from triqto.training.config import LossConfig

from .losses import compute_model_ready_action_losses
from .multitask_types import ModelReadySupervisedBatch


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if mask.dtype != torch.bool or values.shape != mask.shape:
        raise ValueError("masked mean requires shape-matched boolean mask")
    selected = values[mask]
    return selected.mean() if selected.numel() else _zero(values)


def _distribution_losses(
    predicted: Tensor,
    target: Tensor,
    row_mask: Tensor,
    outcome_batch: Tensor,
    graph_count: int,
) -> tuple[Tensor, Tensor]:
    if predicted.shape != target.shape or row_mask.shape != target.shape:
        raise ValueError("Born prediction, target, and mask shapes must match")
    if outcome_batch.shape != target.shape or outcome_batch.dtype != torch.long:
        raise ValueError("Born outcome_batch contract mismatch")
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
    hellinger_sq = torch.where(
        row_mask,
        0.5 * (torch.sqrt(p) - torch.sqrt(q)).square(),
        torch.zeros_like(q),
    )
    per_graph_kl = segment_sum(kl_terms, outcome_batch, graph_count)
    per_graph_h2 = segment_sum(hellinger_sq, outcome_batch, graph_count)
    active = segment_sum(
        row_mask.to(predicted.dtype), outcome_batch, graph_count
    ) > 0
    return (
        per_graph_kl[active].mean(),
        torch.sqrt(per_graph_h2[active].clamp_min(0.0)).mean(),
    )


def _uncertainty_weight(
    loss: Tensor, log_variance: Tensor, active_mask: Tensor
) -> Tensor:
    selected = log_variance[active_mask]
    if selected.numel() == 0:
        return loss
    scale = selected.mean()
    return torch.exp(-scale) * loss + scale


def compute_model_ready_multitask_losses(
    output: TriQTOModelOutput,
    batch: ModelReadySupervisedBatch,
    config: LossConfig,
) -> dict[str, Tensor]:
    """Compute diagnosis, two-stage action, Born, geometry, and zero topology loss."""
    reference = output.graph_embedding
    graph_count = batch.graph_count

    diagnosis_target = batch.targets.diagnosis
    if bool(diagnosis_target.class_mask.any()):
        diagnosis_type = F.cross_entropy(
            output.distortion.class_logits[diagnosis_target.class_mask],
            diagnosis_target.class_index[diagnosis_target.class_mask],
        )
    else:
        diagnosis_type = _zero(reference)
    if bool(diagnosis_target.strength_mask.any()):
        active = diagnosis_target.strength_mask
        error = output.distortion.strength_mean[active] - diagnosis_target.strength[active]
        log_scale = output.distortion.strength_log_scale[active]
        diagnosis_strength = (
            0.5 * torch.exp(-2.0 * log_scale) * error.square() + log_scale
        ).mean()
    else:
        diagnosis_strength = _zero(reference)
    if bool(diagnosis_target.affected_qubit_mask.any()):
        values = F.binary_cross_entropy_with_logits(
            output.distortion.affected_qubit_logits,
            diagnosis_target.affected_qubit.to(reference.dtype),
            reduction="none",
        )
        diagnosis_affected = _masked_mean(
            values, diagnosis_target.affected_qubit_mask
        )
    else:
        diagnosis_affected = _zero(reference)
    diagnosis = (
        config.diagnosis_type_weight * diagnosis_type
        + config.diagnosis_strength_weight * diagnosis_strength
        + config.diagnosis_affected_qubit_weight * diagnosis_affected
    )

    action_parts = compute_model_ready_action_losses(
        output,
        batch.targets.action,
        should_act_weight=config.action_selection_weight,
        ranking_weight=config.action_rank_distribution_weight,
        reward_weight=config.action_reward_weight,
    )
    action = action_parts["total"]

    born_target = batch.targets.born_prediction
    if born_target.probabilities.numel():
        born_kl, born_hellinger = _distribution_losses(
            output.born_prediction.probabilities,
            born_target.probabilities,
            born_target.row_mask,
            born_target.outcome_batch,
            graph_count,
        )
    else:
        born_kl = born_hellinger = _zero(reference)
    born = (
        config.born_kl_weight * born_kl
        + config.born_hellinger_weight * born_hellinger
    )

    geometry_target = batch.targets.geometry
    if bool(geometry_target.pair_mask.any()):
        active_heads = batch.model_batch.resolved_head_active_mask().to(
            reference.dtype
        ).unsqueeze(2)
        latent = (output.head_latents * active_heads).sum(dim=1) / active_heads.sum(
            dim=1
        ).clamp_min(1.0)
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
        diagnosis = _uncertainty_weight(
            diagnosis,
            uncertainty[:, 0],
            active[:, HEAD_ORDER.index("diagnosis")],
        )
        action = _uncertainty_weight(
            action,
            uncertainty[:, 1],
            batch.targets.action.should_act_mask,
        )
        born_graph_mask = torch.zeros(graph_count, dtype=torch.bool, device=reference.device)
        if born_target.row_mask.numel():
            counts = segment_sum(
                born_target.row_mask.to(reference.dtype),
                born_target.outcome_batch,
                graph_count,
            )
            born_graph_mask = counts > 0
        born = _uncertainty_weight(
            born,
            uncertainty[:, 2],
            born_graph_mask,
        )

    topology = _zero(reference)
    total = diagnosis + action + born + geometry + topology
    components = {
        "diagnosis_type": diagnosis_type,
        "diagnosis_strength": diagnosis_strength,
        "diagnosis_affected_qubit": diagnosis_affected,
        "diagnosis": diagnosis,
        "action_should_act": action_parts["action_should_act"],
        "action_rank_distribution": action_parts["action_rank_distribution"],
        "action_reward": action_parts["action_reward"],
        "action": action,
        "born_kl": born_kl,
        "born_hellinger": born_hellinger,
        "born_prediction": born,
        "geometry": geometry,
        "topology": topology,
        "total": total,
    }
    for name, value in components.items():
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise FloatingPointError(
                f"model-ready multi-task loss {name} is not a finite scalar"
            )
    return components


__all__ = ["compute_model_ready_multitask_losses"]
