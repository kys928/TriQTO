"""Phase 14 supervised objective over strict Phase 13 outputs."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from triqto.model.constants import UNCERTAINTY_TARGETS
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


def _active_mean(values: Tensor, active_mask: Tensor) -> Tensor:
    if values.ndim != 1 or active_mask.shape != values.shape:
        raise ValueError("per-example loss values and active mask must be one-dimensional")
    if active_mask.dtype != torch.bool:
        raise TypeError("per-example active mask must be bool")
    selected = values[active_mask]
    return selected.mean() if selected.numel() else _zero(values)


def _heteroscedastic_mean(
    per_example_loss: Tensor,
    log_variance: Tensor,
    active_mask: Tensor,
) -> Tensor:
    """Apply a distinct heteroscedastic likelihood weight to every example."""
    if per_example_loss.shape != log_variance.shape:
        raise ValueError("per-example loss and log_variance shapes must match")
    if per_example_loss.ndim != 1 or active_mask.shape != per_example_loss.shape:
        raise ValueError("heteroscedastic inputs must be one-dimensional")
    selected_loss = per_example_loss[active_mask]
    selected_log_variance = log_variance[active_mask]
    if selected_loss.numel() == 0:
        return _zero(per_example_loss)
    return (
        torch.exp(-selected_log_variance) * selected_loss
        + selected_log_variance
    ).mean()


def _distribution_losses(
    predicted: Tensor,
    target: Tensor,
    row_mask: Tensor,
    distribution_index: Tensor,
    distribution_count: int,
) -> tuple[Tensor, Tensor]:
    """Average complete KL and Hellinger distances across active distributions."""
    if predicted.shape != target.shape or row_mask.shape != target.shape:
        raise ValueError("Born prediction, target, and mask shapes must match")
    if distribution_index.shape != target.shape or distribution_index.dtype != torch.long:
        raise ValueError("distribution_index must be int64 with the distribution shape")
    if (
        isinstance(distribution_count, bool)
        or not isinstance(distribution_count, int)
        or distribution_count <= 0
    ):
        raise ValueError("distribution_count must be a positive integer")
    if distribution_index.numel() and (
        int(distribution_index.min()) < 0
        or int(distribution_index.max()) >= distribution_count
    ):
        raise ValueError("distribution_index contains an index out of range")
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
    per_distribution_kl = segment_sum(
        kl_terms,
        distribution_index,
        distribution_count,
    )
    per_distribution_hellinger_sq = segment_sum(
        hellinger_sq_terms,
        distribution_index,
        distribution_count,
    )
    active_distributions = segment_sum(
        row_mask.to(predicted.dtype),
        distribution_index,
        distribution_count,
    ) > 0
    return (
        per_distribution_kl[active_distributions].mean(),
        torch.sqrt(
            per_distribution_hellinger_sq[active_distributions].clamp_min(0.0)
        ).mean(),
    )


def _distribution_losses_by_graph(
    predicted: Tensor,
    target: Tensor,
    row_mask: Tensor,
    distribution_index: Tensor,
    outcome_batch: Tensor,
    graph_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return setting-complete KL/Hellinger means for every graph."""
    if distribution_index.numel() == 0:
        zero = predicted.new_zeros(graph_count)
        return zero, zero, torch.zeros(graph_count, dtype=torch.bool, device=predicted.device)
    distribution_count = int(distribution_index.max()) + 1
    if outcome_batch.shape != target.shape or outcome_batch.dtype != torch.long:
        raise ValueError("outcome_batch must be int64 with the distribution shape")
    if outcome_batch.numel() and (
        int(outcome_batch.min()) < 0 or int(outcome_batch.max()) >= graph_count
    ):
        raise ValueError("outcome_batch contains a graph index out of range")

    epsilon = torch.finfo(predicted.dtype).tiny
    p = predicted.clamp_min(epsilon)
    q = target.clamp_min(0.0)
    kl_terms = torch.where(
        row_mask & (q > 0),
        q * (torch.log(q.clamp_min(epsilon)) - torch.log(p)),
        torch.zeros_like(q),
    )
    hellinger_terms = torch.where(
        row_mask,
        0.5 * (torch.sqrt(p) - torch.sqrt(q)).square(),
        torch.zeros_like(q),
    )
    setting_kl = segment_sum(kl_terms, distribution_index, distribution_count)
    setting_hellinger = torch.sqrt(
        segment_sum(hellinger_terms, distribution_index, distribution_count).clamp_min(0.0)
    )
    setting_active = (
        segment_sum(row_mask.to(predicted.dtype), distribution_index, distribution_count)
        > 0
    )
    setting_owner = torch.zeros(
        distribution_count,
        dtype=torch.long,
        device=predicted.device,
    )
    for setting in range(distribution_count):
        rows = (distribution_index == setting) & row_mask
        if not bool(rows.any()):
            continue
        owners = outcome_batch[rows].unique()
        if owners.numel() != 1:
            raise ValueError("one measurement distribution must not span graphs")
        setting_owner[setting] = owners[0]
    active_float = setting_active.to(predicted.dtype)
    per_graph_count = segment_sum(active_float, setting_owner, graph_count)
    per_graph_kl = segment_sum(
        setting_kl * active_float,
        setting_owner,
        graph_count,
    ) / per_graph_count.clamp_min(1.0)
    per_graph_hellinger = segment_sum(
        setting_hellinger * active_float,
        setting_owner,
        graph_count,
    ) / per_graph_count.clamp_min(1.0)
    return per_graph_kl, per_graph_hellinger, per_graph_count > 0


def compute_supervised_losses(
    output: TriQTOModelOutput,
    batch: SupervisedBatch,
    config: LossConfig,
    *,
    auxiliary_hilbert_output: TriQTOModelOutput | None = None,
) -> dict[str, Tensor]:
    """Return transparent scalar components and their exact total."""
    reference = output.graph_embedding
    graph_count = batch.graph_count
    diagnosis_target = batch.targets.diagnosis
    diagnosis_per_graph = reference.new_zeros(graph_count)
    diagnosis_active = torch.zeros(
        graph_count,
        dtype=torch.bool,
        device=reference.device,
    )
    if bool(diagnosis_target.class_mask.any()):
        class_values = F.cross_entropy(
            output.distortion.class_logits[diagnosis_target.class_mask],
            diagnosis_target.class_index[diagnosis_target.class_mask],
            reduction="none",
        )
        diagnosis_type = class_values.mean()
        diagnosis_per_graph[diagnosis_target.class_mask] += (
            config.diagnosis_type_weight * class_values
        )
        diagnosis_active |= diagnosis_target.class_mask
    else:
        diagnosis_type = _zero(reference)
    if bool(diagnosis_target.strength_mask.any()):
        indices = diagnosis_target.strength_mask
        error = output.distortion.strength_mean[indices] - diagnosis_target.strength[indices]
        log_scale = output.distortion.strength_log_scale[indices]
        strength_values = (
            0.5 * torch.exp(-2.0 * log_scale) * error.square() + log_scale
        )
        diagnosis_strength = strength_values.mean()
        diagnosis_per_graph[indices] += (
            config.diagnosis_strength_weight * strength_values
        )
        diagnosis_active |= indices
    else:
        diagnosis_strength = _zero(reference)
    if bool(diagnosis_target.affected_qubit_mask.any()):
        values = F.binary_cross_entropy_with_logits(
            output.distortion.affected_qubit_logits,
            diagnosis_target.affected_qubit.to(reference.dtype),
            reduction="none",
        )
        diagnosis_affected = _masked_mean(values, diagnosis_target.affected_qubit_mask)
        node_batch = batch.model_batch.graph.node_batch
        affected_mask = diagnosis_target.affected_qubit_mask
        affected_sum = segment_sum(
            values * affected_mask.to(values.dtype),
            node_batch,
            graph_count,
        )
        affected_count = segment_sum(
            affected_mask.to(values.dtype),
            node_batch,
            graph_count,
        )
        affected_active = affected_count > 0
        diagnosis_per_graph += config.diagnosis_affected_qubit_weight * (
            affected_sum / affected_count.clamp_min(1.0)
        )
        diagnosis_active |= affected_active
    else:
        diagnosis_affected = _zero(reference)

    action_target = batch.targets.action
    action_mask = action_target.candidate_target_mask & output.action_ranking.candidate_available_mask
    action_per_graph = reference.new_zeros(graph_count)
    action_active = segment_sum(
        action_mask.to(reference.dtype),
        action_target.candidate_batch,
        graph_count,
    ) > 0
    if bool(action_mask.any()):
        selected = action_target.selected_mask & action_mask
        selected_count = segment_sum(
            selected.to(reference.dtype),
            action_target.candidate_batch,
            graph_count,
        )
        if not torch.equal(
            selected_count[action_active],
            torch.ones_like(selected_count[action_active]),
        ):
            raise ValueError("Each active action-ranking graph must have exactly one selected target")
        predicted = output.action_ranking.candidate_probabilities.clamp_min(1e-12)
        candidate_weight = torch.where(
            action_target.privileged_oracle_mask,
            torch.full_like(predicted, config.privileged_oracle_loss_weight),
            torch.ones_like(predicted),
        )
        selection_numerator = segment_sum(
            -torch.log(predicted) * candidate_weight * selected.to(reference.dtype),
            action_target.candidate_batch,
            graph_count,
        )
        selection_denominator = segment_sum(
            candidate_weight * selected.to(reference.dtype),
            action_target.candidate_batch,
            graph_count,
        )
        selection_by_graph = selection_numerator / selection_denominator.clamp_min(1e-12)
        action_selection = _active_mean(selection_by_graph, action_active)

        target_logits = -action_target.rank.to(reference.dtype)
        target_distribution = segment_softmax(
            target_logits,
            action_target.candidate_batch,
            batch.graph_count,
            action_mask,
        )
        rank_terms = -target_distribution * torch.log(predicted)
        rank_numerator = segment_sum(
            rank_terms * candidate_weight * action_mask.to(reference.dtype),
            action_target.candidate_batch,
            graph_count,
        )
        rank_denominator = segment_sum(
            target_distribution * candidate_weight * action_mask.to(reference.dtype),
            action_target.candidate_batch,
            graph_count,
        )
        rank_by_graph = rank_numerator / rank_denominator.clamp_min(1e-12)
        action_rank = _active_mean(rank_by_graph, action_active)
        reward_error = (
            output.action_ranking.predicted_rewards - action_target.reward
        ).square()
        reward_numerator = segment_sum(
            reward_error * candidate_weight * action_mask.to(reference.dtype),
            action_target.candidate_batch,
            graph_count,
        )
        reward_denominator = segment_sum(
            candidate_weight * action_mask.to(reference.dtype),
            action_target.candidate_batch,
            graph_count,
        )
        reward_by_graph = reward_numerator / reward_denominator.clamp_min(1e-12)
        action_reward = _active_mean(reward_by_graph, action_active)
        action_per_graph = (
            config.action_selection_weight * selection_by_graph
            + config.action_rank_distribution_weight * rank_by_graph
            + config.action_reward_weight * reward_by_graph
        )
    else:
        action_selection = action_rank = action_reward = _zero(reference)

    born_target = batch.targets.born_prediction
    if born_target.probabilities.numel():
        setting_index = output.born_prediction.measurement_setting_index
        born_kl_by_graph, born_hellinger_by_graph, born_active = (
            _distribution_losses_by_graph(
                output.born_prediction.probabilities,
                born_target.probabilities,
                born_target.row_mask,
                setting_index,
                born_target.outcome_batch,
                graph_count,
            )
        )
        born_kl = _active_mean(born_kl_by_graph, born_active)
        born_hellinger = _active_mean(born_hellinger_by_graph, born_active)
        born_per_graph = (
            config.born_kl_weight * born_kl_by_graph
            + config.born_hellinger_weight * born_hellinger_by_graph
        )
    else:
        born_kl = born_hellinger = _zero(reference)
        born_per_graph = reference.new_zeros(graph_count)
        born_active = torch.zeros(graph_count, dtype=torch.bool, device=reference.device)

    hilbert_target = batch.targets.hilbert_to_born
    if bool(hilbert_target.row_mask.any()):
        if auxiliary_hilbert_output is None:
            raise ValueError("Hilbert-to-Born targets require an auxiliary forward output")
        setting_index = auxiliary_hilbert_output.born_prediction.measurement_setting_index
        hilbert_kl_by_graph, hilbert_hellinger_by_graph, hilbert_active = (
            _distribution_losses_by_graph(
                auxiliary_hilbert_output.born_prediction.probabilities,
                hilbert_target.probabilities,
                hilbert_target.row_mask,
                setting_index,
                hilbert_target.outcome_batch,
                graph_count,
            )
        )
        hilbert_kl = _active_mean(hilbert_kl_by_graph, hilbert_active)
        hilbert_hellinger = _active_mean(
            hilbert_hellinger_by_graph,
            hilbert_active,
        )
        hilbert_per_graph = config.hilbert_to_born_weight * (
            hilbert_kl_by_graph + hilbert_hellinger_by_graph
        )
    else:
        hilbert_kl = hilbert_hellinger = _zero(reference)
        hilbert_per_graph = reference.new_zeros(graph_count)
        hilbert_active = torch.zeros(
            graph_count,
            dtype=torch.bool,
            device=reference.device,
        )

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

    uncertainty = output.uncertainty.log_variance
    task_rows = (
        (
            "diagnosis",
            diagnosis_per_graph,
            diagnosis_active,
        ),
        (
            "action_ranking",
            action_per_graph,
            action_active,
        ),
        (
            "born_prediction",
            born_per_graph,
            born_active,
        ),
        (
            "hilbert_deformation",
            hilbert_per_graph,
            hilbert_active,
        ),
    )
    weighted_tasks: dict[str, Tensor] = {}
    for uncertainty_name, values, active_mask in task_rows:
        index = UNCERTAINTY_TARGETS.index(uncertainty_name)
        weighted_tasks[uncertainty_name] = (
            _heteroscedastic_mean(values, uncertainty[:, index], active_mask)
            if config.uncertainty_weighting
            else _active_mean(values, active_mask)
        )
    diagnosis = weighted_tasks["diagnosis"]
    action = weighted_tasks["action_ranking"]
    born = weighted_tasks["born_prediction"]
    hilbert_to_born = weighted_tasks["hilbert_deformation"]

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
