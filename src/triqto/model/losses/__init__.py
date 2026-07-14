"""Loss primitives for the future Phase 14 trainer."""
from .action_losses import ActionLosses, action_ranking_losses
from .diagnosis_losses import DiagnosisLosses, diagnosis_losses
from .geometry_losses import pairwise_distance_consistency_loss
from .multitask_loss import MultiTaskLossWeights, combine_multitask_losses
from .task_losses import (
    distribution_kl_loss,
    masked_binary_cross_entropy_with_logits,
    masked_cross_entropy,
    masked_mean,
    masked_mse_loss,
)
from .topology_losses import apply_phase13_topology_weight, topology_feature_discrepancy

__all__ = [
    "ActionLosses",
    "DiagnosisLosses",
    "MultiTaskLossWeights",
    "action_ranking_losses",
    "apply_phase13_topology_weight",
    "combine_multitask_losses",
    "diagnosis_losses",
    "distribution_kl_loss",
    "masked_binary_cross_entropy_with_logits",
    "masked_cross_entropy",
    "masked_mean",
    "masked_mse_loss",
    "pairwise_distance_consistency_loss",
    "topology_feature_discrepancy",
]

from .uncertainty_losses import per_example_gaussian_nll, reduce_masked_per_example_loss, uncertainty_error_correlation
