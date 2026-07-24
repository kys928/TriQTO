"""Typed forward outputs for the untrained Phase 13 architecture."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class DistortionHeadOutput:
    class_logits: Tensor
    strength_mean: Tensor
    strength_log_scale: Tensor
    affected_qubit_logits: Tensor
    graph_available_mask: Tensor


@dataclass(slots=True)
class ActionRankingHeadOutput:
    should_act_logit: Tensor
    should_act_probability: Tensor
    should_act_available_mask: Tensor
    candidate_scores: Tensor
    candidate_probabilities: Tensor
    predicted_rewards: Tensor
    candidate_batch: Tensor
    candidate_available_mask: Tensor
    graph_available_mask: Tensor


@dataclass(slots=True)
class BornPredictionHeadOutput:
    outcome_logits: Tensor
    probabilities: Tensor
    outcome_batch: Tensor
    graph_available_mask: Tensor


@dataclass(slots=True)
class HilbertDeformationHeadOutput:
    mean: Tensor
    log_scale: Tensor
    graph_available_mask: Tensor


@dataclass(slots=True)
class UncertaintyHeadOutput:
    log_variance: Tensor
    graph_available_mask: Tensor


@dataclass(slots=True)
class TopologyHeadOutput:
    feature_prediction: Tensor
    confidence_logit: Tensor
    graph_available_mask: Tensor
    supervised_target_available_mask: Tensor


@dataclass(slots=True)
class TriQTOModelOutput:
    model_architecture_id: str
    graph_embedding: Tensor
    node_embeddings: Tensor
    stream_embeddings: Tensor
    stream_available_mask: Tensor
    effective_head_stream_mask: Tensor
    fusion_weights: Tensor
    head_latents: Tensor
    distortion: DistortionHeadOutput
    action_ranking: ActionRankingHeadOutput
    born_prediction: BornPredictionHeadOutput
    hilbert_deformation: HilbertDeformationHeadOutput
    uncertainty: UncertaintyHeadOutput
    topology: TopologyHeadOutput


__all__ = [
    "ActionRankingHeadOutput",
    "BornPredictionHeadOutput",
    "DistortionHeadOutput",
    "HilbertDeformationHeadOutput",
    "TopologyHeadOutput",
    "TriQTOModelOutput",
    "UncertaintyHeadOutput",
]
