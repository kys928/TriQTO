"""Candidate-aware and per-head metrics for model-ready multi-task training."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import torch
from torch import Tensor

from triqto.model.constants import DISTORTION_LABELS, HEAD_ORDER, STREAM_ORDER
from triqto.model.outputs import TriQTOModelOutput

from .multitask_types import ModelReadySupervisedBatch


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _binary_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average_rank
        position = end
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels, strict=True) if label == 1
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


class ModelReadyMetricAccumulator:
    """Streaming CPU-side metrics; no metric tensor enters the gradient graph."""

    def __init__(self, *, train_reward_mean: float, no_op_feature_index: int) -> None:
        if not math.isfinite(train_reward_mean):
            raise ValueError("train_reward_mean must be finite")
        self.train_reward_mean = float(train_reward_mean)
        self.no_op_feature_index = int(no_op_feature_index)
        self.item_count = 0
        self.batch_count = 0
        self.loss_sums: dict[str, float] = defaultdict(float)
        self.task_item_count: dict[str, int] = defaultdict(int)
        self.task_loss_sums: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.gate_scores: list[float] = []
        self.gate_labels: list[int] = []
        self.gate_tp = self.gate_tn = self.gate_fp = self.gate_fn = 0
        self.ranking_graphs = 0
        self.ranking_top1 = 0
        self.ranking_top3 = 0
        self.ranking_reciprocal_rank = 0.0
        self.ranking_ndcg = 0.0
        self.ranking_percentile = 0.0
        self.random_top1_baseline = 0.0
        self.reward_squared_error = 0.0
        self.reward_zero_squared_error = 0.0
        self.reward_mean_squared_error = 0.0
        self.reward_candidate_count = 0
        self.selected_minus_noop_sum = 0.0
        self.selected_minus_noop_count = 0
        self.diagnosis_count = 0
        self.diagnosis_correct = 0
        self.diagnosis_confusion = [
            [0 for _ in DISTORTION_LABELS] for _ in DISTORTION_LABELS
        ]
        self.strength_absolute_error = 0.0
        self.strength_count = 0
        self.affected_tp = self.affected_fp = self.affected_fn = 0
        self.stream_mask_sums: dict[str, float] = defaultdict(float)
        self.stream_mask_items = 0

    def update(
        self,
        output: TriQTOModelOutput,
        batch: ModelReadySupervisedBatch,
        losses: dict[str, Tensor],
    ) -> None:
        count = batch.graph_count
        if count <= 0:
            raise ValueError("metric batch must contain graphs")
        if len(set(batch.tasks)) != 1:
            raise ValueError("metric batches must be task-homogeneous")
        task = batch.tasks[0]
        self.item_count += count
        self.batch_count += 1
        self.task_item_count[task] += count
        for name, value in losses.items():
            numeric = float(value.detach().cpu())
            self.loss_sums[name] += numeric * count
            self.task_loss_sums[task][name] += numeric * count

        target = batch.targets.action
        gate_mask = target.should_act_mask & output.action_ranking.should_act_available_mask
        probabilities = output.action_ranking.should_act_probability.detach().cpu()
        labels = target.should_act.detach().cpu()
        for index in torch.nonzero(gate_mask.detach().cpu(), as_tuple=False).flatten().tolist():
            score = float(probabilities[index])
            label = int(labels[index] > 0.5)
            predicted = int(score >= 0.5)
            self.gate_scores.append(score)
            self.gate_labels.append(label)
            if predicted == 1 and label == 1:
                self.gate_tp += 1
            elif predicted == 0 and label == 0:
                self.gate_tn += 1
            elif predicted == 1:
                self.gate_fp += 1
            else:
                self.gate_fn += 1

        action = output.action_ranking
        candidate_features = (
            None
            if batch.model_batch.actions is None
            else batch.model_batch.actions.candidate_features.detach().cpu()
        )
        scores = action.candidate_scores.detach().cpu()
        rewards = action.predicted_rewards.detach().cpu()
        available = action.candidate_available_mask.detach().cpu()
        candidate_batch = target.candidate_batch.detach().cpu()
        candidate_target = target.candidate_target_mask.detach().cpu()
        selected = target.candidate_selected_mask.detach().cpu()
        target_rewards = target.candidate_reward.detach().cpu()
        ranking_mask = target.ranking_loss_mask.detach().cpu()
        for graph in range(count):
            if not bool(ranking_mask[graph]):
                continue
            mask = (candidate_batch == graph) & candidate_target & available
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                raise ValueError("ranking-active graph has no eligible candidates")
            selected_indices = indices[selected.index_select(0, indices)]
            if selected_indices.numel() != 1:
                raise ValueError("ranking-active graph must select exactly one candidate")
            target_index = int(selected_indices[0])
            ordered = indices[torch.argsort(scores.index_select(0, indices), descending=True)]
            rank_position = int(
                torch.nonzero(ordered == target_index, as_tuple=False).flatten()[0]
            ) + 1
            candidate_count = int(indices.numel())
            self.ranking_graphs += 1
            self.ranking_top1 += int(rank_position <= 1)
            self.ranking_top3 += int(rank_position <= min(3, candidate_count))
            self.ranking_reciprocal_rank += 1.0 / rank_position
            self.ranking_ndcg += 1.0 / math.log2(rank_position + 1.0)
            self.ranking_percentile += (
                1.0
                if candidate_count == 1
                else 1.0 - (rank_position - 1) / (candidate_count - 1)
            )
            self.random_top1_baseline += 1.0 / candidate_count

            graph_rewards = target_rewards.index_select(0, indices)
            graph_predictions = rewards.index_select(0, indices)
            self.reward_squared_error += float(
                (graph_predictions - graph_rewards).square().sum()
            )
            self.reward_zero_squared_error += float(graph_rewards.square().sum())
            self.reward_mean_squared_error += float(
                (graph_rewards - self.train_reward_mean).square().sum()
            )
            self.reward_candidate_count += candidate_count

            if candidate_features is not None:
                graph_features = candidate_features.index_select(0, indices)
                if self.no_op_feature_index < graph_features.shape[1]:
                    no_op_local = torch.nonzero(
                        graph_features[:, self.no_op_feature_index] > 0.5,
                        as_tuple=False,
                    ).flatten()
                    if no_op_local.numel() == 1:
                        no_op_index = int(indices[int(no_op_local[0])])
                        self.selected_minus_noop_sum += float(
                            target_rewards[target_index] - target_rewards[no_op_index]
                        )
                        self.selected_minus_noop_count += 1

        diagnosis = batch.targets.diagnosis
        class_mask = diagnosis.class_mask.detach().cpu()
        predicted_class = output.distortion.class_logits.detach().cpu().argmax(dim=1)
        true_class = diagnosis.class_index.detach().cpu()
        for index in torch.nonzero(class_mask, as_tuple=False).flatten().tolist():
            expected = int(true_class[index])
            predicted = int(predicted_class[index])
            self.diagnosis_count += 1
            self.diagnosis_correct += int(expected == predicted)
            self.diagnosis_confusion[expected][predicted] += 1

        strength_mask = diagnosis.strength_mask.detach().cpu()
        strength_prediction = output.distortion.strength_mean.detach().cpu()
        strength_target = diagnosis.strength.detach().cpu()
        active_strength = torch.nonzero(strength_mask, as_tuple=False).flatten()
        if active_strength.numel():
            self.strength_absolute_error += float(
                (
                    strength_prediction.index_select(0, active_strength)
                    - strength_target.index_select(0, active_strength)
                ).abs().sum()
            )
            self.strength_count += int(active_strength.numel())

        affected_mask = diagnosis.affected_qubit_mask.detach().cpu()
        if bool(affected_mask.any()):
            predicted_affected = (
                torch.sigmoid(output.distortion.affected_qubit_logits.detach().cpu())
                >= 0.5
            )
            expected_affected = diagnosis.affected_qubit.detach().cpu() >= 0.5
            self.affected_tp += int(
                (predicted_affected & expected_affected & affected_mask).sum()
            )
            self.affected_fp += int(
                (predicted_affected & ~expected_affected & affected_mask).sum()
            )
            self.affected_fn += int(
                (~predicted_affected & expected_affected & affected_mask).sum()
            )

        stream = output.stream_available_mask.detach().cpu().to(torch.float32).mean(dim=0)
        for index, name in enumerate(STREAM_ORDER):
            self.stream_mask_sums[f"stream_available:{name}"] += float(stream[index]) * count
        effective = output.effective_head_stream_mask.detach().cpu().to(torch.float32).mean(dim=0)
        for head_index, head in enumerate(HEAD_ORDER):
            for stream_index, stream_name in enumerate(STREAM_ORDER):
                self.stream_mask_sums[
                    f"head_stream:{head}:{stream_name}"
                ] += float(effective[head_index, stream_index]) * count
        self.stream_mask_items += count

    def finalize(self) -> dict[str, Any]:
        if self.item_count <= 0:
            raise ValueError("cannot finalize empty metrics")
        gate_count = len(self.gate_labels)
        gate_accuracy = _safe_ratio(self.gate_tp + self.gate_tn, gate_count)
        precision = _safe_ratio(self.gate_tp, self.gate_tp + self.gate_fp)
        recall = _safe_ratio(self.gate_tp, self.gate_tp + self.gate_fn)
        specificity = _safe_ratio(self.gate_tn, self.gate_tn + self.gate_fp)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
        balanced_accuracy = (
            None
            if recall is None or specificity is None
            else 0.5 * (recall + specificity)
        )
        affected_precision = _safe_ratio(
            self.affected_tp, self.affected_tp + self.affected_fp
        )
        affected_recall = _safe_ratio(
            self.affected_tp, self.affected_tp + self.affected_fn
        )
        affected_f1 = (
            None
            if affected_precision is None
            or affected_recall is None
            or affected_precision + affected_recall == 0
            else 2.0
            * affected_precision
            * affected_recall
            / (affected_precision + affected_recall)
        )
        per_task = {
            task: {
                "item_count": self.task_item_count[task],
                "losses": {
                    name: value / self.task_item_count[task]
                    for name, value in sorted(self.task_loss_sums[task].items())
                },
            }
            for task in sorted(self.task_item_count)
        }
        return {
            "item_count": self.item_count,
            "batch_count": self.batch_count,
            "losses": {
                name: value / self.item_count
                for name, value in sorted(self.loss_sums.items())
            },
            "per_task": per_task,
            "should_act": {
                "count": gate_count,
                "accuracy": gate_accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "auroc": _binary_auc(self.gate_scores, self.gate_labels),
                "confusion": {
                    "tp": self.gate_tp,
                    "tn": self.gate_tn,
                    "fp": self.gate_fp,
                    "fn": self.gate_fn,
                },
            },
            "ranking": {
                "active_graphs": self.ranking_graphs,
                "top1_accuracy": _safe_ratio(self.ranking_top1, self.ranking_graphs),
                "top3_accuracy": _safe_ratio(self.ranking_top3, self.ranking_graphs),
                "mean_reciprocal_rank": _safe_ratio(
                    self.ranking_reciprocal_rank, self.ranking_graphs
                ),
                "ndcg": _safe_ratio(self.ranking_ndcg, self.ranking_graphs),
                "selected_candidate_percentile": _safe_ratio(
                    self.ranking_percentile, self.ranking_graphs
                ),
                "random_top1_baseline": _safe_ratio(
                    self.random_top1_baseline, self.ranking_graphs
                ),
            },
            "reward": {
                "candidate_count": self.reward_candidate_count,
                "mse": _safe_ratio(
                    self.reward_squared_error, self.reward_candidate_count
                ),
                "zero_baseline_mse": _safe_ratio(
                    self.reward_zero_squared_error, self.reward_candidate_count
                ),
                "train_mean_baseline_mse": _safe_ratio(
                    self.reward_mean_squared_error, self.reward_candidate_count
                ),
                "selected_minus_noop_target_reward_mean": _safe_ratio(
                    self.selected_minus_noop_sum, self.selected_minus_noop_count
                ),
                "selected_minus_noop_count": self.selected_minus_noop_count,
            },
            "diagnosis": {
                "count": self.diagnosis_count,
                "class_accuracy": _safe_ratio(
                    self.diagnosis_correct, self.diagnosis_count
                ),
                "confusion_matrix": self.diagnosis_confusion,
                "labels": list(DISTORTION_LABELS),
                "strength_mae": _safe_ratio(
                    self.strength_absolute_error, self.strength_count
                ),
                "affected_qubit_precision": affected_precision,
                "affected_qubit_recall": affected_recall,
                "affected_qubit_f1": affected_f1,
            },
            "mask_utilization": {
                name: value / self.stream_mask_items
                for name, value in sorted(self.stream_mask_sums.items())
            },
        }


__all__ = ["ModelReadyMetricAccumulator"]
