"""Vectorized variable-size collation for immutable model-ready examples."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from triqto.model import (
    ActionCandidateTensorBatch,
    BornTensorBatch,
    DenseFeatureBatch,
    GraphTensorBatch,
    OutcomeQueryTensorBatch,
    ParameterTensorBatch,
    TriQTOBatch,
)
from triqto.training.config import TrainingConfig

from .multitask_types import (
    ModelReadyBornTargets,
    ModelReadyDiagnosisTargets,
    ModelReadyGeometryTargets,
    ModelReadyMultitaskExample,
    ModelReadyMultitaskTargets,
    ModelReadySupervisedBatch,
)
from .types import ModelReadyActionTargets


def _cat_or_empty(parts: list[Tensor], *, dtype: torch.dtype) -> Tensor:
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=dtype)


def _collate_graph(examples: Sequence[ModelReadyMultitaskExample]) -> GraphTensorBatch:
    graphs = [example.model_batch.graph for example in examples]
    node_offsets: list[int] = []
    gate_offsets: list[int] = []
    node_total = 0
    gate_total = 0
    for graph in graphs:
        node_offsets.append(node_total)
        gate_offsets.append(gate_total)
        node_total += int(graph.node_features.shape[0])
        gate_total += int(graph.gate_features.shape[0])

    edge_indices = [
        graph.edge_index + node_offsets[index]
        for index, graph in enumerate(graphs)
    ]
    edge_events = [
        graph.edge_event_index + gate_offsets[index]
        for index, graph in enumerate(graphs)
    ]
    gate_ptr_values = [0]
    gate_indices: list[Tensor] = []
    incidence_total = 0
    for index, graph in enumerate(graphs):
        counts = graph.gate_qubit_ptr[1:] - graph.gate_qubit_ptr[:-1]
        for count in counts.tolist():
            incidence_total += int(count)
            gate_ptr_values.append(incidence_total)
        gate_indices.append(graph.gate_qubit_indices + node_offsets[index])

    return GraphTensorBatch(
        node_features=torch.cat([graph.node_features for graph in graphs], dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat([graph.edge_features for graph in graphs], dim=0),
        edge_event_index=torch.cat(edge_events, dim=0),
        gate_features=torch.cat([graph.gate_features for graph in graphs], dim=0),
        gate_qubit_ptr=torch.tensor(gate_ptr_values, dtype=torch.long),
        gate_qubit_indices=torch.cat(gate_indices, dim=0),
        node_batch=torch.cat(
            [
                torch.full(
                    (graph.node_features.shape[0],), index, dtype=torch.long
                )
                for index, graph in enumerate(graphs)
            ]
        ),
        gate_batch=torch.cat(
            [
                torch.full(
                    (graph.gate_features.shape[0],), index, dtype=torch.long
                )
                for index, graph in enumerate(graphs)
            ]
        ),
        graph_count=len(examples),
    )


def _collate_parameter(
    examples: Sequence[ModelReadyMultitaskExample],
) -> ParameterTensorBatch | None:
    rows = [
        (index, example.model_batch.parameter)
        for index, example in enumerate(examples)
        if example.model_batch.parameter is not None
    ]
    if not rows:
        return None
    available = torch.zeros(len(examples), dtype=torch.bool)
    for index, _ in rows:
        available[index] = True
    return ParameterTensorBatch(
        values=torch.cat([value.values for _, value in rows], dim=0),
        sin=torch.cat([value.sin for _, value in rows], dim=0),
        cos=torch.cat([value.cos for _, value in rows], dim=0),
        batch_index=torch.cat(
            [
                torch.full((value.values.numel(),), index, dtype=torch.long)
                for index, value in rows
            ]
        ),
        available_mask=available,
    )


def _collate_born(
    examples: Sequence[ModelReadyMultitaskExample], attribute: str
) -> BornTensorBatch | None:
    rows = [
        (index, getattr(example.model_batch, attribute))
        for index, example in enumerate(examples)
        if getattr(example.model_batch, attribute) is not None
    ]
    if not rows:
        return None
    width = max(value.outcome_bits.shape[1] for _, value in rows)
    bits: list[Tensor] = []
    masks: list[Tensor] = []
    available = torch.zeros(len(examples), dtype=torch.bool)
    for index, value in rows:
        padding = width - value.outcome_bits.shape[1]
        bits.append(F.pad(value.outcome_bits, (0, padding)))
        masks.append(F.pad(value.outcome_bit_mask, (0, padding)))
        available[index] = True
    return BornTensorBatch(
        outcome_bits=torch.cat(bits, dim=0),
        outcome_bit_mask=torch.cat(masks, dim=0),
        probabilities=torch.cat([value.probabilities for _, value in rows], dim=0),
        batch_index=torch.cat(
            [
                torch.full((value.probabilities.numel(),), index, dtype=torch.long)
                for index, value in rows
            ]
        ),
        available_mask=available,
    )


def _collate_dense(
    examples: Sequence[ModelReadyMultitaskExample], attribute: str
) -> DenseFeatureBatch | None:
    present = [getattr(example.model_batch, attribute) for example in examples]
    available_values = [value for value in present if value is not None]
    if not available_values:
        return None
    width = int(available_values[0].features.shape[1])
    features = torch.zeros((len(examples), width), dtype=torch.float32)
    available = torch.zeros(len(examples), dtype=torch.bool)
    for index, value in enumerate(present):
        if value is None:
            continue
        if value.features.shape != (1, width):
            raise ValueError(f"{attribute} feature width changed within batch")
        features[index] = value.features[0]
        available[index] = True
    return DenseFeatureBatch(features=features, available_mask=available)


def _collate_actions(
    examples: Sequence[ModelReadyMultitaskExample],
) -> ActionCandidateTensorBatch | None:
    rows = [
        (index, example.model_batch.actions)
        for index, example in enumerate(examples)
        if example.model_batch.actions is not None
    ]
    if not rows:
        return None
    offsets: list[int] = []
    total = 0
    for _, value in rows:
        offsets.append(total)
        total += int(value.candidate_features.shape[0])
    return ActionCandidateTensorBatch(
        candidate_features=torch.cat(
            [value.candidate_features for _, value in rows], dim=0
        ),
        candidate_batch=torch.cat(
            [
                torch.full(
                    (value.candidate_features.shape[0],), index, dtype=torch.long
                )
                for index, value in rows
            ]
        ),
        candidate_available_mask=torch.cat(
            [value.candidate_available_mask for _, value in rows], dim=0
        ),
        edit_type_ids=_cat_or_empty(
            [value.edit_type_ids for _, value in rows], dtype=torch.long
        ),
        edit_magnitudes=_cat_or_empty(
            [value.edit_magnitudes for _, value in rows], dtype=torch.float32
        ),
        edit_qubit_positions=_cat_or_empty(
            [value.edit_qubit_positions for _, value in rows], dtype=torch.float32
        ),
        edit_candidate_index=_cat_or_empty(
            [
                value.edit_candidate_index + offsets[position]
                for position, (_, value) in enumerate(rows)
            ],
            dtype=torch.long,
        ),
    )


def _collate_queries(
    examples: Sequence[ModelReadyMultitaskExample],
) -> OutcomeQueryTensorBatch | None:
    rows = [
        (index, example.model_batch.born_queries)
        for index, example in enumerate(examples)
        if example.model_batch.born_queries is not None
    ]
    if not rows:
        return None
    width = max(value.outcome_bits.shape[1] for _, value in rows)
    bits: list[Tensor] = []
    masks: list[Tensor] = []
    available = torch.zeros(len(examples), dtype=torch.bool)
    for index, value in rows:
        padding = width - value.outcome_bits.shape[1]
        bits.append(F.pad(value.outcome_bits, (0, padding)))
        masks.append(F.pad(value.outcome_bit_mask, (0, padding)))
        available[index] = True
    return OutcomeQueryTensorBatch(
        outcome_bits=torch.cat(bits, dim=0),
        outcome_bit_mask=torch.cat(masks, dim=0),
        batch_index=torch.cat(
            [
                torch.full((value.outcome_bits.shape[0],), index, dtype=torch.long)
                for index, value in rows
            ]
        ),
        available_mask=available,
    )


def _collate_diagnosis(
    examples: Sequence[ModelReadyMultitaskExample],
) -> ModelReadyDiagnosisTargets:
    return ModelReadyDiagnosisTargets(
        class_index=torch.cat(
            [example.diagnosis_targets.class_index for example in examples]
        ),
        class_mask=torch.cat(
            [example.diagnosis_targets.class_mask for example in examples]
        ),
        strength=torch.cat(
            [example.diagnosis_targets.strength for example in examples]
        ),
        strength_mask=torch.cat(
            [example.diagnosis_targets.strength_mask for example in examples]
        ),
        affected_qubit=torch.cat(
            [example.diagnosis_targets.affected_qubit for example in examples]
        ),
        affected_qubit_mask=torch.cat(
            [example.diagnosis_targets.affected_qubit_mask for example in examples]
        ),
    )


def _collate_action_targets(
    examples: Sequence[ModelReadyMultitaskExample],
) -> ModelReadyActionTargets:
    candidate_counts = [
        int(example.action_targets.candidate_reward.numel()) for example in examples
    ]
    for example, count in zip(examples, candidate_counts, strict=True):
        target = example.action_targets
        candidate_fields = (
            target.candidate_reward,
            target.candidate_rank,
            target.candidate_selected_mask,
            target.candidate_listwise_distribution,
            target.candidate_target_mask,
            target.candidate_batch,
        )
        if any(value.numel() != count for value in candidate_fields):
            raise ValueError(
                f"candidate target widths differ for {example.view_item_id}"
            )
    return ModelReadyActionTargets(
        should_act=torch.cat(
            [example.action_targets.should_act for example in examples]
        ),
        should_act_weight=torch.cat(
            [example.action_targets.should_act_weight for example in examples]
        ),
        should_act_mask=torch.cat(
            [example.action_targets.should_act_mask for example in examples]
        ),
        ranking_loss_mask=torch.cat(
            [example.action_targets.ranking_loss_mask for example in examples]
        ),
        candidate_reward=_cat_or_empty(
            [example.action_targets.candidate_reward for example in examples],
            dtype=torch.float32,
        ),
        candidate_rank=_cat_or_empty(
            [example.action_targets.candidate_rank for example in examples],
            dtype=torch.long,
        ),
        candidate_selected_mask=_cat_or_empty(
            [example.action_targets.candidate_selected_mask for example in examples],
            dtype=torch.bool,
        ),
        candidate_listwise_distribution=_cat_or_empty(
            [
                example.action_targets.candidate_listwise_distribution
                for example in examples
            ],
            dtype=torch.float32,
        ),
        candidate_target_mask=_cat_or_empty(
            [example.action_targets.candidate_target_mask for example in examples],
            dtype=torch.bool,
        ),
        candidate_batch=_cat_or_empty(
            [
                torch.full((count,), index, dtype=torch.long)
                for index, count in enumerate(candidate_counts)
            ],
            dtype=torch.long,
        ),
    )


def _collate_born_targets(
    examples: Sequence[ModelReadyMultitaskExample],
) -> ModelReadyBornTargets:
    probability_parts: list[Tensor] = []
    mask_parts: list[Tensor] = []
    batch_parts: list[Tensor] = []
    for index, example in enumerate(examples):
        queries = example.model_batch.born_queries
        query_count = 0 if queries is None else int(queries.outcome_bits.shape[0])
        target = example.born_targets
        if target.probabilities.numel() not in {0, query_count}:
            raise ValueError(
                f"Born target/query row mismatch for {example.view_item_id}"
            )
        if target.probabilities.numel() == 0:
            probability_parts.append(torch.zeros(query_count, dtype=torch.float32))
            mask_parts.append(torch.zeros(query_count, dtype=torch.bool))
        else:
            probability_parts.append(target.probabilities)
            mask_parts.append(target.row_mask)
        batch_parts.append(torch.full((query_count,), index, dtype=torch.long))
    return ModelReadyBornTargets(
        probabilities=_cat_or_empty(probability_parts, dtype=torch.float32),
        outcome_batch=_cat_or_empty(batch_parts, dtype=torch.long),
        row_mask=_cat_or_empty(mask_parts, dtype=torch.bool),
    )


def _geometry_targets(
    examples: Sequence[ModelReadyMultitaskExample],
) -> ModelReadyGeometryTargets:
    count = len(examples)
    distances = torch.zeros((count, count), dtype=torch.float32)
    mask = torch.zeros((count, count), dtype=torch.bool)
    for left in range(count):
        for right in range(left + 1, count):
            if examples[left].n_qubits != examples[right].n_qubits:
                continue
            p_map = dict(examples[left].born_distribution)
            q_map = dict(examples[right].born_distribution)
            if not p_map or not q_map:
                continue
            support = sorted(set(p_map) | set(q_map))
            p = np.asarray([p_map.get(key, 0.0) for key in support], dtype=np.float64)
            q = np.asarray([q_map.get(key, 0.0) for key in support], dtype=np.float64)
            value = float(
                np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum())
            )
            if not math.isfinite(value):
                raise FloatingPointError("geometry target is non-finite")
            distances[left, right] = distances[right, left] = value
            mask[left, right] = mask[right, left] = True
    return ModelReadyGeometryTargets(target_distance=distances, pair_mask=mask)


def collate_model_ready_multitask_examples(
    examples: Sequence[ModelReadyMultitaskExample],
) -> ModelReadySupervisedBatch:
    """Collate variable-size examples into one true vectorized model forward."""
    if not examples:
        raise ValueError("cannot collate an empty model-ready batch")
    graph = _collate_graph(examples)
    model_batch = TriQTOBatch(
        graph=graph,
        parameter=_collate_parameter(examples),
        born=_collate_born(examples, "born"),
        hilbert=None,
        backend=_collate_dense(examples, "backend"),
        topology=_collate_dense(examples, "topology"),
        actions=_collate_actions(examples),
        born_queries=_collate_queries(examples),
        hardware_mode_mask=torch.cat(
            [example.model_batch.resolved_hardware_mask() for example in examples]
        ),
        topology_hilbert_dependent_mask=torch.cat(
            [
                example.model_batch.resolved_topology_hilbert_dependency()
                for example in examples
            ]
        ),
        head_stream_mask=torch.cat(
            [example.model_batch.head_stream_mask for example in examples], dim=0
        ),
        head_active_mask=torch.cat(
            [example.model_batch.head_active_mask for example in examples], dim=0
        ),
    )
    targets = ModelReadyMultitaskTargets(
        diagnosis=_collate_diagnosis(examples),
        action=_collate_action_targets(examples),
        born_prediction=_collate_born_targets(examples),
        geometry=_geometry_targets(examples),
    )
    return ModelReadySupervisedBatch(
        item_ids=tuple(example.view_item_id for example in examples),
        entity_ids=tuple(example.entity_id for example in examples),
        tasks=tuple(example.task for example in examples),
        splits=tuple(example.split for example in examples),
        split_group_ids=tuple(example.split_group_id for example in examples),
        model_batch=model_batch,
        targets=targets,
    )


def validate_model_ready_batch_budget(
    batch: ModelReadySupervisedBatch, config: TrainingConfig
) -> dict[str, int]:
    model = batch.model_batch
    sizes = {
        "graphs": batch.graph_count,
        "nodes": int(model.graph.node_features.shape[0]),
        "edges": int(model.graph.edge_features.shape[0]),
        "gates": int(model.graph.gate_features.shape[0]),
        "candidates": (
            0 if model.actions is None else int(model.actions.candidate_features.shape[0])
        ),
        "outcomes": (
            0 if model.born_queries is None else int(model.born_queries.outcome_bits.shape[0])
        ),
        "hilbert": 0,
    }
    limits = {
        "graphs": config.batch_size,
        "nodes": config.max_nodes_per_batch,
        "edges": config.max_edges_per_batch,
        "gates": config.max_gates_per_batch,
        "candidates": config.max_candidates_per_batch,
        "outcomes": config.max_outcomes_per_batch,
        "hilbert": config.max_hilbert_amplitudes_per_batch,
    }
    exceeded = {
        name: (sizes[name], limits[name])
        for name in sizes
        if sizes[name] > limits[name]
    }
    if exceeded:
        raise RuntimeError(f"model-ready vectorized batch exceeds guardrails: {exceeded}")
    return sizes


__all__ = [
    "collate_model_ready_multitask_examples",
    "validate_model_ready_batch_budget",
]
