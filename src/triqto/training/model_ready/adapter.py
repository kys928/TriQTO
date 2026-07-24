"""Leakage-safe assembly of model-ready x_* arrays into one model graph."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from triqto.model import TriQTOBatch, TriQTOModelConfig
from triqto.model.constants import HEAD_ORDER, STREAM_ORDER

from .action_inputs import action_batch, action_targets
from .graph_inputs import (
    backend_batch,
    born_input,
    born_queries_from_inputs,
    graph_batch,
    parameter_batch,
)
from .topology_inputs import canonical_topology_input, topology_batch
from .types import ModelReadyArtifact, ModelReadyExample


def _head_masks(
    task: str,
    *,
    parameter: bool,
    born: bool,
    backend: bool,
    topology: bool,
) -> tuple[Tensor, Tensor]:
    stream_available = {
        "circuit_graph": True,
        "parameter": parameter,
        "phasor": parameter,
        "hilbert": False,
        "born": born,
        "backend": backend,
        "topology": topology,
    }
    primary_by_task = {
        "diagnosis": ("diagnosis",),
        "action_ranking": ("action_ranking",),
        "born_prediction": ("born_prediction",),
        "joint_multitask": ("diagnosis", "action_ranking", "born_prediction"),
        "hardware_masked": ("diagnosis", "action_ranking", "born_prediction"),
    }
    active_names = list(primary_by_task[task])
    if topology and task == "joint_multitask":
        active_names.append("topology")
    active_names.append("uncertainty")
    active = torch.zeros((1, len(HEAD_ORDER)), dtype=torch.bool)
    mask = torch.zeros((1, len(HEAD_ORDER), len(STREAM_ORDER)), dtype=torch.bool)
    head_position = {name: index for index, name in enumerate(HEAD_ORDER)}
    stream_position = {name: index for index, name in enumerate(STREAM_ORDER)}
    for head in active_names:
        row = head_position[head]
        active[0, row] = True
        for stream, available in stream_available.items():
            if available:
                mask[0, row, stream_position[stream]] = True
    for head in ("action_ranking", "born_prediction"):
        mask[0, head_position[head], stream_position["topology"]] = False
    mask[0, head_position["born_prediction"], stream_position["born"]] = False
    mask[0, head_position["topology"], stream_position["topology"]] = False
    return mask, active


def _verify_born_support(
    inputs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    queries_present: bool,
) -> None:
    if "y_born_target_outcome_bitstrings" not in targets or not queries_present:
        return
    target_outcomes = [
        str(value)
        for value in np.asarray(
            targets["y_born_target_outcome_bitstrings"]
        ).reshape(-1).tolist()
    ]
    input_outcomes = [
        str(value)
        for value in np.asarray(
            inputs["x_born_input_outcome_bitstrings"]
        ).reshape(-1).tolist()
    ]
    if target_outcomes != input_outcomes:
        raise ValueError(
            "Born target support differs from x_born_input support; refusing "
            "to feed y_* support into the model"
        )


def build_model_ready_example(
    artifact: ModelReadyArtifact,
    model_config: TriQTOModelConfig,
) -> ModelReadyExample:
    """Build one graph using only x_* arrays; y_* remains target-side."""
    if not isinstance(model_config, TriQTOModelConfig):
        raise TypeError("model_config must be TriQTOModelConfig")
    task = str(artifact.record["task"])
    graph = graph_batch(artifact.inputs)
    n_qubits = graph.node_features.shape[0]
    parameter = parameter_batch(artifact.inputs)
    born = born_input(artifact.inputs)
    backend = backend_batch(artifact.inputs, model_config)
    topology, ablations = topology_batch(artifact.inputs, model_config)
    actions = action_batch(artifact.inputs, n_qubits, model_config)
    queries = born_queries_from_inputs(artifact.inputs, task)
    head_mask, head_active = _head_masks(
        task,
        parameter=parameter is not None,
        born=born is not None,
        backend=backend is not None,
        topology=topology is not None,
    )
    hardware = task == "hardware_masked"
    manifold_mask = artifact.inputs.get("x_topology_manifold_available_mask")
    topology_hilbert_dependent = bool(
        topology is not None
        and manifold_mask is not None
        and np.asarray(manifold_mask, dtype=np.bool_).reshape(-1).size >= 2
        and np.asarray(manifold_mask, dtype=np.bool_).reshape(-1)[1]
    )
    model_batch = TriQTOBatch(
        graph=graph,
        parameter=parameter,
        born=born,
        hilbert=None,
        backend=backend,
        topology=topology,
        actions=actions,
        born_queries=queries,
        hardware_mode_mask=torch.tensor([hardware]),
        topology_hilbert_dependent_mask=torch.tensor([topology_hilbert_dependent]),
        head_stream_mask=head_mask,
        head_active_mask=head_active,
    )
    _verify_born_support(
        artifact.inputs,
        artifact.targets,
        queries_present=queries is not None,
    )
    candidate_count = 0 if actions is None else actions.candidate_features.shape[0]
    targets = action_targets(artifact.targets, candidate_count)
    model_batch.validate(model_config)
    return ModelReadyExample(
        view_item_id=str(artifact.record["view_item_id"]),
        entity_id=str(artifact.record["entity_id"]),
        task=task,
        split=str(artifact.record["split"]),
        split_group_id=str(artifact.record["split_group_id"]),
        model_batch=model_batch,
        action_targets=targets,
        targets={name: value.copy() for name, value in artifact.targets.items()},
        topology_ablation_inputs=ablations,
    )


__all__ = ["build_model_ready_example", "canonical_topology_input"]
