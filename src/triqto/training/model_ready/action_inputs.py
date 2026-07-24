"""Deployable action inputs and two-stage model-ready targets."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch

from triqto.model import ACTION_EDIT_TYPES, ActionCandidateTensorBatch, TriQTOModelConfig

from .graph_inputs import require_input
from .source import scalar_bool
from .types import ModelReadyActionTargets

_ACTION_EDIT_TYPE_MAP = {
    "append_rx": "rx",
    "append_ry": "ry",
    "append_rz": "rz",
    "append_rzz": "rzz",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "rzz": "rzz",
    "layout": "layout",
    "routing": "routing",
    "diagnostic_basis": "diagnostic_basis",
}


def action_batch(
    inputs: Mapping[str, np.ndarray], n_qubits: int, config: TriQTOModelConfig
) -> ActionCandidateTensorBatch | None:
    features_value = inputs.get("x_action_candidate_features")
    if features_value is None:
        return None
    features = np.asarray(features_value, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != config.action_candidate_feature_dim:
        raise ValueError("action candidate feature width does not match model config")
    if not np.isfinite(features).all():
        raise ValueError("action candidate features contain non-finite values")
    count = features.shape[0]
    declared_count = inputs.get("x_action_candidate_count")
    if declared_count is not None and int(np.asarray(declared_count).reshape(-1)[0]) != count:
        raise ValueError("x_action_candidate_count does not match feature rows")
    edit_ptr = np.asarray(
        require_input(inputs, "x_action_edit_ptr"), dtype=np.int64
    ).reshape(-1)
    edit_types = [
        str(value)
        for value in np.asarray(
            require_input(inputs, "x_action_edit_types")
        ).reshape(-1).tolist()
    ]
    magnitudes = np.asarray(
        require_input(inputs, "x_action_edit_magnitudes"), dtype=np.float32
    ).reshape(-1)
    qubit_ptr = np.asarray(
        require_input(inputs, "x_action_edit_qubit_ptr"), dtype=np.int64
    ).reshape(-1)
    qubits = np.asarray(
        require_input(inputs, "x_action_edit_qubits"), dtype=np.int64
    ).reshape(-1)
    if edit_ptr.shape != (count + 1,) or edit_ptr[0] != 0:
        raise ValueError("x_action_edit_ptr is inconsistent")
    if edit_ptr[-1] != len(edit_types) or len(edit_types) != magnitudes.size:
        raise ValueError("action edit arrays have inconsistent lengths")
    if qubit_ptr.shape != (len(edit_types) + 1,) or qubit_ptr[0] != 0:
        raise ValueError("x_action_edit_qubit_ptr is inconsistent")
    if qubit_ptr[-1] != qubits.size:
        raise ValueError("action edit qubit pointer does not span qubits")
    if bool((edit_ptr[1:] < edit_ptr[:-1]).any()) or bool(
        (qubit_ptr[1:] < qubit_ptr[:-1]).any()
    ):
        raise ValueError("action CSR pointers must be nondecreasing")

    type_position = {name: index for index, name in enumerate(ACTION_EDIT_TYPES)}
    type_ids: list[int] = []
    expanded_magnitudes: list[float] = []
    normalized_qubits: list[float] = []
    edit_candidate: list[int] = []
    denominator = max(n_qubits - 1, 1)
    for candidate in range(count):
        for edit_index in range(int(edit_ptr[candidate]), int(edit_ptr[candidate + 1])):
            mapped = _ACTION_EDIT_TYPE_MAP.get(edit_types[edit_index])
            if mapped is None or mapped not in type_position:
                raise ValueError(f"unsupported action edit type {edit_types[edit_index]!r}")
            operands = qubits[
                int(qubit_ptr[edit_index]) : int(qubit_ptr[edit_index + 1])
            ]
            if operands.size == 0:
                raise ValueError("non-no-op action edit has no qubit operands")
            for qubit in operands.tolist():
                if qubit < 0 or qubit >= n_qubits:
                    raise ValueError(f"action edit qubit {qubit} is out of range")
                type_ids.append(type_position[mapped])
                expanded_magnitudes.append(float(magnitudes[edit_index]))
                normalized_qubits.append(float(qubit) / denominator)
                edit_candidate.append(candidate)
    return ActionCandidateTensorBatch(
        candidate_features=torch.from_numpy(features.copy()),
        candidate_batch=torch.zeros(count, dtype=torch.long),
        candidate_available_mask=torch.ones(count, dtype=torch.bool),
        edit_type_ids=torch.tensor(type_ids, dtype=torch.long),
        edit_magnitudes=torch.tensor(expanded_magnitudes, dtype=torch.float32),
        edit_qubit_positions=torch.tensor(normalized_qubits, dtype=torch.float32),
        edit_candidate_index=torch.tensor(edit_candidate, dtype=torch.long),
    )


def action_targets(
    targets: Mapping[str, np.ndarray], candidate_count: int
) -> ModelReadyActionTargets:
    if "y_should_act" not in targets:
        return ModelReadyActionTargets(
            should_act=torch.zeros(1),
            should_act_weight=torch.ones(1),
            should_act_mask=torch.zeros(1, dtype=torch.bool),
            ranking_loss_mask=torch.zeros(1, dtype=torch.bool),
            candidate_reward=torch.zeros(candidate_count),
            candidate_rank=torch.zeros(candidate_count, dtype=torch.long),
            candidate_selected_mask=torch.zeros(candidate_count, dtype=torch.bool),
            candidate_listwise_distribution=torch.zeros(candidate_count),
            candidate_target_mask=torch.zeros(candidate_count, dtype=torch.bool),
            candidate_batch=torch.zeros(candidate_count, dtype=torch.long),
        )
    should_act = scalar_bool(np.asarray(targets["y_should_act"]), "y_should_act")
    ranking = scalar_bool(
        np.asarray(targets["y_ranking_loss_mask"]), "y_ranking_loss_mask"
    )
    weight = float(np.asarray(targets["y_should_act_weight"]).reshape(-1)[0])
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("y_should_act_weight must be finite and positive")
    reward = np.asarray(targets["y_candidate_reward"], dtype=np.float32).reshape(-1)
    rank = np.asarray(targets["y_candidate_rank"], dtype=np.int64).reshape(-1)
    selected = np.asarray(
        targets["y_candidate_selected_mask"], dtype=np.bool_
    ).reshape(-1)
    distribution = np.asarray(
        targets["y_candidate_listwise_distribution"], dtype=np.float32
    ).reshape(-1)
    eligible = np.asarray(
        targets.get(
            "y_candidate_eligible_mask",
            np.ones(candidate_count, dtype=np.bool_),
        ),
        dtype=np.bool_,
    ).reshape(-1)
    for name, array in (
        ("reward", reward),
        ("rank", rank),
        ("selected", selected),
        ("distribution", distribution),
        ("eligible", eligible),
    ):
        if array.size != candidate_count:
            raise ValueError(f"candidate target {name} width mismatch")
    if not np.isfinite(reward).all() or not np.isfinite(distribution).all():
        raise ValueError("candidate targets contain non-finite values")
    if ranking:
        if not should_act:
            raise ValueError("ranking loss cannot be active when should_act is false")
        if int(selected.sum()) != 1:
            raise ValueError("active ranking target must select exactly one candidate")
        if float(distribution.sum()) <= 0.0:
            raise ValueError("active listwise distribution must have positive mass")
    return ModelReadyActionTargets(
        should_act=torch.tensor([float(should_act)], dtype=torch.float32),
        should_act_weight=torch.tensor([weight], dtype=torch.float32),
        should_act_mask=torch.tensor([True]),
        ranking_loss_mask=torch.tensor([ranking]),
        candidate_reward=torch.from_numpy(reward.copy()),
        candidate_rank=torch.from_numpy(rank.copy()),
        candidate_selected_mask=torch.from_numpy(selected.copy()),
        candidate_listwise_distribution=torch.from_numpy(distribution.copy()),
        candidate_target_mask=torch.from_numpy(eligible.copy()),
        candidate_batch=torch.zeros(candidate_count, dtype=torch.long),
    )


__all__ = ["action_batch", "action_targets"]
