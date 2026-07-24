"""Convert strict x_*/y_* artifacts into full model-ready supervision."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from triqto.model import DISTORTION_LABELS, TriQTOModelConfig
from triqto.training.constants import DISTORTION_TO_COARSE_LABEL

from .adapter import build_model_ready_example
from .source import scalar_bool
from .types import ModelReadyArtifact
from .multitask_types import (
    ModelReadyBornTargets,
    ModelReadyDiagnosisTargets,
    ModelReadyMultitaskExample,
)

_DIAGNOSIS_TASKS = {"diagnosis", "joint_multitask", "hardware_masked"}
_ACTION_TASKS = {"action_ranking", "joint_multitask", "hardware_masked"}
_BORN_TASKS = {"born_prediction", "joint_multitask", "hardware_masked"}


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar value")
    return str(array.reshape(-1)[0])


def _head_available(
    targets: Mapping[str, np.ndarray], task: str, head: str
) -> bool:
    if task == "joint_multitask":
        prefix = "y_joint_head"
    elif task == "hardware_masked":
        prefix = "y_hardware_head"
    else:
        return {
            "diagnosis": task in _DIAGNOSIS_TASKS,
            "action_ranking": task in _ACTION_TASKS,
            "born_prediction": task in _BORN_TASKS,
        }[head]
    names_value = targets.get(f"{prefix}_names")
    mask_value = targets.get(f"{prefix}_target_available_mask")
    if names_value is None or mask_value is None:
        # Older preprocessing products did not always carry per-head target masks.
        return {
            "diagnosis": task in _DIAGNOSIS_TASKS,
            "action_ranking": task in _ACTION_TASKS,
            "born_prediction": task in _BORN_TASKS,
        }[head]
    names = [str(value) for value in np.asarray(names_value).reshape(-1).tolist()]
    mask = np.asarray(mask_value, dtype=np.bool_).reshape(-1)
    if len(names) != mask.size:
        raise ValueError(f"{prefix} names/target mask widths differ")
    positions = [index for index, name in enumerate(names) if name == head]
    if len(positions) != 1:
        raise ValueError(f"{prefix} must contain head {head!r} exactly once")
    return bool(mask[positions[0]])


def _empty_diagnosis(n_qubits: int) -> ModelReadyDiagnosisTargets:
    return ModelReadyDiagnosisTargets(
        class_index=torch.zeros(1, dtype=torch.long),
        class_mask=torch.zeros(1, dtype=torch.bool),
        strength=torch.zeros(1, dtype=torch.float32),
        strength_mask=torch.zeros(1, dtype=torch.bool),
        affected_qubit=torch.zeros(n_qubits, dtype=torch.float32),
        affected_qubit_mask=torch.zeros(n_qubits, dtype=torch.bool),
    )


def _diagnosis_targets(
    targets: Mapping[str, np.ndarray], task: str, n_qubits: int
) -> ModelReadyDiagnosisTargets:
    if not _head_available(targets, task, "diagnosis"):
        return _empty_diagnosis(n_qubits)
    raw_value = targets.get("y_diagnosis_distortion_type")
    affected_value = targets.get("y_diagnosis_affected_qubit_mask")
    if raw_value is None or affected_value is None:
        raise ValueError(f"diagnosis-active task {task!r} is missing diagnosis targets")
    raw_name = _scalar_text(np.asarray(raw_value), "y_diagnosis_distortion_type")
    coarse = raw_name if raw_name in DISTORTION_LABELS else DISTORTION_TO_COARSE_LABEL.get(raw_name)
    if coarse is None or coarse not in DISTORTION_LABELS:
        raise ValueError(f"no versioned diagnosis mapping for distortion {raw_name!r}")
    affected = np.asarray(affected_value, dtype=np.bool_).reshape(-1)
    if affected.size != n_qubits:
        raise ValueError("diagnosis affected-qubit target width does not match graph")
    strength_value = targets.get("y_diagnosis_strength")
    strength_mask_value = targets.get("y_diagnosis_strength_available_mask")
    if strength_value is None:
        strength = np.asarray([0.0], dtype=np.float32)
        strength_mask = np.asarray([False], dtype=np.bool_)
    else:
        strength = np.asarray(strength_value, dtype=np.float32).reshape(-1)
        if strength.size != 1 or not np.isfinite(strength).all():
            raise ValueError("diagnosis strength must be one finite scalar")
        strength_mask = (
            np.asarray([True], dtype=np.bool_)
            if strength_mask_value is None
            else np.asarray(strength_mask_value, dtype=np.bool_).reshape(-1)
        )
        if strength_mask.size != 1:
            raise ValueError("diagnosis strength mask must contain one value")
    return ModelReadyDiagnosisTargets(
        class_index=torch.tensor([DISTORTION_LABELS.index(coarse)], dtype=torch.long),
        class_mask=torch.tensor([True], dtype=torch.bool),
        strength=torch.from_numpy(strength.copy()),
        strength_mask=torch.from_numpy(strength_mask.copy()),
        affected_qubit=torch.from_numpy(affected.astype(np.float32, copy=True)),
        affected_qubit_mask=torch.ones(n_qubits, dtype=torch.bool),
    )


def _born_targets(
    targets: Mapping[str, np.ndarray], task: str
) -> tuple[ModelReadyBornTargets, tuple[tuple[str, float], ...]]:
    if not _head_available(targets, task, "born_prediction"):
        empty = ModelReadyBornTargets(
            probabilities=torch.zeros(0, dtype=torch.float32),
            outcome_batch=torch.zeros(0, dtype=torch.long),
            row_mask=torch.zeros(0, dtype=torch.bool),
        )
        return empty, ()
    outcomes_value = targets.get("y_born_target_outcome_bitstrings")
    probabilities_value = targets.get("y_born_target_probabilities")
    if outcomes_value is None or probabilities_value is None:
        raise ValueError(f"Born-active task {task!r} is missing Born targets")
    outcomes = [str(value) for value in np.asarray(outcomes_value).reshape(-1).tolist()]
    probabilities = np.asarray(probabilities_value, dtype=np.float32).reshape(-1)
    if len(outcomes) != probabilities.size or probabilities.size == 0:
        raise ValueError("Born target support/probability widths are inconsistent")
    if not np.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("Born targets must be finite and nonnegative")
    mass = float(probabilities.sum())
    if not np.isclose(mass, 1.0, rtol=0.0, atol=1.0e-5):
        raise ValueError(f"Born target probabilities must sum to one, observed {mass}")
    result = ModelReadyBornTargets(
        probabilities=torch.from_numpy(probabilities.copy()),
        outcome_batch=torch.zeros(probabilities.size, dtype=torch.long),
        row_mask=torch.ones(probabilities.size, dtype=torch.bool),
    )
    distribution = tuple(zip(outcomes, (float(value) for value in probabilities), strict=True))
    return result, distribution


def build_model_ready_multitask_example(
    artifact: ModelReadyArtifact,
    model_config: TriQTOModelConfig,
) -> ModelReadyMultitaskExample:
    """Build one strictly separated model/target example for any published task."""
    base = build_model_ready_example(artifact, model_config)
    n_qubits = int(base.model_batch.graph.node_features.shape[0])
    diagnosis = _diagnosis_targets(artifact.targets, base.task, n_qubits)
    born, distribution = _born_targets(artifact.targets, base.task)

    action_available = _head_available(artifact.targets, base.task, "action_ranking")
    if not action_available:
        base.action_targets.should_act_mask.zero_()
        base.action_targets.ranking_loss_mask.zero_()
        base.action_targets.candidate_target_mask.zero_()
    elif not bool(base.action_targets.should_act_mask.any()):
        raise ValueError(f"action-active task {base.task!r} is missing should-act targets")

    metadata = {
        "topology_available": bool(
            scalar_bool(
                np.asarray(
                    artifact.inputs.get(
                        "x_topology_available_mask", np.asarray(False, dtype=np.bool_)
                    )
                ),
                "x_topology_available_mask",
            )
        ),
        "head_available": {
            "diagnosis": _head_available(artifact.targets, base.task, "diagnosis"),
            "action_ranking": action_available,
            "born_prediction": _head_available(
                artifact.targets, base.task, "born_prediction"
            ),
        },
    }
    return ModelReadyMultitaskExample(
        base=base,
        diagnosis_targets=diagnosis,
        born_targets=born,
        n_qubits=n_qubits,
        born_distribution=distribution,
        metadata=metadata,
    )


__all__ = ["build_model_ready_multitask_example"]
