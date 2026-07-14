"""Versioned constants for deterministic Phase 15 evaluation."""
from __future__ import annotations

EVALUATION_SCHEMA_VERSION = "triqto.evaluation.phase15.v1"
EVALUATION_ARTIFACT_VERSION = "triqto.evaluation.item_artifact.v1"
EVALUATION_METRIC_VERSION = "triqto.evaluation.metrics.v1"

EVALUATION_TASKS = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "hilbert_to_born",
    "joint_multitask",
    "hardware_masked",
)

ABLATION_NAMES = (
    "full",
    "no_topology",
    "no_hilbert",
)

GROUP_DIMENSIONS = (
    "task",
    "family",
    "n_qubits",
    "distortion_id",
)

CALIBRATION_HEADS = (
    "diagnosis",
    "action_ranking",
)

__all__ = [
    "ABLATION_NAMES",
    "CALIBRATION_HEADS",
    "EVALUATION_ARTIFACT_VERSION",
    "EVALUATION_METRIC_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "EVALUATION_TASKS",
    "GROUP_DIMENSIONS",
]
