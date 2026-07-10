"""Versioned constants for the deterministic Phase 9 action engine."""
from __future__ import annotations

ACTION_SCHEMA_VERSION = "triqto.action.phase9.v1"
ACTION_ARTIFACT_SCHEMA_VERSION = "triqto.action.artifact.v1"
ROLLOUT_ARTIFACT_SCHEMA_VERSION = "triqto.action.rollout_artifact.v1"
ACTION_DATASET_SCHEMA_VERSION = "triqto.action.dataset.v1"
ACTION_APPLICATION_VERSION = "triqto.action.application.v1"
ACTION_RANKING_VERSION = "triqto.action.ranking.v1"
ACTION_ANGLE_NORMALIZATION_VERSION = "triqto.action.angle_wrap.v1"
ACTION_RISK_VERSION = "triqto.action.risk.v2"
RISK_EDIT_COUNT_SCALE = 8.0

SUPPORTED_EDIT_TYPES: tuple[str, ...] = (
    "append_rx",
    "append_ry",
    "append_rz",
    "append_rzz",
)

GENERATION_SOURCES: tuple[str, ...] = (
    "blind_physics_prior",
    "no_op",
    "oracle_inverse",
)

PRIMARY_REWARD_METRICS: tuple[str, ...] = (
    "total_variation",
    "jensen_shannon_divergence",
    "hellinger",
)

__all__ = [
    "ACTION_ANGLE_NORMALIZATION_VERSION",
    "ACTION_APPLICATION_VERSION",
    "ACTION_ARTIFACT_SCHEMA_VERSION",
    "ACTION_DATASET_SCHEMA_VERSION",
    "ACTION_RANKING_VERSION",
    "ACTION_RISK_VERSION",
    "ACTION_SCHEMA_VERSION",
    "GENERATION_SOURCES",
    "PRIMARY_REWARD_METRICS",
    "RISK_EDIT_COUNT_SCALE",
    "ROLLOUT_ARTIFACT_SCHEMA_VERSION",
    "SUPPORTED_EDIT_TYPES",
]
