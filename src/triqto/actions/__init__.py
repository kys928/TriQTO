"""Public deterministic Phase 9 action and correction APIs."""
from __future__ import annotations

from .action_space import supported_edit_types
from .apply_actions import apply_action
from .artifacts import (
    load_action_artifact,
    load_candidate_circuit,
    load_rollout_artifact,
    save_action_artifact,
    save_candidate_circuit,
    save_rollout_artifact,
    write_action_dataset,
)
from .candidates import (
    action_risk_score,
    generate_action_candidates,
    normalize_rotation_angle,
    observed_two_qubit_edges,
    oracle_inverse_edits,
)
from .config import (
    ActionEngineConfig,
    action_config_from_dict,
    action_config_to_dict,
    load_action_config,
    save_action_config,
)
from .identities import (
    action_content_hash,
    action_engine_id,
    action_operational_config_id,
    action_rollout_id,
    action_rollout_id_from_config_id,
    action_scientific_config_id,
    action_schema_id,
    candidate_action_id,
    candidate_circuit_id,
    circuit_semantic_hash,
    rollout_content_hash,
)
from .models import (
    ActionCandidate,
    ActionEdit,
    ActionEngineResult,
    ActionEngineSources,
    ActionRollout,
    ActionWriteResult,
    AppliedAction,
    CompletedGraphDataset,
)
from .pipeline import build_action_engine_result
from .rewards import RewardBreakdown, primary_metric_array, score_action_rollout
from .rollout_runner import run_action_rollouts
from .source import (
    load_action_engine_sources,
    load_completed_graph_dataset,
    verify_action_source_snapshots,
)
from .validators import (
    validate_action_candidate,
    validate_action_dataset_joins,
    validate_action_edit,
    validate_action_rollout,
    validate_applied_action,
)

__all__ = [
    "ActionCandidate",
    "ActionEdit",
    "ActionEngineConfig",
    "ActionEngineResult",
    "ActionEngineSources",
    "ActionRollout",
    "ActionWriteResult",
    "AppliedAction",
    "CompletedGraphDataset",
    "RewardBreakdown",
    "action_config_from_dict",
    "action_config_to_dict",
    "action_content_hash",
    "action_engine_id",
    "action_operational_config_id",
    "action_risk_score",
    "action_rollout_id",
    "action_rollout_id_from_config_id",
    "action_scientific_config_id",
    "action_schema_id",
    "apply_action",
    "build_action_engine_result",
    "candidate_action_id",
    "candidate_circuit_id",
    "circuit_semantic_hash",
    "generate_action_candidates",
    "load_action_artifact",
    "load_action_config",
    "load_action_engine_sources",
    "load_candidate_circuit",
    "load_completed_graph_dataset",
    "load_rollout_artifact",
    "normalize_rotation_angle",
    "observed_two_qubit_edges",
    "oracle_inverse_edits",
    "primary_metric_array",
    "rollout_content_hash",
    "run_action_rollouts",
    "save_action_artifact",
    "save_action_config",
    "save_candidate_circuit",
    "save_rollout_artifact",
    "score_action_rollout",
    "supported_edit_types",
    "validate_action_candidate",
    "validate_action_dataset_joins",
    "validate_action_edit",
    "validate_action_rollout",
    "validate_applied_action",
    "verify_action_source_snapshots",
    "write_action_dataset",
]
