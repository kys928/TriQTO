"""Deterministic identities and logical content hashes for Phase 9."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from triqto.core.ids import canonical_json, make_deterministic_id

from .config import ActionEngineConfig, action_config_to_dict
from .constants import (
    ACTION_ANGLE_NORMALIZATION_VERSION,
    ACTION_APPLICATION_VERSION,
    ACTION_ARTIFACT_SCHEMA_VERSION,
    ACTION_RANKING_VERSION,
    ACTION_RISK_VERSION,
    ACTION_SCHEMA_VERSION,
    PRIMARY_REWARD_METRICS,
    RISK_EDIT_COUNT_SCALE,
    ROLLOUT_ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_EDIT_TYPES,
)
from .models import ActionCandidate, ActionEdit, ActionRollout


def action_schema_id() -> str:
    """Return the fixed representation identity for Phase 9 v1 actions."""
    return make_deterministic_id(
        "actionschema",
        {
            "schema_version": ACTION_SCHEMA_VERSION,
            "action_artifact_schema_version": ACTION_ARTIFACT_SCHEMA_VERSION,
            "rollout_artifact_schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
            "application_version": ACTION_APPLICATION_VERSION,
            "ranking_version": ACTION_RANKING_VERSION,
            "risk_version": ACTION_RISK_VERSION,
            "angle_normalization_version": ACTION_ANGLE_NORMALIZATION_VERSION,
            "supported_edit_types": SUPPORTED_EDIT_TYPES,
            "primary_reward_metrics": PRIMARY_REWARD_METRICS,
            "risk_edit_count_scale": RISK_EDIT_COUNT_SCALE,
        },
    )


def scientific_action_config_payload(config: ActionEngineConfig) -> dict[str, Any]:
    """Return scientific candidate/ranking choices, excluding inactive guardrails."""
    return {
        "schema_version": config.schema_version,
        "candidate_magnitudes": list(config.candidate_magnitudes),
        "include_no_op": config.include_no_op,
        "include_blind_candidates": config.include_blind_candidates,
        "include_oracle_inverse": config.include_oracle_inverse,
        "observed_edges_only": config.observed_edges_only,
        "max_abs_angle": config.max_abs_angle,
        "reward_total_variation_weight": config.reward_total_variation_weight,
        "reward_jensen_shannon_weight": config.reward_jensen_shannon_weight,
        "reward_hellinger_weight": config.reward_hellinger_weight,
        "depth_penalty_weight": config.depth_penalty_weight,
        "gate_penalty_weight": config.gate_penalty_weight,
        "edit_penalty_weight": config.edit_penalty_weight,
        "risk_penalty_weight": config.risk_penalty_weight,
        "improvement_atol": config.improvement_atol,
    }


def action_engine_id(
    source_scientific_generation_id: str,
    graph_conversion_id: str,
    config: ActionEngineConfig,
) -> str:
    """Identify the deterministic scientific Phase 9 candidate universe."""
    return make_deterministic_id(
        "actionengine",
        {
            "source_scientific_generation_id": source_scientific_generation_id,
            "graph_conversion_id": graph_conversion_id,
            "action_schema_id": action_schema_id(),
            "scientific_config": scientific_action_config_payload(config),
        },
    )


def action_operational_config_id(config: ActionEngineConfig) -> str:
    """Identify the complete operational Phase 9 configuration."""
    return make_deterministic_id("actionconfig", action_config_to_dict(config))


def edit_payload(edit: ActionEdit) -> dict[str, Any]:
    return {
        "edit_type": edit.edit_type,
        "qubits": list(edit.qubits),
        "magnitude": edit.magnitude,
    }


def action_risk_from_edits(
    edits: tuple[ActionEdit, ...],
    max_abs_angle: float,
) -> float:
    """Return the fixed-version deterministic action-size risk heuristic."""
    if not edits:
        return 0.0
    average_magnitude = sum(abs(edit.magnitude) for edit in edits) / len(edits)
    magnitude_component = min(1.0, average_magnitude / max_abs_angle)
    count_component = min(1.0, len(edits) / RISK_EDIT_COUNT_SCALE)
    return float(min(1.0, 0.75 * magnitude_component + 0.25 * count_component))


def candidate_action_id(
    *,
    sample_id: str,
    graph_pair_id: str,
    source_circuit_id: str,
    source_run_id: str,
    edits: tuple[ActionEdit, ...],
) -> str:
    """Return a stable action ID independent of generation provenance and paths."""
    return make_deterministic_id(
        "action",
        {
            "sample_id": sample_id,
            "graph_pair_id": graph_pair_id,
            "source_circuit_id": source_circuit_id,
            "source_run_id": source_run_id,
            "edits": [edit_payload(edit) for edit in edits],
            "action_schema_id": action_schema_id(),
        },
    )


def candidate_circuit_id(source_circuit_id: str, action_id: str) -> str:
    """Return a stable ID for the circuit produced by an action."""
    return make_deterministic_id(
        "candidatecircuit",
        {
            "source_circuit_id": source_circuit_id,
            "action_id": action_id,
            "application_version": ACTION_APPLICATION_VERSION,
        },
    )


def action_scientific_config_id(config: ActionEngineConfig) -> str:
    """Identify scientific action/reward choices independently of guardrails."""
    return make_deterministic_id(
        "actionscience", scientific_action_config_payload(config)
    )


def action_rollout_id_from_config_id(
    action_id: str,
    clean_target_run_id: str,
    scientific_config_id: str,
) -> str:
    """Build a rollout identity from its validated scientific config identity."""
    return make_deterministic_id(
        "actionrollout",
        {
            "action_id": action_id,
            "clean_target_run_id": clean_target_run_id,
            "scientific_config_id": scientific_config_id,
            "ranking_version": ACTION_RANKING_VERSION,
            "rollout_artifact_schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
        },
    )


def action_rollout_id(
    action_id: str,
    clean_target_run_id: str,
    config: ActionEngineConfig,
) -> str:
    """Return a stable ideal-validation rollout ID."""
    return action_rollout_id_from_config_id(
        action_id,
        clean_target_run_id,
        action_scientific_config_id(config),
    )


def _sha256_payload(payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def action_content_hash(candidate: ActionCandidate) -> str:
    """Hash the complete deterministic action artifact content, not its file path."""
    return _sha256_payload(
        {
            "action_schema_id": action_schema_id(),
            "action_id": candidate.action_id,
            "sample_id": candidate.sample_id,
            "graph_pair_id": candidate.graph_pair_id,
            "source_circuit_id": candidate.source_circuit_id,
            "source_run_id": candidate.source_run_id,
            "distortion_id": candidate.distortion_id,
            "edits": [edit_payload(edit) for edit in candidate.edits],
            "generation_sources": list(candidate.generation_sources),
            "risk_score": candidate.risk_score,
            "metadata": candidate.metadata,
        }
    )


def _normalize_circuit_parameter(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Circuit parameters must be finite")
        return numeric
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("Circuit complex parameters must be finite")
        return [float(value.real), float(value.imag)]
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, "name"):
        return {"symbol": str(value)}
    return str(value)


def circuit_semantic_payload(circuit: Any) -> dict[str, Any]:
    """Return a stable semantic circuit payload for QPY readback verification."""
    instructions: list[dict[str, Any]] = []
    for item in circuit.data:
        operation = item.operation
        instructions.append(
            {
                "name": operation.name,
                "params": [
                    _normalize_circuit_parameter(parameter)
                    for parameter in operation.params
                ],
                "qubits": [circuit.find_bit(qubit).index for qubit in item.qubits],
                "clbits": [circuit.find_bit(clbit).index for clbit in item.clbits],
                "condition": _normalize_circuit_parameter(
                    getattr(operation, "condition", None)
                ),
            }
        )
    return {
        "n_qubits": circuit.num_qubits,
        "n_clbits": circuit.num_clbits,
        "global_phase": _normalize_circuit_parameter(circuit.global_phase),
        "parameters": sorted(str(parameter) for parameter in circuit.parameters),
        "instructions": instructions,
    }


def circuit_semantic_hash(circuit: Any) -> str:
    return _sha256_payload(circuit_semantic_payload(circuit))


def _array_payload(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def rollout_content_hash(rollout: ActionRollout) -> str:
    """Hash the validated ranking evidence stored in a rollout artifact."""
    return _sha256_payload(
        {
            "rollout_schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
            "rollout_id": rollout.rollout_id,
            "action_id": rollout.action_id,
            "sample_id": rollout.sample_id,
            "graph_pair_id": rollout.graph_pair_id,
            "candidate_circuit_id": rollout.candidate_circuit_id,
            "clean_target_run_id": rollout.clean_target_run_id,
            "scientific_config_id": rollout.scientific_config_id,
            "rank": rollout.rank,
            "reward": rollout.reward,
            "risk_score": rollout.risk_score,
            "metric_names": _array_payload(rollout.metric_names),
            "baseline_metric_values": _array_payload(
                rollout.baseline_metric_values
            ),
            "candidate_metric_values": _array_payload(
                rollout.candidate_metric_values
            ),
            "improvement_values": _array_payload(rollout.improvement_values),
            "outcome_bitstrings": _array_payload(rollout.outcome_bitstrings),
            "exact_probabilities": _array_payload(rollout.exact_probabilities),
            "dominates_baseline": rollout.dominates_baseline,
            "primary_metric_nonworsening": rollout.primary_metric_nonworsening,
            "selected": rollout.selected,
            "depth_delta": rollout.depth_delta,
            "gate_delta": rollout.gate_delta,
            "metadata": rollout.metadata,
        }
    )


__all__ = [
    "action_content_hash",
    "action_engine_id",
    "action_operational_config_id",
    "action_risk_from_edits",
    "action_rollout_id",
    "action_rollout_id_from_config_id",
    "action_scientific_config_id",
    "action_schema_id",
    "candidate_action_id",
    "candidate_circuit_id",
    "circuit_semantic_hash",
    "circuit_semantic_payload",
    "edit_payload",
    "rollout_content_hash",
    "scientific_action_config_payload",
]
