"""Ideal-simulator rollout and deterministic ranking for Phase 9 actions."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from triqto.graph import validate_probability_mapping
from triqto.metrics import compare_born_distributions
from triqto.simulation import simulate_ideal_statevector

from .apply_actions import apply_action
from .config import ActionEngineConfig
from .identities import (
    action_rollout_id,
    action_scientific_config_id,
    rollout_content_hash,
)
from .models import ActionCandidate, ActionRollout
from .rewards import score_action_rollout
from .validators import validate_action_rollout


def _probability_arrays(
    probabilities: Mapping[str, Any],
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    validate_probability_mapping(probabilities, n_qubits)
    items = sorted(probabilities.items())
    width = max(1, n_qubits)
    outcomes = np.asarray([key for key, _ in items], dtype=f"<U{width}")
    values = np.asarray([float(value) for _, value in items], dtype=np.float64)
    return outcomes, values


def run_action_rollouts(
    *,
    distorted_circuit: QuantumCircuit,
    clean_target_run_id: str,
    clean_probabilities: Mapping[str, Any],
    distorted_probabilities: Mapping[str, Any],
    candidates: list[ActionCandidate],
    config: ActionEngineConfig,
) -> list[ActionRollout]:
    """Apply, exactly simulate, score, and rank every candidate for one sample."""
    if not isinstance(distorted_circuit, QuantumCircuit):
        raise TypeError("distorted_circuit must be qiskit.QuantumCircuit")
    if not isinstance(clean_target_run_id, str) or not clean_target_run_id.strip():
        raise ValueError("clean_target_run_id must be nonblank")
    if not candidates:
        raise ValueError("At least one action candidate is required")
    validate_probability_mapping(clean_probabilities, distorted_circuit.num_qubits)
    validate_probability_mapping(
        distorted_probabilities,
        distorted_circuit.num_qubits,
    )

    baseline_bundle = compare_born_distributions(
        clean_probabilities,
        distorted_probabilities,
        include_kl=False,
        include_js_distance=False,
    )
    provisional: list[ActionRollout] = []
    for candidate in candidates:
        applied = apply_action(distorted_circuit, candidate, config)
        simulated = simulate_ideal_statevector(applied.circuit)
        candidate_bundle = compare_born_distributions(
            clean_probabilities,
            simulated.probabilities,
            include_kl=False,
            include_js_distance=False,
        )
        depth_delta = applied.candidate_depth - applied.source_depth
        gate_delta = applied.candidate_gate_count - applied.source_gate_count
        breakdown = score_action_rollout(
            baseline_metrics=baseline_bundle,
            candidate_metrics=candidate_bundle,
            depth_delta=depth_delta,
            gate_delta=gate_delta,
            edit_count=len(candidate.edits),
            risk_score=candidate.risk_score,
            config=config,
        )
        outcomes, probabilities = _probability_arrays(
            simulated.probabilities,
            distorted_circuit.num_qubits,
        )
        provisional.append(
            ActionRollout(
                rollout_id=action_rollout_id(
                    candidate.action_id,
                    clean_target_run_id,
                    config,
                ),
                action_id=candidate.action_id,
                sample_id=candidate.sample_id,
                graph_pair_id=candidate.graph_pair_id,
                candidate_circuit_id=applied.candidate_circuit_id,
                clean_target_run_id=clean_target_run_id,
                scientific_config_id=action_scientific_config_id(config),
                rank=1,
                reward=breakdown.reward,
                risk_score=candidate.risk_score,
                metric_names=breakdown.metric_names,
                baseline_metric_values=breakdown.baseline_metric_values,
                candidate_metric_values=breakdown.candidate_metric_values,
                improvement_values=breakdown.improvement_values,
                outcome_bitstrings=outcomes,
                exact_probabilities=probabilities,
                dominates_baseline=breakdown.dominates_baseline,
                primary_metric_nonworsening=(
                    breakdown.primary_metric_nonworsening
                ),
                selected=False,
                candidate_circuit=applied.circuit,
                depth_delta=depth_delta,
                gate_delta=gate_delta,
                metadata={
                    "candidate_circuit_hash": applied.circuit_hash,
                    "decomposition_metadata": applied.decomposition_metadata,
                    "metric_weights": [
                        config.reward_total_variation_weight,
                        config.reward_jensen_shannon_weight,
                        config.reward_hellinger_weight,
                    ],
                    "improvement_atol": config.improvement_atol,
                    "weighted_improvement": breakdown.weighted_improvement,
                    "depth_penalty": breakdown.depth_penalty,
                    "gate_penalty": breakdown.gate_penalty,
                    "edit_penalty": breakdown.edit_penalty,
                    "risk_penalty": breakdown.risk_penalty,
                    "exact_born_recovery": breakdown.exact_born_recovery,
                    "validation_mode": "ideal_statevector",
                    "candidate_generation_is_not_a_learned_policy": True,
                },
            )
        )

    provisional.sort(
        key=lambda rollout: (
            not rollout.primary_metric_nonworsening,
            -rollout.reward,
            rollout.risk_score,
            rollout.action_id,
        )
    )
    for index, rollout in enumerate(provisional, start=1):
        rollout.rank = index
        rollout.selected = index == 1
        rollout.content_hash = rollout_content_hash(rollout)
        validate_action_rollout(rollout, require_hash=True)
    return provisional


__all__ = ["run_action_rollouts"]
