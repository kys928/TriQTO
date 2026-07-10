"""Privileged synthetic rule-only correction baseline for Phase 10."""
from __future__ import annotations

from typing import Any


def select_rule_only(
    *,
    rollouts: tuple[Any, ...],
    candidates_by_id: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Select the smallest known synthetic inverse, or no-op when none is valid.

    This baseline intentionally consumes privileged Phase 7 distortion metadata encoded in
    Phase 9's `oracle_inverse` provenance. It is a simulator-only upper control, not a
    hardware-facing diagnosis rule.
    """
    oracle_rollouts = [
        rollout
        for rollout in rollouts
        if "oracle_inverse"
        in candidates_by_id[rollout.action_id].generation_sources
    ]
    oracle_rollouts.sort(
        key=lambda item: (
            len(candidates_by_id[item.action_id].edits),
            item.risk_score,
            item.action_id,
        )
    )
    if oracle_rollouts:
        selected = oracle_rollouts[0]
        fallback = False
    else:
        no_ops = [
            rollout
            for rollout in rollouts
            if not candidates_by_id[rollout.action_id].edits
        ]
        if len(no_ops) != 1:
            raise ValueError("Each sample must expose exactly one no-op rollout")
        selected = no_ops[0]
        fallback = True
    return selected, {
        "selection_rule": "smallest deterministic oracle inverse; otherwise no-op",
        "oracle_candidate_count": len(oracle_rollouts),
        "fallback_to_no_op": fallback,
        "clean_target_used_for_selection": False,
        "distortion_metadata_used_for_selection": True,
        "privileged_synthetic_rule": True,
        "deployable_hardware_rule": False,
    }


__all__ = ["select_rule_only"]
