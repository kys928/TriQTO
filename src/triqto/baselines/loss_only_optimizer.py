"""Clean-target loss-only action-selection baseline for Phase 10."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import BaselineSuiteConfig
from .optimizer_common import weighted_objective


def select_loss_only(
    *,
    rollouts: tuple[Any, ...],
    candidates_by_id: dict[str, Any],
    config: BaselineSuiteConfig,
) -> tuple[Any, dict[str, Any]]:
    """Select the Phase 9 candidate with minimum Born loss and no risk/cost penalty."""
    eligible: list[Any] = []
    for rollout in rollouts:
        candidate = candidates_by_id[rollout.action_id]
        if (
            not config.loss_only_allow_oracle
            and "oracle_inverse" in candidate.generation_sources
        ):
            continue
        eligible.append(rollout)
    if not eligible:
        raise ValueError("loss_only has no eligible Phase 9 rollout")
    ranked = sorted(
        eligible,
        key=lambda item: (
            weighted_objective(
                np.asarray(item.candidate_metric_values, dtype=np.float64), config
            ),
            item.action_id,
        ),
    )
    selected = ranked[0]
    return selected, {
        "selection_rule": "minimum weighted clean-target Born loss; action ID tie-break",
        "eligible_action_count": len(eligible),
        "clean_target_used_for_selection": True,
        "distortion_metadata_used_for_selection": False,
        "oracle_candidates_allowed": config.loss_only_allow_oracle,
        "depth_gate_edit_risk_penalties_used": False,
        "deployable_without_target": False,
    }


__all__ = ["select_loss_only"]
