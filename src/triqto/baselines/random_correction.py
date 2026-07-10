"""Deterministic random-correction sanity baseline for Phase 10."""
from __future__ import annotations

import hashlib
from typing import Any

from .config import BaselineSuiteConfig


def _no_op_rollout(rollouts: tuple[Any, ...], candidates_by_id: dict[str, Any]) -> Any:
    no_ops = [
        rollout
        for rollout in rollouts
        if not candidates_by_id[rollout.action_id].edits
    ]
    if len(no_ops) != 1:
        raise ValueError("Each Phase 9 sample must expose exactly one no-op rollout")
    return no_ops[0]


def select_random_correction(
    *,
    sample_id: str,
    rollouts: tuple[Any, ...],
    candidates_by_id: dict[str, Any],
    config: BaselineSuiteConfig,
) -> tuple[Any, dict[str, Any]]:
    """Choose one eligible Phase 9 candidate without consulting clean-target metrics."""
    eligible: list[Any] = []
    for rollout in rollouts:
        candidate = candidates_by_id[rollout.action_id]
        if not config.random_include_no_op and not candidate.edits:
            continue
        if (
            not config.random_allow_oracle
            and "oracle_inverse" in candidate.generation_sources
        ):
            continue
        eligible.append(rollout)
    eligible.sort(key=lambda item: item.action_id)
    fallback = False
    if not eligible:
        eligible = [_no_op_rollout(rollouts, candidates_by_id)]
        fallback = True
    digest = hashlib.sha256(
        f"triqto-phase10-random:{config.random_seed}:{sample_id}".encode("utf-8")
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(eligible)
    selected = eligible[index]
    return selected, {
        "selection_rule": "sha256(seed, sample_id) modulo sorted eligible action IDs",
        "eligible_action_count": len(eligible),
        "selected_index": index,
        "fallback_to_no_op": fallback,
        "clean_target_used_for_selection": False,
        "distortion_metadata_used_for_selection": False,
        "oracle_candidates_allowed": config.random_allow_oracle,
        "no_op_allowed_in_random_pool": config.random_include_no_op,
    }


__all__ = ["select_random_correction"]
