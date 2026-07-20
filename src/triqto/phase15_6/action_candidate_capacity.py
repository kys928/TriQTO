"""Losslessly resolve variable Phase 12 action-set capacities before construction.

Phase 12 action-ranking items intentionally retain every candidate generated for a
sample.  The campaign configuration therefore acts as a requested operational
capacity, not a scientific truncation rule.  This module derives the exact maximum
action-set size from immutable Phase 9 manifests, expands the operational capacity
when necessary, and keeps separate fail-closed hard safety limits.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import time
from typing import Any


DEFAULT_ACTION_CANDIDATE_HARD_LIMIT = 65_536
DEFAULT_ACTION_SOURCE_REF_HARD_LIMIT = 262_144
_BATCH_SIZE = 65_536
_SCHEMA = "triqto.phase15_6.action_candidate_capacity_resolution.v1"
_BASE_ACTION_SOURCE_REFS = 2
_SOURCE_REFS_PER_CANDIDATE = 3


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _parquet(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to resolve Phase 12 action capacities"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"required manifest does not exist: {path}")
    return pq.ParquetFile(path)


def _log(message: str) -> None:
    print(f"[Phase 12][action-capacity] {message}", flush=True)


def _sample_counts(path: Path, label: str) -> Counter[str]:
    parquet = _parquet(path)
    total_rows = int(parquet.metadata.num_rows) if parquet.metadata is not None else 0
    total_batches = max(1, math.ceil(total_rows / _BATCH_SIZE))
    counts: Counter[str] = Counter()
    completed = 0
    started = time.monotonic()
    _log(f"scanning {label} | rows={total_rows:,} | batches={total_batches}")
    for batch_index, batch in enumerate(
        parquet.iter_batches(columns=["sample_id"], batch_size=_BATCH_SIZE),
        start=1,
    ):
        for sample_id in batch.column(0).to_pylist():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{label} contains an invalid sample_id")
            counts[sample_id] += 1
        completed += int(batch.num_rows)
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed / elapsed
        remaining = max(total_rows - completed, 0)
        eta = remaining / rate if rate > 0.0 else 0.0
        _log(
            f"{label} batch={batch_index}/{total_batches} | "
            f"rows={completed:,}/{total_rows:,} | rate={rate:,.0f}/s | "
            f"ETA≈{eta/60.0:.1f}m"
        )
    if total_rows and completed != total_rows:
        raise ValueError(f"{label} row count changed during scan")
    if not counts:
        raise ValueError(f"{label} contains no sample action rows")
    return counts


def _count_mismatch_examples(
    candidate_counts: Counter[str],
    rollout_counts: Counter[str],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for sample_id in sorted(set(candidate_counts) | set(rollout_counts)):
        candidates = int(candidate_counts.get(sample_id, 0))
        rollouts = int(rollout_counts.get(sample_id, 0))
        if candidates != rollouts:
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "candidate_count": candidates,
                    "rollout_count": rollouts,
                }
            )
            if len(mismatches) == 10:
                break
    return mismatches


def resolve_action_candidate_capacity(
    phase9_root: str | Path,
    requested_max_candidates: int,
    requested_max_source_refs: int,
    *,
    hard_candidate_limit: int = DEFAULT_ACTION_CANDIDATE_HARD_LIMIT,
    hard_source_ref_limit: int = DEFAULT_ACTION_SOURCE_REF_HARD_LIMIT,
) -> tuple[int, int, dict[str, Any]]:
    """Return lossless effective action and source-reference capacities.

    The action-ranking view stores two fixed Phase 8 provenance references and three
    Phase 9 references per retained candidate.  The source-reference capacity is
    therefore expanded alongside the candidate capacity so the next guardrail cannot
    reject the same scientifically valid variable-size action set.
    """
    requested_candidates = _require_positive_int(
        requested_max_candidates,
        "requested_max_candidates",
    )
    requested_refs = _require_positive_int(
        requested_max_source_refs,
        "requested_max_source_refs",
    )
    hard_candidates = _require_positive_int(
        hard_candidate_limit,
        "hard_candidate_limit",
    )
    hard_refs = _require_positive_int(
        hard_source_ref_limit,
        "hard_source_ref_limit",
    )
    if hard_candidates < requested_candidates:
        raise ValueError(
            "hard_candidate_limit must be at least requested_max_candidates"
        )
    if hard_refs < requested_refs:
        raise ValueError(
            "hard_source_ref_limit must be at least requested_max_source_refs"
        )

    phase9 = Path(phase9_root)
    candidate_counts = _sample_counts(
        phase9 / "manifests" / "action_candidate_manifest.parquet",
        "Phase 9 candidate manifest",
    )
    rollout_counts = _sample_counts(
        phase9 / "manifests" / "action_rollout_manifest.parquet",
        "Phase 9 rollout manifest",
    )
    if candidate_counts != rollout_counts:
        raise ValueError(
            "Phase 9 candidate/rollout sample counts differ; "
            f"mismatch_examples={_count_mismatch_examples(candidate_counts, rollout_counts)}"
        )

    required_candidates = max(candidate_counts.values())
    required_refs = (
        _BASE_ACTION_SOURCE_REFS
        + _SOURCE_REFS_PER_CANDIDATE * required_candidates
    )
    effective_candidates = max(requested_candidates, required_candidates)
    effective_refs = max(requested_refs, required_refs)

    if effective_candidates > hard_candidates:
        raise RuntimeError(
            "Phase 12 requires an action-ranking item with "
            f"{required_candidates:,} candidates, exceeding the hard operational "
            f"limit {hard_candidates:,}. Increase the hard limit only after reviewing "
            "memory and artifact-size cost; no candidates were sampled or truncated."
        )
    if effective_refs > hard_refs:
        raise RuntimeError(
            "Phase 12 requires capacity for up to "
            f"{required_refs:,} action source references, exceeding the hard "
            f"operational limit {hard_refs:,}. Increase the hard limit only after "
            "reviewing artifact-size cost; no source references were omitted."
        )

    largest = sorted(
        candidate_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]
    metadata = {
        "schema": _SCHEMA,
        "requested_max_candidates_per_item": requested_candidates,
        "required_max_candidates_per_item": required_candidates,
        "effective_max_candidates_per_item": effective_candidates,
        "hard_candidate_limit": hard_candidates,
        "requested_max_source_refs_per_item": requested_refs,
        "required_action_source_ref_capacity": required_refs,
        "effective_max_source_refs_per_item": effective_refs,
        "hard_source_ref_limit": hard_refs,
        "base_action_source_refs": _BASE_ACTION_SOURCE_REFS,
        "source_refs_per_candidate": _SOURCE_REFS_PER_CANDIDATE,
        "sample_count": len(candidate_counts),
        "candidate_row_count": sum(candidate_counts.values()),
        "rollout_row_count": sum(rollout_counts.values()),
        "largest_action_sets": [
            {"sample_id": sample_id, "candidate_count": int(count)}
            for sample_id, count in largest
        ],
        "candidate_capacity_auto_expanded": (
            effective_candidates > requested_candidates
        ),
        "source_ref_capacity_auto_expanded": effective_refs > requested_refs,
        "auto_expanded": (
            effective_candidates > requested_candidates
            or effective_refs > requested_refs
        ),
        "action_set_policy": (
            "lossless_variable_size_auto_expand_no_sampling_no_truncation"
        ),
    }
    _log(
        "resolved | "
        f"requested_candidates={requested_candidates:,} | "
        f"required_candidates={required_candidates:,} | "
        f"effective_candidates={effective_candidates:,} | "
        f"effective_source_refs={effective_refs:,} | "
        "policy=no_sampling_no_truncation"
    )
    return effective_candidates, effective_refs, metadata


__all__ = [
    "DEFAULT_ACTION_CANDIDATE_HARD_LIMIT",
    "DEFAULT_ACTION_SOURCE_REF_HARD_LIMIT",
    "resolve_action_candidate_capacity",
]
