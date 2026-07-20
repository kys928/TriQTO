from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from triqto.phase15_6.action_candidate_capacity import (
    resolve_action_candidate_capacity,
)


def _write_phase9(
    root: Path,
    counts: dict[str, int],
    *,
    rollout_counts: dict[str, int] | None = None,
) -> Path:
    phase9 = root / "phase9"
    manifests = phase9 / "manifests"
    manifests.mkdir(parents=True)
    candidate_sample_ids = [
        sample_id
        for sample_id, count in sorted(counts.items())
        for _ in range(count)
    ]
    resolved_rollouts = counts if rollout_counts is None else rollout_counts
    rollout_sample_ids = [
        sample_id
        for sample_id, count in sorted(resolved_rollouts.items())
        for _ in range(count)
    ]
    pq.write_table(
        pa.table({"sample_id": candidate_sample_ids}),
        manifests / "action_candidate_manifest.parquet",
    )
    pq.write_table(
        pa.table({"sample_id": rollout_sample_ids}),
        manifests / "action_rollout_manifest.parquet",
    )
    return phase9


def test_action_capacity_auto_expands_without_truncation(tmp_path: Path) -> None:
    phase9 = _write_phase9(tmp_path, {"sample-small": 15, "sample-large": 218})

    effective_candidates, effective_refs, metadata = (
        resolve_action_candidate_capacity(
            phase9,
            requested_max_candidates=128,
            requested_max_source_refs=512,
        )
    )

    assert effective_candidates == 218
    assert effective_refs == 656
    assert metadata["requested_max_candidates_per_item"] == 128
    assert metadata["required_max_candidates_per_item"] == 218
    assert metadata["effective_max_candidates_per_item"] == 218
    assert metadata["required_action_source_ref_capacity"] == 656
    assert metadata["candidate_capacity_auto_expanded"] is True
    assert metadata["source_ref_capacity_auto_expanded"] is True
    assert metadata["candidate_row_count"] == 233
    assert metadata["rollout_row_count"] == 233
    assert metadata["largest_action_sets"][0] == {
        "sample_id": "sample-large",
        "candidate_count": 218,
    }
    assert metadata["action_set_policy"] == (
        "lossless_variable_size_auto_expand_no_sampling_no_truncation"
    )


def test_action_capacity_keeps_larger_requested_guardrails(tmp_path: Path) -> None:
    phase9 = _write_phase9(tmp_path, {"sample-a": 15, "sample-b": 20})

    effective_candidates, effective_refs, metadata = (
        resolve_action_candidate_capacity(
            phase9,
            requested_max_candidates=256,
            requested_max_source_refs=2048,
        )
    )

    assert effective_candidates == 256
    assert effective_refs == 2048
    assert metadata["auto_expanded"] is False


def test_action_capacity_rejects_candidate_rollout_count_mismatch(
    tmp_path: Path,
) -> None:
    phase9 = _write_phase9(
        tmp_path,
        {"sample-a": 15, "sample-b": 20},
        rollout_counts={"sample-a": 15, "sample-b": 19},
    )

    with pytest.raises(ValueError, match="candidate/rollout sample counts differ"):
        resolve_action_candidate_capacity(
            phase9,
            requested_max_candidates=128,
            requested_max_source_refs=512,
        )


def test_action_capacity_fails_closed_above_hard_limit(tmp_path: Path) -> None:
    phase9 = _write_phase9(tmp_path, {"sample-large": 218})

    with pytest.raises(RuntimeError, match="no candidates were sampled or truncated"):
        resolve_action_candidate_capacity(
            phase9,
            requested_max_candidates=128,
            requested_max_source_refs=512,
            hard_candidate_limit=200,
        )
