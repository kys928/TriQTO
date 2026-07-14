from __future__ import annotations

import pytest

from triqto.evaluation.baseline_comparison import build_comparison_records, comparison_id, validate_unique_comparisons


def test_comparison_id_includes_task_view_ablation_and_execution_mode() -> None:
    first = comparison_id(run_id="r", sample_id="s", baseline_id="b", task="action_ranking", view_id="v", execution_mode="ideal", ablation_id="a")
    second = comparison_id(run_id="r", sample_id="s", baseline_id="b", task="basis_probe", view_id="v", execution_mode="ideal", ablation_id="a")
    third = comparison_id(run_id="r", sample_id="s", baseline_id="b", task="action_ranking", view_id="v2", execution_mode="ideal", ablation_id="a")
    fourth = comparison_id(run_id="r", sample_id="s", baseline_id="b", task="action_ranking", view_id="v", execution_mode="hardware_masked", ablation_id="a")
    assert len({first, second, third, fourth}) == 4


def test_enabled_multiple_action_tasks_do_not_collide() -> None:
    records = build_comparison_records(
        run_id="phase15_run",
        sample_id="sample_1",
        baselines=["random_correction", "rule_only"],
        tasks=["action_ranking", "basis_probe"],
        view_id="phase12_view",
        execution_mode="ideal_statevector",
    )
    assert len(records) == 4
    assert len({row["comparison_id"] for row in records}) == 4


def test_conflicting_duplicate_comparisons_fail_closed() -> None:
    cid = comparison_id(run_id="r", sample_id="s", baseline_id="b", task="t", view_id="v", execution_mode="ideal")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        validate_unique_comparisons([
            {"comparison_id": cid, "task": "t", "score": 1.0},
            {"comparison_id": cid, "task": "t", "score": 2.0},
        ])
