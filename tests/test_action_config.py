from __future__ import annotations

import json

import pytest

from triqto.actions import (
    ActionEdit,
    ActionEngineConfig,
    action_config_from_dict,
    action_config_to_dict,
    action_engine_id,
    action_operational_config_id,
    action_risk_score,
    action_rollout_id,
    load_action_config,
    save_action_config,
)


def test_action_config_strict_roundtrip(tmp_path):
    config = ActionEngineConfig(candidate_magnitudes=(0.1, 0.2))
    assert action_config_from_dict(action_config_to_dict(config)) == config
    path = tmp_path / "action.json"
    save_action_config(config, path)
    assert load_action_config(path) == config
    assert json.loads(path.read_text())["candidate_magnitudes"] == [0.1, 0.2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": "other"},
        {"candidate_magnitudes": ()},
        {"candidate_magnitudes": (0.2, 0.1)},
        {"candidate_magnitudes": (0.1, 0.1)},
        {"candidate_magnitudes": (True,)},
        {"candidate_magnitudes": ("0.1",)},
        {"include_no_op": 1},
        {"max_candidates_per_sample": True},
        {"max_edits_per_action": 0},
        {"max_abs_angle": float("nan")},
        {"observed_edges_only": False},
        {"reward_total_variation_weight": -1.0},
    ],
)
def test_action_config_rejects_malformed_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ActionEngineConfig(**kwargs)


def test_action_config_rejects_unknown_and_nonfinite_json(tmp_path):
    payload = action_config_to_dict(ActionEngineConfig())
    with pytest.raises(ValueError):
        action_config_from_dict({**payload, "split": "train"})
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":"triqto.action.phase9.v1","max_abs_angle":NaN}')
    with pytest.raises(ValueError):
        load_action_config(path)


def test_operational_guardrails_do_not_change_scientific_engine_identity():
    first = ActionEngineConfig(max_candidates_per_sample=128, max_edits_per_action=16)
    second = ActionEngineConfig(max_candidates_per_sample=1024, max_edits_per_action=128)
    assert action_engine_id("generation_x", "graphconv_x", first) == action_engine_id(
        "generation_x", "graphconv_x", second
    )
    assert action_operational_config_id(first) != action_operational_config_id(second)


def test_operational_edit_guardrail_does_not_change_risk_or_rollout_identity():
    edits = (ActionEdit("append_rx", (0,), 0.2),)
    first = ActionEngineConfig(max_edits_per_action=2)
    second = ActionEngineConfig(max_edits_per_action=200)
    assert action_risk_score(edits, first) == action_risk_score(edits, second)
    assert action_rollout_id("action_x", "run_x", first) == action_rollout_id(
        "action_x", "run_x", second
    )


def test_scientific_reward_configuration_changes_rollout_identity():
    first = ActionEngineConfig(reward_total_variation_weight=1.0)
    second = ActionEngineConfig(reward_total_variation_weight=2.0)
    assert action_rollout_id("action_x", "run_x", first) != action_rollout_id(
        "action_x", "run_x", second
    )
