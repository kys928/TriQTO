from __future__ import annotations

import pytest

from triqto.training_views import (
    TrainingViewConfig,
    assign_split,
    training_view_config_from_dict,
    training_view_config_to_dict,
)


def test_training_view_config_is_strict_and_roundtrips() -> None:
    config = TrainingViewConfig(
        tasks=("diagnosis", "born_prediction", "hardware_masked"),
        split_seed=17,
    )
    assert training_view_config_from_dict(training_view_config_to_dict(config)) == config
    with pytest.raises(ValueError, match="Unknown training-view config fields"):
        training_view_config_from_dict(
            {**training_view_config_to_dict(config), "unexpected": 1}
        )
    with pytest.raises(TypeError, match="split_seed"):
        TrainingViewConfig(split_seed=True)
    with pytest.raises(ValueError, match="sum to exactly one"):
        TrainingViewConfig(
            train_fraction=0.7,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
    with pytest.raises(ValueError, match="clean_circuit_id"):
        TrainingViewConfig(split_grouping="sample_id")
    with pytest.raises(ValueError, match="exactly 0.0"):
        TrainingViewConfig(topology_loss_weight=0.1)
    with pytest.raises(ValueError, match="fixed Phase 12 order"):
        TrainingViewConfig(tasks=("born_prediction", "diagnosis"))


def test_clean_circuit_split_is_deterministic_and_seeded() -> None:
    config = TrainingViewConfig(tasks=("diagnosis",), split_seed=123)
    first = assign_split("circuit_same", config)
    second = assign_split("circuit_same", config)
    assert first == second
    assert first in {"train", "validation", "test"}
    observed = {
        assign_split(f"circuit_{index}", config)
        for index in range(100)
    }
    assert observed == {"train", "validation", "test"}
