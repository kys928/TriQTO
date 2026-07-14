from __future__ import annotations

import pytest
from types import SimpleNamespace

from triqto.training_views import (
    TrainingViewConfig,
    assign_split,
    build_sample_split_maps,
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
    holdout = TrainingViewConfig(
        tasks=("diagnosis",),
        split_strategy="axis_holdout",
        holdout_axis="family",
        holdout_values=("ghz",),
        train_fraction=0.8,
        validation_fraction=0.2,
        test_fraction=0.0,
    )
    assert training_view_config_from_dict(
        training_view_config_to_dict(holdout)
    ) == holdout
    with pytest.raises(ValueError, match="test_fraction=0"):
        TrainingViewConfig(
            split_strategy="axis_holdout",
            holdout_axis="family",
            holdout_values=("ghz",),
        )


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


def test_family_holdout_is_axis_disjoint_and_backend_fails_closed() -> None:
    phase7 = SimpleNamespace(
        samples=[
            SimpleNamespace(
                sample_id="bell_sample",
                clean_circuit_id="bell_circuit",
                family="bell",
                n_qubits=2,
                distortion_id="distortion_a",
                metadata={},
            ),
            SimpleNamespace(
                sample_id="ghz_sample",
                clean_circuit_id="ghz_circuit",
                family="ghz",
                n_qubits=3,
                distortion_id="distortion_b",
                metadata={},
            ),
        ],
        distortions=[
            SimpleNamespace(
                distortion_id="distortion_a",
                distortion_type="rx_overrotation",
            ),
            SimpleNamespace(
                distortion_id="distortion_b",
                distortion_type="phase_rz_drift",
            ),
        ],
    )
    family_config = TrainingViewConfig(
        tasks=("diagnosis",),
        split_strategy="axis_holdout",
        holdout_axis="family",
        holdout_values=("ghz",),
        train_fraction=0.8,
        validation_fraction=0.2,
        test_fraction=0.0,
    )
    splits, groups = build_sample_split_maps(phase7, family_config)
    assert splits["ghz_sample"] == "test"
    assert splits["bell_sample"] in {"train", "validation"}
    assert groups["ghz_sample"] == "ghz_circuit"

    backend_config = TrainingViewConfig(
        tasks=("diagnosis",),
        split_strategy="axis_holdout",
        holdout_axis="backend_id",
        holdout_values=("fake_backend_a",),
        train_fraction=0.8,
        validation_fraction=0.2,
        test_fraction=0.0,
    )
    with pytest.raises(ValueError, match="backend_feature_unavailable"):
        build_sample_split_maps(phase7, backend_config)
