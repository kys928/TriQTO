from __future__ import annotations

import json
from pathlib import Path

from triqto.training_views import (
    TrainingViewConfig,
    load_training_view_config,
    save_training_view_config,
    training_view_config_from_dict,
    training_view_config_to_dict,
)
from triqto.training_views.splits import assign_split


def test_training_view_config_roundtrip_and_split_labels(tmp_path: Path) -> None:
    config = TrainingViewConfig(
        tasks=("diagnosis", "born_prediction"),
        split_seed=123,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        include_topology=False,
        topology_loss_weight=0.0,
    )
    payload = training_view_config_to_dict(config)
    assert payload["tasks"] == ["diagnosis", "born_prediction"]
    assert training_view_config_from_dict(payload) == config

    path = tmp_path / "training_view.yaml"
    save_training_view_config(config, path)
    assert load_training_view_config(path) == config
    loaded = json.loads(json.dumps(payload, sort_keys=True))
    assert training_view_config_from_dict(loaded) == config

    splits = {assign_split(f"clean_circuit_{i}", config) for i in range(20)}
    assert splits <= {"train", "validation", "test"}
    assert "train" in splits
