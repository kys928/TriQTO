from __future__ import annotations

import json

import pytest

from triqto.graph import (
    GraphConversionConfig,
    graph_config_from_dict,
    graph_config_to_dict,
    load_graph_config,
    save_graph_config,
)


def test_graph_config_strict_roundtrip_and_persistence(tmp_path):
    config = GraphConversionConfig(
        max_gate_events=23,
        max_probability_outcomes=17,
        include_supplemental_counts=False,
    )
    assert graph_config_from_dict(graph_config_to_dict(config)) == config
    path = tmp_path / "graph_config.json"
    save_graph_config(config, path)
    assert load_graph_config(path) == config
    assert json.loads(path.read_text())["max_gate_events"] == 23


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": "other"},
        {"max_gate_events": True},
        {"max_gate_events": 0},
        {"max_probability_outcomes": "4"},
        {"include_supplemental_counts": 1},
        {"reject_conditioned_operations": False},
    ],
)
def test_graph_config_rejects_untruthful_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GraphConversionConfig(**kwargs)


def test_graph_config_rejects_unknown_and_nonfinite_json(tmp_path):
    payload = graph_config_to_dict(GraphConversionConfig())
    with pytest.raises(ValueError):
        graph_config_from_dict({**payload, "split": "train"})
    path = tmp_path / "bad.json"
    path.write_text('{"max_gate_events": NaN}')
    with pytest.raises(ValueError):
        load_graph_config(path)
