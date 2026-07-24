from __future__ import annotations

import numpy as np

from triqto.phase15_6.topology_attachment import (
    attach_topology_to_item,
    classify_group_membership,
    fit_feature_scaler,
    transform_feature_vector,
)


def test_cross_split_cohort_is_audit_only() -> None:
    members, missing, splits, split_groups, status = classify_group_membership(
        group_kind="family_qubit_cohort",
        metadata={},
        group_key="ghz:4",
        point_ids=["s1", "s2"],
        entity_split={"s1": "train", "s2": "test"},
        entity_split_group={"s1": "g1", "s2": "g2"},
    )
    assert members == ("s1", "s2")
    assert missing == ()
    assert splits == ("test", "train")
    assert split_groups == ("g1", "g2")
    assert status == "audit_only_cross_split"


def test_action_neighborhood_resolves_metadata_sample() -> None:
    members, missing, splits, _groups, status = classify_group_membership(
        group_kind="action_neighborhood",
        metadata={"sample_id": "s1"},
        group_key="ignored",
        point_ids=["action1", "action2"],
        entity_split={"s1": "validation"},
        entity_split_group={"s1": "g1"},
    )
    assert members == ("s1",)
    assert missing == ()
    assert splits == ("validation",)
    assert status == "attachable_same_split"


def test_scaler_masks_infinity_and_is_finite() -> None:
    scaler = fit_feature_scaler("essential_count", [0.0, 1.0, 3.0, np.inf])
    transformed, finite, positive_inf, negative_inf = transform_feature_vector(
        np.asarray([np.inf]),
        {
            "columns": [
                {
                    "name": scaler.name,
                    "transform": scaler.transform,
                    "center": scaler.center,
                    "scale": scaler.scale,
                }
            ]
        },
        clip_value=10.0,
    )
    assert transformed.tolist() == [0.0]
    assert finite.tolist() == [False]
    assert positive_inf.tolist() == [True]
    assert negative_inf.tolist() == [False]


def test_joint_attachment_enables_only_safe_heads() -> None:
    source = {
        "x_input_group_names": np.asarray(
            ["circuit_graph", "topology"], dtype="<U16"
        ),
        "x_input_group_available_mask": np.asarray([True, False]),
        "y_joint_head_names": np.asarray(
            ["diagnosis", "action_ranking", "born_prediction", "topology_audit"],
            dtype="<U20",
        ),
        "y_joint_head_input_group_names": np.asarray(
            ["circuit_graph", "topology"], dtype="<U16"
        ),
        "y_joint_head_input_mask": np.zeros((4, 2), dtype=bool),
    }
    topology = {
        "x_topology_features": np.asarray([0.25], dtype=np.float32),
        "x_topology_available_mask": np.asarray(True),
    }
    result = attach_topology_to_item(
        source,
        topology,
        task="joint_multitask",
        attach_hardware=False,
        enable_joint_diagnosis=True,
    )
    assert result["x_input_group_available_mask"].tolist() == [True, True]
    mask = result["y_joint_head_input_mask"]
    assert mask[:, 1].tolist() == [True, False, False, True]
    assert not any(name.startswith("y_topology") for name in result)
