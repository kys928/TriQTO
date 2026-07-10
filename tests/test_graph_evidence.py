from __future__ import annotations

import math

import numpy as np
import pytest

from triqto.graph import (
    decode_born_metric_arrays,
    validate_born_metric_arrays,
    validate_count_mapping,
    validate_probability_arrays,
    validate_probability_mapping,
)


@pytest.mark.parametrize("value", [True, "0.5", None, math.nan, math.inf, -math.inf])
def test_probability_values_reject_permissive_coercion(value):
    with pytest.raises((TypeError, ValueError)):
        validate_probability_mapping({"0": value, "1": 1.0}, 1)


def test_probability_tiny_positive_preserved_and_small_negative_clipped():
    bitstrings, values, clipped = validate_probability_mapping(
        {"0": -1e-12, "1": 1.0},
        1,
    )
    assert bitstrings.tolist() == ["0", "1"]
    assert values.tolist() == [0.0, 1.0]
    assert clipped == 1
    tiny = 1e-15
    _, preserved, _ = validate_probability_mapping(
        {"0": tiny, "1": 1.0 - tiny},
        1,
    )
    assert preserved[0] == tiny


def test_probability_arrays_reject_duplicate_outcomes():
    with pytest.raises(ValueError, match="duplicate"):
        validate_probability_arrays(
            np.asarray(["0", "0"], dtype="<U1"),
            np.asarray([0.5, 0.5], dtype=np.float64),
            1,
        )


@pytest.mark.parametrize("value", [True, "2", 2.0, -1])
def test_counts_are_strict_nonnegative_integers(value):
    with pytest.raises((TypeError, ValueError)):
        validate_count_mapping({"0": value, "1": 2}, 1, 4)


def test_counts_total_must_equal_shots():
    with pytest.raises(ValueError, match="does not equal shots"):
        validate_count_mapping({"0": 1, "1": 2}, 1, 4)


def test_born_metric_positive_infinity_roundtrip_contract():
    names, values, mask = decode_born_metric_arrays(
        {
            "hellinger": 0.2,
            "kl_divergence": None,
            "kl_divergence__nonfinite": "positive_infinity",
        }
    )
    assert names.tolist() == ["hellinger", "kl_divergence"]
    assert values.tolist() == [0.2, 0.0]
    assert mask.tolist() == [False, True]
    validate_born_metric_arrays(names, values, mask)


@pytest.mark.parametrize(
    "payload",
    [
        {"x": None},
        {"x__nonfinite": "positive_infinity"},
        {"x": 1.0, "x__nonfinite": "positive_infinity"},
        {"x": None, "x__nonfinite": "negative_infinity"},
        {"x": math.nan},
        {"x": -math.inf},
    ],
)
def test_born_metric_malformed_encodings_rejected(payload):
    with pytest.raises((TypeError, ValueError)):
        decode_born_metric_arrays(payload)
