from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
V02 = SCRIPTS / "v0_2"
for value in (str(SCRIPTS), str(V02)):
    if value not in sys.path:
        sys.path.insert(0, value)

import analyze_step14_fresh_holdout_residual_bottleneck as residual


def test_reproduction_audit_accepts_exact_replay() -> None:
    logits = np.asarray([[2.0, 1.0, -1.0], [0.1, 0.2, 0.3]], dtype=np.float64)
    audit = residual.audit_frozen_logit_reproduction(logits, logits.copy())
    assert audit["max_abs_diff"] == 0.0
    assert audit["argmax_mismatch_count"] == 0
    assert audit["unexplained_argmax_mismatch_count"] == 0


def test_reproduction_audit_accepts_only_numerical_near_tie_flip() -> None:
    frozen = np.asarray([[1.0, 1.0 - 1.0e-7, 0.0]], dtype=np.float64)
    replayed = np.asarray([[1.0 - 1.0e-7, 1.0, 0.0]], dtype=np.float64)
    audit = residual.audit_frozen_logit_reproduction(frozen, replayed)
    assert audit["argmax_mismatch_count"] == 1
    assert audit["unexplained_argmax_mismatch_count"] == 0
    assert audit["frozen_logits_authoritative_for_headline_metrics"] is True


def test_reproduction_audit_rejects_logit_drift_above_existing_tolerance() -> None:
    frozen = np.asarray([[1.0, 0.5, 0.0]], dtype=np.float64)
    replayed = np.asarray([[1.0, 0.5002, 0.0]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="exceeded the numerical replay contract"):
        residual.audit_frozen_logit_reproduction(frozen, replayed)


def test_reproduction_audit_rejects_shape_drift() -> None:
    frozen = np.zeros((2, 3), dtype=np.float64)
    replayed = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="shape drift"):
        residual.audit_frozen_logit_reproduction(frozen, replayed)


def test_seed_prediction_distinct_counts_pools_each_frozen_seed() -> None:
    seed_candidate_logits = [
        np.asarray([[[2.0, 1.0, 0.0]], [[2.0, 1.0, 0.0]]], dtype=np.float64),
        np.asarray([[[3.0, 1.0, 0.0]], [[0.0, 3.0, 1.0]]], dtype=np.float64),
    ]
    profiles = np.zeros((2, 1, 3), dtype=np.float64)
    distance = np.zeros((2, 1, 1), dtype=np.float64)
    gate = np.ones((2, 1, 1), dtype=np.bool_)
    mask = np.ones((2, 1), dtype=np.bool_)

    counts = residual.seed_prediction_distinct_counts(
        seed_candidate_logits,
        profiles,
        distance,
        gate,
        mask,
        tau=1.0,
        temperature=1.0,
    )

    np.testing.assert_array_equal(counts, np.asarray([1, 2], dtype=np.int64))


def test_seed_prediction_distinct_counts_rejects_empty_seed_set() -> None:
    with pytest.raises(RuntimeError, match="no candidate logits"):
        residual.seed_prediction_distinct_counts(
            [],
            np.zeros((1, 1, 3), dtype=np.float64),
            np.zeros((1, 1, 1), dtype=np.float64),
            np.ones((1, 1, 1), dtype=np.bool_),
            np.ones((1, 1), dtype=np.bool_),
            tau=1.0,
            temperature=1.0,
        )


def test_quantile_labels_serialize_open_bounds_without_nonfinite_numbers() -> None:
    labels, edges = residual.quantile_labels(
        np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        2,
    )

    np.testing.assert_array_equal(labels, np.asarray(["Q1", "Q1", "Q2", "Q2"], dtype=object))
    assert edges == ["-inf", 1.5, "inf"]
    assert json.loads(json.dumps({"edges": edges}, allow_nan=False)) == {"edges": edges}


def test_quantile_labels_rejects_nonfinite_input() -> None:
    with pytest.raises(RuntimeError, match="nonempty finite vector"):
        residual.quantile_labels(np.asarray([0.0, np.inf], dtype=np.float64), 2)
