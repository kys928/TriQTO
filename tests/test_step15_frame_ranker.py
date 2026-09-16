from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "v0_2"))

import step14_equivalence_aware_common as equiv
import step15_frame_ranker_common as ranker


def geometry(count: int) -> tuple[np.ndarray, np.ndarray]:
    distance = np.ones((count, count), dtype=np.float32)
    gate = np.zeros((count, count), dtype=np.bool_)
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(gate, True)
    return distance, gate


def test_zero_anchor_correction_is_exact_step14_pooling() -> None:
    rng = np.random.default_rng(17)
    logits = rng.normal(size=(5, 3))
    profiles = rng.normal(size=(5, 3))
    distance, gate = geometry(5)
    baseline = equiv.equivalence_aware_pool_one(
        logits, profiles, distance, gate, tau=0.02, frame_temperature=2.0
    )
    recovered = ranker.equivalence_aware_pool_one_with_anchor_correction(
        logits,
        profiles,
        distance,
        gate,
        np.zeros(5, dtype=np.float64),
        tau=0.02,
        frame_temperature=2.0,
    )
    assert np.array_equal(recovered, baseline)


def test_rank_features_do_not_name_privileged_truth_fields() -> None:
    lowered = " ".join(ranker.FEATURE_NAMES).lower()
    for forbidden in ("true", "affected", "injection", "exact", "target"):
        assert forbidden not in lowered


def test_query_wide_profile_offset_cancels_from_rank_features() -> None:
    rng = np.random.default_rng(3)
    canonical = rng.normal(size=(4, 24)).astype(np.float32)
    profiles = rng.normal(size=(4, 3))
    distance, gate = geometry(4)
    candidates = [(0, 0), (0, 2), (1, 1), (1, 3)]
    a = ranker.build_rank_features_one(
        canonical,
        profiles,
        distance,
        gate,
        candidates,
        qubit_count=2,
        gate_count=4,
        tau=0.02,
        frame_temperature=2.0,
    )
    b = ranker.build_rank_features_one(
        canonical,
        profiles + 91.25,
        distance,
        gate,
        candidates,
        qubit_count=2,
        gate_count=4,
        tau=0.02,
        frame_temperature=2.0,
    )
    np.testing.assert_allclose(a, b, atol=2e-5, rtol=0.0)


def test_hard_negatives_are_frozen_baseline_near_ties() -> None:
    features = np.zeros((1, 5, len(ranker.FEATURE_NAMES)), dtype=np.float32)
    features[0, :, 0] = np.asarray([3.0, 2.9, 2.8, 1.0, -1.0])
    valid = np.ones((1, 5), dtype=np.bool_)
    positive = np.asarray([[True, False, False, False, False]])
    hard = ranker.hard_negative_mask(features, valid, positive, hard_negative_count=2)
    assert hard.tolist() == [[False, True, True, False, False]]


def test_ranking_metrics_are_equivalence_set_aware() -> None:
    scores = np.asarray([[5.0, 4.0, 3.0], [1.0, 3.0, 2.0]])
    valid = np.ones((2, 3), dtype=np.bool_)
    positives = np.asarray([[False, True, True], [False, True, False]], dtype=np.bool_)
    metric = ranker.ranking_metrics(scores, valid, positives)
    assert metric["top1"] == 0.5
    assert metric["top3"] == 1.0
    assert metric["mrr"] == 0.75
    assert 0.0 <= metric["set_brier"] <= 1.0
    assert metric["set_nll"] >= 0.0


def test_linear_rank_adapter_can_learn_a_reordering_signal() -> None:
    rng = np.random.default_rng(41)
    bags = 96
    candidates = 4
    feature_count = len(ranker.FEATURE_NAMES)
    features = np.zeros((bags, candidates, feature_count), dtype=np.float32)
    features[:, :, 0] = np.asarray([2.0, 1.5, 1.0, 0.5], dtype=np.float32)
    signal_index = ranker.FEATURE_NAMES.index("canonical_00")
    features[:, :, signal_index] = rng.normal(scale=0.05, size=(bags, candidates))
    features[:, 3, signal_index] += 3.0
    valid = np.ones((bags, candidates), dtype=np.bool_)
    positive = np.zeros_like(valid)
    positive[:, 3] = True

    before = ranker.ranking_metrics(features[:, :, 0], valid, positive)
    fitted = ranker.train_linear_rank_adapter(
        features,
        valid,
        positive,
        learning_rate=0.05,
        weight_decay=1e-4,
        pairwise_weight=0.5,
        pairwise_margin=0.25,
        hard_negative_count=2,
        epochs=30,
        batch_size=32,
        seed=7,
        device=torch.device("cpu"),
    )
    after_scores = ranker.adapter_scores(features, valid, fitted)
    after = ranker.ranking_metrics(after_scores, valid, positive)
    assert before["top1"] == 0.0
    assert after["top1"] > 0.95
    assert after["mrr"] > before["mrr"]


def test_json_adapter_schema_roundtrip() -> None:
    adapter = {
        "weights": np.zeros(len(ranker.FEATURE_NAMES), dtype=np.float32),
        "mean": np.zeros(len(ranker.FEATURE_NAMES), dtype=np.float32),
        "std": np.ones(len(ranker.FEATURE_NAMES), dtype=np.float32),
        "seed": 9,
        "hard_negative_count": 2,
        "final_loss": {"total": 1.0, "listwise": 1.0, "pairwise": 0.0, "l2": 0.0},
    }
    restored = ranker.adapter_from_json(ranker.adapter_to_json(adapter))
    np.testing.assert_array_equal(restored["weights"], adapter["weights"])
    np.testing.assert_array_equal(restored["mean"], adapter["mean"])
    np.testing.assert_array_equal(restored["std"], adapter["std"])
