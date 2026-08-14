from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v0_2/benchmark_step6_cheap_baselines.py"
CONFIG = ROOT / "configs/v0_2/step6_cheap_baseline_benchmark.json"


def load_module():
    spec = importlib.util.spec_from_file_location("triqto_step6_baselines", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_contract_binds_final_step5_product_and_keeps_oracles_privileged() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_BASELINE_OUTCOME"
    assert config["source_dataset"]["product_id"] == "product_b2d78ad2309b71a55f9bb54f"
    assert config["source_dataset"]["clean_circuit_root_count"] == 5000
    assert config["source_dataset"]["example_count"] == 65000
    assert config["scientific_boundaries"]["historical_v0_1_test_access"] is False
    assert config["scientific_boundaries"]["spent_confirmatory_cohort_access"] is False
    assert "exact_diag_full_oracle" not in config["deployable_feature_variants"]
    assert "family_oracle" not in config["deployable_feature_variants"]
    assert set(config["privileged_analysis_variants"]) == {
        "exact_diag_full_oracle",
        "family_oracle",
    }


def test_canonical_diagnostic_layout_reorders_basis_and_maps_pair_slots() -> None:
    module = load_module()
    loaded = {
        "x__diagnostic_basis_codes": np.asarray([2, 0, 1], dtype=np.int8),
        "x__layout_logical_to_physical": np.asarray([0, 1], dtype=np.int16),
        "x__pair_indices": np.asarray([[0, 1]], dtype=np.int16),
        "x__delta_local_expectations": np.asarray(
            [[20.0, 21.0], [0.0, 1.0], [10.0, 11.0]], dtype=np.float64
        ),
        "x__delta_pairwise_correlations": np.asarray(
            [[22.0], [2.0], [12.0]], dtype=np.float64
        ),
        "x__delta_global_parity": np.asarray([23.0, 3.0, 13.0], dtype=np.float64),
        "audit__exact_delta_local_expectations": np.asarray(
            [[120.0, 121.0], [100.0, 101.0], [110.0, 111.0]], dtype=np.float64
        ),
        "audit__exact_delta_pairwise_correlations": np.asarray(
            [[122.0], [102.0], [112.0]], dtype=np.float64
        ),
        "audit__exact_delta_global_parity": np.asarray([123.0, 103.0, 113.0], dtype=np.float64),
    }
    pair_slot = module.fixed_pair_slots(4)
    local, pair, parity = module.canonical_diag_features(
        loaded, max_n=4, pair_slot=pair_slot, exact=False
    )
    assert local.shape == (16,)  # 3*4 local values + 4-qubit mask
    assert pair.shape == (24,)  # 3*C(4,2) values + C(4,2) mask
    assert parity.tolist() == [3.0, 13.0, 23.0]  # fixed Z, X, Y order
    assert local[:4].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert local[4:8].tolist() == [10.0, 11.0, 0.0, 0.0]
    assert local[8:12].tolist() == [20.0, 21.0, 0.0, 0.0]
    first_pair_slot = pair_slot[(0, 1)]
    pair_value_count = 3 * len(pair_slot)
    assert pair[first_pair_slot] == 2.0
    assert pair[len(pair_slot) + first_pair_slot] == 12.0
    assert pair[2 * len(pair_slot) + first_pair_slot] == 22.0
    assert pair[pair_value_count + first_pair_slot] == 1.0


def test_graph_stats_are_finite_and_use_only_deployable_graph_arrays() -> None:
    module = load_module()
    loaded = {
        "x__layout_logical_to_physical": np.asarray([0, 1], dtype=np.int16),
        "x__graph_gate_names": np.asarray(["h", "cx", "rz"]),
        "x__graph_gate_qubit_ptr": np.asarray([0, 1, 3, 4], dtype=np.int32),
        "x__graph_gate_qubit_indices": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "x__graph_gate_parameter_ptr": np.asarray([0, 0, 0, 1], dtype=np.int32),
    }
    features = module.graph_stats(loaded)
    assert features.shape == (10,)
    assert features[0] == 2.0
    assert np.all(np.isfinite(features))
    assert features[4] > 0.0  # multi-qubit gate fraction
    assert features[5] > 0.0  # parameterized gate fraction


def test_training_oof_folds_never_use_validation_roots() -> None:
    module = load_module()
    occurrence = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    train = (occurrence % 5) != 0
    folds = module.training_oof_folds(occurrence, train)
    assert len(folds) == 4
    assert all(not np.any(fold & ~train) for fold in folds)
    covered = np.zeros_like(train)
    for fold in folds:
        assert not np.any(covered & fold)
        covered |= fold
    assert np.array_equal(covered, train)


def test_binary_threshold_selection_balances_class_recall() -> None:
    module = load_module()
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    scores = np.asarray([0.05, 0.2, 0.4, 0.6, 0.8, 0.95], dtype=np.float64)
    threshold, metrics = module.select_binary_threshold(y, scores)
    pred = (scores >= threshold).astype(np.int8)
    assert np.array_equal(pred, y)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["minimum_class_recall"] == 1.0


def test_ridge_oof_selection_and_validation_are_training_group_only() -> None:
    module = load_module()
    # Five occurrence residues repeat; residue 0 is frozen validation, 1..4 are OOF training folds.
    occurrence = np.tile(np.arange(5, dtype=np.int64), 12)
    train = (occurrence % 5) != 0
    validation = ~train
    x = np.linspace(-3.0, 3.0, len(occurrence), dtype=np.float64)
    X = np.stack([x, x**2], axis=1)
    y = (x > 0.0).astype(np.int8)
    result = module.tune_and_fit_ridge(
        X,
        y,
        np.ones(len(y), dtype=bool),
        train,
        validation,
        occurrence,
        2,
        [0.001, 0.01, 0.1],
    )
    assert result["best_lambda"] in {0.001, 0.01, 0.1}
    assert np.isfinite(result["best_threshold"])
    assert result["validation_pred_all"].shape == (int(validation.sum()),)
    assert result["validation_scores_all"].shape == (int(validation.sum()), 1)


def test_class_balanced_multiclass_ridge_scores_all_classes() -> None:
    module = load_module()
    occurrence = np.tile(np.arange(5, dtype=np.int64), 18)
    train = (occurrence % 5) != 0
    validation = ~train
    labels = np.arange(len(occurrence), dtype=np.int64) % 3
    X = np.eye(3, dtype=np.float64)[labels]
    result = module.tune_and_fit_ridge(
        X,
        labels.astype(np.int8),
        np.ones(len(labels), dtype=bool),
        train,
        validation,
        occurrence,
        3,
        [0.001, 0.01],
    )
    assert result["validation_scores_all"].shape == (int(validation.sum()), 3)
    predicted = result["validation_pred_all"]
    assert np.mean(predicted == labels[validation]) > 0.9
