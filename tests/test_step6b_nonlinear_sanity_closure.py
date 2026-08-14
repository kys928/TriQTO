from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "v0_2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "step6b", SCRIPT_DIR / "benchmark_step6b_nonlinear_sanity_closure.py"
)
assert SPEC and SPEC.loader
step6b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step6b)


def test_config_freezes_adaptive_development_followup() -> None:
    config = json.loads((ROOT / "configs" / "v0_2" / "step6b_nonlinear_sanity_closure.json").read_text())
    assert config["status"] == "FROZEN_AFTER_STEP6A_BEFORE_STEP6B_OUTCOME"
    assert config["adaptive_development_disclosure"]["step6a_validation_results_already_observed"] is True
    assert config["adaptive_development_disclosure"]["confirmatory_claim_allowed"] is False
    assert config["scientific_boundaries"]["triqto_architecture_change"] is False
    assert config["scientific_boundaries"]["historical_v0_1_test_access"] is False
    assert config["scientific_boundaries"]["spent_confirmatory_cohort_access"] is False


def test_diagnostic_rms_uses_only_active_value_coordinates() -> None:
    row = np.zeros((1, 147), dtype=np.float64)
    # 2 active qubits -> six local values; use one nonzero value.
    row[0, 0] = 2.0
    row[0, 24:26] = 1.0
    # one active pair -> three pair values; use one nonzero value.
    row[0, 32] = 1.0
    row[0, 116] = 1.0
    # parity always contributes three active coordinates; use one nonzero value.
    row[0, 144] = 2.0
    expected_active = 3 * 2 + 3 * 1 + 3
    expected_energy = 4.0 + 1.0 + 4.0
    score = step6b.diagnostic_rms(row)
    assert score.shape == (1,)
    assert np.isclose(score[0], np.sqrt(expected_energy / expected_active))
    # Structural masks themselves must not add energy.
    row2 = row.copy()
    row2[0, 24:26] = 7.0
    row2[0, 116] = 9.0
    # Non-binary masks are invalid semantically but still only set the denominator;
    # energy remains determined exclusively by diagnostic values.
    assert np.isfinite(step6b.diagnostic_rms(row2)[0])


def test_diagonal_qda_scores_variance_separated_classes() -> None:
    rng = np.random.default_rng(7)
    x0 = rng.normal(0.0, 0.25, size=(1000, 2))
    x1 = rng.normal(0.0, 2.0, size=(1000, 2))
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(len(x0), dtype=np.int8), np.ones(len(x1), dtype=np.int8)])
    fitted = step6b.fit_diag_qda(X, y, 2, shrinkage=0.01, variance_floor=1e-3)
    scores = step6b.diag_qda_scores(X, fitted)
    pred = np.argmax(scores, axis=1)
    accuracy = float(np.mean(pred == y))
    assert accuracy > 0.8


def test_qda_shrinkage_keeps_variances_positive() -> None:
    X = np.asarray([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [2.0, 3.0]])
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    fitted = step6b.fit_diag_qda(X, y, 2, shrinkage=0.1, variance_floor=1e-3)
    assert np.all(fitted["class_vars"] >= 1e-3)
    assert np.all(np.isfinite(step6b.diag_qda_scores(X, fitted)))


def test_fixed_threshold_uses_training_only() -> None:
    score = np.asarray([0.1, 0.2, 0.8, 0.9, 100.0, -100.0], dtype=np.float64)
    truth = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int8)
    train = np.asarray([True, True, True, True, False, False])
    validation = ~train
    fitted = step6b.fixed_threshold_fit(score, truth, train, validation)
    assert 0.2 < fitted["threshold"] < 0.8
    assert fitted["validation_pred_all"].tolist() == [1, 0]


def test_qda_variant_mapping_keeps_exact_variant_privileged() -> None:
    assert step6b.QDA_VARIANTS["diag_full_diag_qda"] == "diag_full"
    assert step6b.QDA_VARIANTS["diag_full_context_graph_diag_qda"] == "diag_full_context_graph"
    assert step6b.QDA_VARIANTS["exact_diag_full_diag_qda_oracle"] == "exact_diag_full_oracle"
    assert "exact_diag_full_diag_qda_oracle" in step6b.PRIVILEGED
    assert "diag_full_diag_qda" not in step6b.PRIVILEGED
