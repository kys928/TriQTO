from __future__ import annotations

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
