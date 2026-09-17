from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "v0_2"))

import fit_step15_frame_ranker as base
import fit_step15_frame_ranker_true_positive as fixed


SPEC = {
    "minimum_same_axis_abs_cosine": 0.98,
    "minimum_identity_assignment_margin": 0.02,
}


def test_literal_true_candidate_is_positive_even_when_axes_are_assignment_ambiguous() -> None:
    # Three identical response axes make identity vs permutation assignment
    # intrinsically ambiguous. The threshold is therefore not a valid reason
    # to label the literal true candidate negative.
    true_exact = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert base.response_equivalent(true_exact, true_exact, SPEC) is False
    assert fixed.response_equivalent_with_literal_identity(true_exact, true_exact, SPEC) is True


def test_nonliteral_candidates_still_use_frozen_equivalence_thresholds() -> None:
    true_exact = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    alternative = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert not np.array_equal(alternative, true_exact)
    assert fixed.response_equivalent_with_literal_identity(alternative, true_exact, SPEC) == base.response_equivalent(
        alternative, true_exact, SPEC
    )


def test_runpod_fit_command_uses_literal_positive_entrypoint() -> None:
    source = (ROOT / "scripts" / "runpod_step15_worker.py").read_text(encoding="utf-8")
    assert 'fit_step15_frame_ranker_true_positive.py' in source
    assert 'evaluate_step15_spent_holdout.py' in source
