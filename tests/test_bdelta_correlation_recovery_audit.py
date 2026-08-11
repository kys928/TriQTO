from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/audit_bdelta_correlation_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("step4_1_corr_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def ghz_phase_state(phi: float) -> np.ndarray:
    state = np.zeros(8, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2.0)
    state[7] = np.exp(1j * phi) / np.sqrt(2.0)
    return state


def test_ghz_relative_phase_is_invisible_locally_but_visible_in_global_x_parity() -> None:
    left = MODULE.summaries_for_state(ghz_phase_state(0.0), 3)
    right = MODULE.summaries_for_state(ghz_phase_state(np.pi), 3)
    assert np.allclose(left["X"]["local"], right["X"]["local"], atol=1e-12)
    assert abs(float(left["X"]["global_parity"]) - float(right["X"]["global_parity"])) > 1.9


def test_bell_state_has_strong_same_basis_zz_pair_correlation() -> None:
    state = np.zeros(4, dtype=np.complex128)
    state[0] = state[3] = 1.0 / np.sqrt(2.0)
    summary = MODULE.summaries_for_state(state, 2)
    assert summary["Z"]["pairwise"].shape == (1,)
    assert abs(float(summary["Z"]["pairwise"][0]) - 1.0) < 1e-12


def test_variant_score_adds_global_parity_information() -> None:
    summary = {
        basis: {
            "local": np.zeros(3),
            "pairwise": np.zeros(3),
            "global_parity": 1.0 if basis == "X" else 0.0,
        }
        for basis in ("X", "Y", "Z")
    }
    local = {"include_local": True, "include_pairwise": False, "include_global_parity": False}
    parity = {"include_local": True, "include_pairwise": False, "include_global_parity": True}
    assert MODULE.variant_score(summary, local) == 0.0
    assert MODULE.variant_score(summary, parity) > 0.0


def test_pair_difference_preserves_joint_terms() -> None:
    left = {
        basis: {
            "local": np.asarray([0.0, 0.0]),
            "pairwise": np.asarray([0.5]),
            "global_parity": 0.5,
        }
        for basis in ("X", "Y", "Z")
    }
    right = {
        basis: {
            "local": np.asarray([0.0, 0.0]),
            "pairwise": np.asarray([-0.5]),
            "global_parity": -0.5,
        }
        for basis in ("X", "Y", "Z")
    }
    diff = MODULE.pair_difference(left, right)
    assert np.allclose(diff["X"]["local"], 0.0)
    assert np.allclose(diff["X"]["pairwise"], 1.0)
    assert abs(float(diff["X"]["global_parity"]) - 1.0) < 1e-12
