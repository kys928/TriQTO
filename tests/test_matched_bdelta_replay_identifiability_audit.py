from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/audit_matched_bdelta_replay_identifiability.py"
)
SPEC = importlib.util.spec_from_file_location("matched_bdelta_replay_audit_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(index: int, name: str, qubits: list[int], params: list[float] | None = None):
    return {
        "index": index,
        "name": name,
        "qubits": qubits,
        "params": list(params or []),
    }


def test_nonterminal_rz_propagates_into_population_change() -> None:
    records = [
        record(0, "h", [0]),
        record(1, "rz", [0], [0.4]),
        record(2, "h", [0]),
    ]
    clean = MODULE.simulate_records(records, 1, removed_index=1)
    distorted = MODULE.simulate_records(records, 1)
    clean_p = np.abs(clean) ** 2
    distorted_p = np.abs(distorted) ** 2
    assert np.allclose(clean_p, np.array([1.0, 0.0]), atol=1e-12)
    assert distorted_p[1] > 0.0
    decomposition = MODULE.BASE.overlap_decomposition(clean, distorted, epsilon=1e-12)
    assert decomposition["population_component"] > decomposition["phase_component"]


def test_replacement_changes_only_midcircuit_axis() -> None:
    records = [
        record(0, "h", [0]),
        record(1, "rx", [0], [0.2]),
        record(2, "rz", [0], [0.3]),
        record(3, "h", [0]),
    ]
    rx_state = MODULE.simulate_records(
        records, 1, removed_index=1, replacement_axis="rx", replacement_angle=0.2
    )
    original = MODULE.simulate_records(records, 1)
    assert MODULE.replay_error(original, rx_state)["overlap_loss"] < 1e-12
    rz_state = MODULE.simulate_records(
        records, 1, removed_index=1, replacement_axis="rz", replacement_angle=0.2
    )
    assert MODULE.replay_error(original, rz_state)["overlap_loss"] > 1e-6


def test_replay_error_is_global_phase_invariant() -> None:
    state = np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0)
    shifted = state * np.exp(1j * 0.731)
    error = MODULE.replay_error(state, shifted)
    assert error["overlap_loss"] < 1e-12
    assert error["aligned_max_abs_error"] < 1e-12


def test_two_qubit_suffix_replay_with_cx() -> None:
    records = [
        record(0, "h", [0]),
        record(1, "ry", [0], [0.15]),
        record(2, "cx", [0, 1]),
        record(3, "rz", [1], [0.4]),
    ]
    clean = MODULE.simulate_records(records, 2, removed_index=1)
    distorted = MODULE.simulate_records(records, 2)
    assert np.isclose(np.linalg.norm(clean), 1.0)
    assert np.isclose(np.linalg.norm(distorted), 1.0)
    assert MODULE.replay_error(clean, distorted)["overlap_loss"] > 0.0
