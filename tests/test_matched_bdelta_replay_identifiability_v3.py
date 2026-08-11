from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/audit_matched_bdelta_replay_identifiability_v3.py"
)
SPEC = importlib.util.spec_from_file_location("matched_bdelta_replay_v3_test", SCRIPT)
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


def test_terminal_measurement_markers_are_excluded_from_statevector_replay() -> None:
    unitary = [
        record(0, "h", [0]),
        record(1, "rx", [0], [0.2]),
    ]
    with_measurement = unitary + [record(2, "measure", [0])]
    expected = MODULE.simulate_records(unitary, 1)
    actual = MODULE.simulate_records(with_measurement, 1)
    assert MODULE.V2.replay_error(expected, actual)["overlap_loss"] < 1e-12


def test_unitary_after_measurement_tail_is_refused() -> None:
    records = [
        record(0, "h", [0]),
        record(1, "measure", [0]),
        record(2, "x", [0]),
    ]
    with pytest.raises(ValueError, match="after measurement tail began"):
        MODULE.simulate_records(records, 1)


def test_measurement_tail_does_not_make_last_unitary_nonterminal() -> None:
    records = [
        record(0, "h", [0]),
        record(1, "rz", [0], [0.3]),
        record(2, "measure", [0]),
    ]
    fraction, depth_bin, count, terminal = MODULE.unitary_insertion_bin(records, 1)
    assert count == 2
    assert terminal is True
    assert fraction == 1.0
    assert depth_bin == "late_75_100"


def test_measurement_tail_count_matches_per_qubit_readout_shape() -> None:
    records = [
        record(0, "h", [0]),
        record(1, "cx", [0, 1]),
        record(2, "measure", [0]),
        record(3, "measure", [1]),
    ]
    assert MODULE.validate_measurement_tail(records) == 2
    state = MODULE.simulate_records(records, 2)
    assert np.isclose(np.linalg.norm(state), 1.0)
