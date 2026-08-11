from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/audit_generalized_bdelta_identifiability.py"
)
SPEC = importlib.util.spec_from_file_location("generalized_bdelta_audit_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(name: str, qubits: list[int], params: list[float] | None = None):
    return {
        "name": name,
        "qubits": qubits,
        "params": list(params or []),
    }


def test_depth_boundaries_are_distinct_and_terminal_is_preserved() -> None:
    rows = MODULE.depth_boundaries(4, [0.25, 0.5, 0.75, 1.0])
    assert [row["boundary_rank"] for row in rows] == [1, 2, 3, 4]
    assert [row["depth_bin"] for row in rows] == [
        "early",
        "middle",
        "late",
        "terminal",
    ]
    assert rows[-1]["terminal_insertion"] is True


def test_short_circuit_depth_targets_are_not_double_counted() -> None:
    rows = MODULE.depth_boundaries(2, [0.25, 0.5, 0.75, 1.0])
    assert [row["boundary_rank"] for row in rows] == [1, 2]
    assert len(rows) == 2
    assert rows[0]["target_fractions"] == "0.25|0.50"
    assert rows[1]["target_fractions"] == "0.75|1.00"
    assert rows[1]["depth_bin"] == "terminal"


def test_prefix_suffix_counterfactual_matches_direct_insertion() -> None:
    records = [
        record("h", [0]),
        record("cx", [0, 1]),
        record("rz", [1], [0.4]),
    ]
    boundary = 1
    prefix = MODULE.prefix_state(records, 2, boundary)
    suffix = MODULE.suffix_operator(records, 2, boundary)
    rotated_prefix = MODULE.BASE.apply_single_qubit_matrix(
        prefix,
        1,
        MODULE.BASE.rotation_matrix("rx", 0.2),
    )
    actual = MODULE.propagate_suffix(rotated_prefix, suffix)

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rx(0.2, 1)
    circuit.cx(0, 1)
    circuit.rz(0.4, 1)
    expected = np.asarray(Statevector.from_instruction(circuit).data)
    expected /= np.linalg.norm(expected)

    error = MODULE.V2.replay_error(expected, actual)
    assert error["overlap_loss"] < 1e-12
    assert error["aligned_max_abs_error"] < 1e-12


def test_qubit_position_classes_remove_q0_only_assumption() -> None:
    assert MODULE.qubit_position_class(0, 4) == "first_q0"
    assert MODULE.qubit_position_class(1, 4) == "interior"
    assert MODULE.qubit_position_class(2, 4) == "interior"
    assert MODULE.qubit_position_class(3, 4) == "last_qN_minus_1"


def test_shortcut_gate_downgrades_base_identifiable_result() -> None:
    base = {"status": "IDENTIFIABLE"}
    shortcut = {"pass": False}
    status, reason = MODULE.apply_shortcut_gate(base, shortcut)
    assert status == "CONTEXT_DEPENDENT"
    assert "shortcut-removal" in reason


def test_shortcut_gate_retains_identifiable_when_all_subsets_pass() -> None:
    base = {"status": "IDENTIFIABLE"}
    shortcut = {"pass": True}
    status, _ = MODULE.apply_shortcut_gate(base, shortcut)
    assert status == "IDENTIFIABLE"
