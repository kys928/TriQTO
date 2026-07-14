from __future__ import annotations

import pytest

from triqto.evaluation.generalization_tests import SplitDefinition, assign_axis_holdout, assign_iid_split, audit_axis_disjointness


def records():
    return [
        {"sample_id": "s1", "clean_circuit_id": "c1", "family": "ghz", "n_qubits": 4, "distortion_type": "rx", "backend_id": "b1"},
        {"sample_id": "s2", "clean_circuit_id": "c2", "family": "qft", "n_qubits": 6, "distortion_type": "rz", "backend_id": "b2"},
        {"sample_id": "s3", "clean_circuit_id": "c3", "family": "ghz", "n_qubits": 4, "distortion_type": "rz", "backend_id": "b1"},
        {"sample_id": "s4", "clean_circuit_id": "c4", "family": "bell", "n_qubits": 8, "distortion_type": "rx", "backend_id": "b3"},
    ]


@pytest.mark.parametrize("axis,value", [("family", "qft"), ("n_qubits", "6"), ("distortion_type", "rz"), ("backend_id", "b2")])
def test_axis_holdouts_are_exclusive(axis: str, value: str) -> None:
    definition = SplitDefinition(axis=axis, heldout_values=(value,), split_name=f"ood_{axis}")
    assignment = assign_axis_holdout(records(), definition)
    audit = audit_axis_disjointness(records(), assignment, definition)
    assert audit["audited_disjointness"] is True
    assert audit["claim_label"] == f"ood_{axis}"
    for row in records():
        if str(row[axis]) == value:
            assert assignment[row["sample_id"]] == "test"
        else:
            assert assignment[row["sample_id"]] in {"train", "validation"}


def test_impossible_holdout_fails_closed() -> None:
    definition = SplitDefinition(axis="family", heldout_values=("missing",), split_name="bad")
    with pytest.raises(ValueError, match="absent"):
        assign_axis_holdout(records(), definition)


def test_lineage_leakage_fails_closed() -> None:
    rows = records() + [{"sample_id": "s5", "clean_circuit_id": "c2", "family": "ghz", "n_qubits": 4, "distortion_type": "rx", "backend_id": "b1"}]
    definition = SplitDefinition(axis="family", heldout_values=("qft",), split_name="ood_family")
    with pytest.raises(ValueError, match="crosses heldout"):
        assign_axis_holdout(rows, definition)


def test_iid_split_is_labeled_iid_test_not_generalization() -> None:
    assignment = assign_iid_split(records())
    assert set(assignment.values()) <= {"train", "validation", "iid_test"}
