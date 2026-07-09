"""Tests for TriQTO manifest schema contracts."""
from __future__ import annotations

import pytest

from triqto.storage.schema import CircuitRecord, MetricRecord, TrainingViewRecord


def test_circuit_record_round_trips_as_dict() -> None:
    record = CircuitRecord(
        circuit_id="circuit_abc",
        family="ghz",
        n_qubits=4,
        n_clbits=4,
        depth=3,
        two_qubit_gate_count=3,
        parameter_count=0,
        metadata={"source": "unit-test"},
    )
    record.validate()
    clone = CircuitRecord.from_dict(record.to_dict())
    assert clone == record


def test_circuit_record_rejects_zero_qubits() -> None:
    record = CircuitRecord("circuit_bad", "ghz", 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        record.validate()


def test_metric_record_keeps_hilbert_mask_explicit() -> None:
    record = MetricRecord("metric_1", "run_1", "circuit_1", None)
    assert record.hilbert_available_mask is False


def test_training_view_record_requires_split() -> None:
    record = TrainingViewRecord("view_1", "diagnosis", [], [], [], "hardware_masked", "")
    with pytest.raises(ValueError):
        record.validate()
