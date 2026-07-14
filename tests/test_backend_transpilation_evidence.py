from __future__ import annotations

from qiskit import QuantumCircuit
import pytest

from triqto.backends import local_line_backend, transpile_with_evidence


def test_local_fake_backend_snapshot_has_availability_masks() -> None:
    backend = local_line_backend(3)
    assert backend.backend_class == "fake"
    assert backend.feature_available["coupling_map"] is True
    assert backend.feature_available["readout_error_summary"] is False
    assert "readout_error_summary" in backend.missing_reasons
    assert backend.feature_values["coupling_summary"]["directed_edge_count"] == 4


def test_transpilation_evidence_is_deterministic_and_nonzero_missing_not_fabricated() -> None:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    backend = local_line_backend(3)
    first_circuit, first = transpile_with_evidence(circuit, backend, seed=99)
    second_circuit, second = transpile_with_evidence(circuit, backend, seed=99)
    assert first.evidence_id == second.evidence_id
    assert first.backend_id == backend.backend_id
    assert first.depth_before == circuit.depth()
    assert first.size_after == second_circuit.size() == first_circuit.size()
    assert first.two_qubit_gates_after >= first.two_qubit_gates_before


def test_transpilation_rejects_too_small_backend() -> None:
    circuit = QuantumCircuit(4)
    backend = local_line_backend(3)
    with pytest.raises(ValueError, match="more qubits"):
        transpile_with_evidence(circuit, backend)
