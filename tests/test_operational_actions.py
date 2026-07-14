from __future__ import annotations

from qiskit import QuantumCircuit

from triqto.actions import (
    basis_probe_action,
    layout_selection_action,
    routing_transpilation_action,
    semantics_verified_depth_reduction,
)
from triqto.backends import local_line_backend


def test_basis_probe_accepts_valid_per_qubit_basis_and_rejects_invalid() -> None:
    accepted = basis_probe_action(2, ("X", "Y"))
    assert accepted.status == "accepted"
    assert accepted.after_metadata["probability_domain"] == "p(y|M)"
    rejected = basis_probe_action(2, ("Z",))
    assert rejected.status == "rejected"
    assert rejected.available is False


def test_layout_and_routing_actions_require_backend_capacity() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    backend = local_line_backend(2)
    transpiled, layout = layout_selection_action(circuit, backend)
    assert transpiled is not None
    assert layout.status == "accepted"
    routed, routing = routing_transpilation_action(circuit, backend, optimization_level=1)
    assert routed is not None
    assert routing.evidence["backend_id"] == backend.backend_id

    too_large = QuantumCircuit(3)
    _, rejected = routing_transpilation_action(too_large, backend)
    assert rejected.status == "rejected"
    assert rejected.available is False


def test_semantics_verified_depth_reduction_positive_noop_and_negative() -> None:
    original = QuantumCircuit(1)
    original.h(0)
    original.h(0)
    original.x(0)
    reduced = QuantumCircuit(1)
    reduced.x(0)
    accepted = semantics_verified_depth_reduction(original, reduced)
    assert accepted.status == "accepted"
    assert accepted.evidence["state_fidelity"] > 1.0 - 1e-12

    noop = semantics_verified_depth_reduction(reduced, reduced)
    assert noop.status == "no_op"

    wrong = QuantumCircuit(1)
    wrong.z(0)
    rejected = semantics_verified_depth_reduction(original, wrong)
    assert rejected.status == "rejected"
    assert rejected.rejection_reason == "semantic distance exceeds tolerance"
