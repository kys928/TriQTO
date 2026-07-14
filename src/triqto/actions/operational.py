"""Semantically honest operational actions with explicit preconditions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from triqto.backends import BackendSnapshot, TranspilationEvidence, transpile_with_evidence
from triqto.core.ids import make_deterministic_id
from triqto.simulation import MeasurementSetting, measurement_setting_for


@dataclass(frozen=True, slots=True)
class OperationalActionResult:
    action_type: str
    status: str
    available: bool
    rejection_reason: str | None
    action_id: str
    before_metadata: dict[str, Any]
    after_metadata: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)


def _result(action_type: str, status: str, available: bool, reason: str | None, before: dict[str, Any], after: dict[str, Any], evidence: dict[str, Any] | None = None) -> OperationalActionResult:
    payload = {
        "action_type": action_type,
        "status": status,
        "available": available,
        "rejection_reason": reason,
        "before": before,
        "after": after,
        "evidence": evidence or {},
        "schema": "triqto.operational_action.v1",
    }
    return OperationalActionResult(action_type, status, available, reason, make_deterministic_id("op_action", payload), before, after, evidence or {})


def basis_probe_action(n_qubits: int, bases: tuple[str, ...]) -> OperationalActionResult:
    """Acquire additional basis-conditioned observable evidence; never changes labels by marker."""
    try:
        setting = MeasurementSetting(tuple(str(value).upper() for value in bases))
        if len(setting.bases) != n_qubits:
            raise ValueError("basis probe must specify one basis per qubit")
    except Exception as exc:
        return _result("basis_probe", "rejected", False, str(exc), {"n_qubits": n_qubits}, {})
    return _result(
        "basis_probe",
        "accepted",
        True,
        None,
        {"n_qubits": n_qubits},
        {"measurement_setting": setting.to_metadata(), "probability_domain": "p(y|M)"},
    )


def layout_selection_action(circuit: QuantumCircuit, backend: BackendSnapshot, *, seed: int = 2026) -> tuple[QuantumCircuit | None, OperationalActionResult]:
    if circuit.num_qubits > backend.n_qubits:
        result = _result("layout_selection", "rejected", False, "circuit uses more qubits than backend", {"depth": circuit.depth()}, {})
        return None, result
    transpiled, evidence = transpile_with_evidence(circuit, backend, seed=seed, optimization_level=0, layout_method="trivial", routing_method="basic")
    result = _result(
        "layout_selection",
        "accepted",
        True,
        None,
        {"depth": circuit.depth(), "size": circuit.size()},
        {"depth": transpiled.depth(), "size": transpiled.size()},
        {"transpilation_evidence_id": evidence.evidence_id, "backend_id": evidence.backend_id, "initial_layout": evidence.initial_layout, "final_layout": evidence.final_layout},
    )
    return transpiled, result


def routing_transpilation_action(circuit: QuantumCircuit, backend: BackendSnapshot, *, seed: int = 2026, optimization_level: int = 1) -> tuple[QuantumCircuit | None, OperationalActionResult]:
    if optimization_level not in {0, 1, 2, 3}:
        return None, _result("routing_transpilation", "rejected", False, "optimization_level must be 0, 1, 2, or 3", {}, {})
    try:
        transpiled, evidence = transpile_with_evidence(circuit, backend, seed=seed, optimization_level=optimization_level, layout_method="trivial", routing_method="basic")
    except ValueError as exc:
        return None, _result("routing_transpilation", "rejected", False, str(exc), {"depth": circuit.depth()}, {})
    return transpiled, _result(
        "routing_transpilation",
        "accepted",
        True,
        None,
        {"depth": evidence.depth_before, "size": evidence.size_before},
        {"depth": evidence.depth_after, "size": evidence.size_after},
        asdict(evidence),
    )


def semantics_verified_depth_reduction(circuit: QuantumCircuit, candidate: QuantumCircuit, *, tolerance: float = 1e-10) -> OperationalActionResult:
    """Accept a proposed depth reduction only when statevector semantics and objective improve."""
    if tolerance < 0:
        return _result("depth_reduction", "rejected", False, "tolerance must be nonnegative", {}, {})
    before = {"depth": circuit.depth(), "size": circuit.size()}
    after = {"depth": candidate.depth(), "size": candidate.size()}
    if candidate.depth() >= circuit.depth():
        return _result("depth_reduction", "no_op", False, "candidate does not reduce depth", before, after)
    try:
        original = circuit.remove_final_measurements(inplace=False)
        reduced = candidate.remove_final_measurements(inplace=False)
        fidelity = float(state_fidelity(Statevector.from_instruction(original), Statevector.from_instruction(reduced)))
    except Exception as exc:
        return _result("depth_reduction", "rejected", False, f"semantic verification failed: {exc}", before, after)
    if 1.0 - fidelity > tolerance:
        return _result("depth_reduction", "rejected", False, "semantic distance exceeds tolerance", before, after, {"state_fidelity": fidelity, "tolerance": tolerance})
    return _result("depth_reduction", "accepted", True, None, before, after, {"state_fidelity": fidelity, "tolerance": tolerance})


__all__ = [
    "OperationalActionResult",
    "basis_probe_action",
    "layout_selection_action",
    "routing_transpilation_action",
    "semantics_verified_depth_reduction",
]
