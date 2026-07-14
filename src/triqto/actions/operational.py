"""Semantically honest offline operational actions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from triqto.backends import BackendSnapshot, transpile_with_evidence
from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.simulation import MeasurementSetting, simulate_ideal_shots
from .identities import circuit_semantic_hash

OPERATIONAL_ACTION_SCHEMA = "triqto.operational_action.v2"
_FAMILY = {
    "basis_probe": "diagnostic_evidence_acquisition",
    "layout_selection": "compilation_layout",
    "routing_transpilation": "compilation_routing",
    "depth_reduction": "semantics_preserving_optimization",
}


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
    action_family: str = ""
    source_circuit_id: str | None = None
    source_backend_snapshot_id: str | None = None
    preconditions: dict[str, Any] = field(default_factory=dict)
    availability_mask: bool = False
    required_evidence_tier: str = "offline"
    semantic_validation_method: str | None = None
    semantic_tolerance: float | None = None
    objective_comparison: dict[str, Any] = field(default_factory=dict)
    changes_circuit_semantics: bool = False
    acquires_evidence: bool = False
    physical_hardware: bool = False
    privileged_information: bool = False
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def operational_action_payload(result: OperationalActionResult, *, include_content_hash: bool = False) -> dict[str, Any]:
    payload = result.to_dict()
    payload["schema"] = OPERATIONAL_ACTION_SCHEMA
    if not include_content_hash:
        payload.pop("content_hash", None)
    return payload


def _hash(payload: Mapping[str, Any]) -> str:
    return make_deterministic_id("operational_action_content", {"schema": OPERATIONAL_ACTION_SCHEMA, "payload": canonical_json(dict(payload))})


def operational_action_content_hash(result: OperationalActionResult) -> str:
    return _hash(operational_action_payload(result))


def _cost(circuit: QuantumCircuit) -> dict[str, int]:
    return {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "two_qubit_gate_count": sum(1 for item in circuit.data if len(item.qubits) == 2),
    }


def _result(action_type: str, status: str, reason: str | None, before: dict[str, Any], after: dict[str, Any], evidence: dict[str, Any] | None = None, **kwargs: Any) -> OperationalActionResult:
    available = status == "accepted"
    payload = {
        "schema": OPERATIONAL_ACTION_SCHEMA,
        "action_type": action_type,
        "action_family": _FAMILY[action_type],
        "status": status,
        "available": available,
        "availability_mask": available,
        "rejection_reason": reason,
        "before_metadata": before,
        "after_metadata": after,
        "evidence": evidence or {},
        "source_circuit_id": kwargs.get("source_circuit_id"),
        "source_backend_snapshot_id": kwargs.get("source_backend_snapshot_id"),
        "preconditions": kwargs.get("preconditions", {}),
        "required_evidence_tier": kwargs.get("required_evidence_tier", "offline"),
        "semantic_validation_method": kwargs.get("semantic_validation_method"),
        "semantic_tolerance": kwargs.get("semantic_tolerance"),
        "objective_comparison": kwargs.get("objective_comparison", {}),
        "changes_circuit_semantics": kwargs.get("changes_circuit_semantics", False),
        "acquires_evidence": kwargs.get("acquires_evidence", False),
        "physical_hardware": kwargs.get("physical_hardware", False),
        "privileged_information": False,
    }
    action_id = make_deterministic_id("op_action", payload)
    content_hash = _hash({**payload, "action_id": action_id})
    return OperationalActionResult(
        action_type=action_type, status=status, available=available,
        rejection_reason=reason, action_id=action_id, before_metadata=before,
        after_metadata=after, evidence=evidence or {}, action_family=_FAMILY[action_type],
        source_circuit_id=payload["source_circuit_id"],
        source_backend_snapshot_id=payload["source_backend_snapshot_id"],
        preconditions=payload["preconditions"], availability_mask=available,
        required_evidence_tier=payload["required_evidence_tier"],
        semantic_validation_method=payload["semantic_validation_method"],
        semantic_tolerance=payload["semantic_tolerance"],
        objective_comparison=payload["objective_comparison"],
        changes_circuit_semantics=payload["changes_circuit_semantics"],
        acquires_evidence=payload["acquires_evidence"],
        physical_hardware=payload["physical_hardware"], privileged_information=False,
        content_hash=content_hash,
    )


def basis_probe_action(n_qubits: int, bases: tuple[str, ...], *, circuit: QuantumCircuit | None = None, shots: int = 1024, seed: int = 2026, evidence_tier: str = "ideal_simulator") -> OperationalActionResult:
    source_id = circuit_semantic_hash(circuit) if circuit is not None else None
    before: dict[str, Any] = {"n_qubits": n_qubits, **(_cost(circuit) if circuit is not None else {})}
    try:
        setting = MeasurementSetting(tuple(str(value).upper() for value in bases))
        if len(setting.bases) != n_qubits:
            raise ValueError("basis probe must specify one basis per qubit")
        if circuit is not None and circuit.num_qubits != n_qubits:
            raise ValueError("basis probe n_qubits must match source circuit")
        if shots <= 0:
            raise ValueError("basis probe shots must be positive")
        if evidence_tier != "ideal_simulator" and circuit is not None:
            raise ValueError("offline probe execution currently supports ideal_simulator only")
    except Exception as exc:
        return _result("basis_probe", "rejected", str(exc), before, {}, source_circuit_id=source_id, preconditions={"one_basis_per_qubit": True, "shots_positive": True}, required_evidence_tier=evidence_tier, acquires_evidence=True, physical_hardware=evidence_tier == "physical_hardware")
    after = {"measurement_setting": setting.to_metadata(), "probability_domain": "p(y|M)", "evidence_namespace": "diagnostic_probe", "evidence_role": "diagnostic_probe", "source_circuit_preserved": True}
    evidence: dict[str, Any] = {"measurement_setting": setting.to_metadata(), "evidence_tier": evidence_tier, "physical_hardware": False}
    if circuit is not None:
        probe = simulate_ideal_shots(circuit.copy(), shots=shots, seed=seed, measurement_basis=setting)
        evidence.update({"shots": shots, "seed": seed, "counts": dict(sorted(probe.counts.items())), "probabilities": dict(sorted(probe.probabilities.items()))})
        evidence["probe_evidence_id"] = make_deterministic_id("basis_probe_evidence", {"source_circuit_id": source_id, "setting": setting.setting_id_payload, "shots": shots, "seed": seed, "counts": evidence["counts"]})
    return _result("basis_probe", "accepted", None, before, after, evidence, source_circuit_id=source_id, preconditions={"one_basis_per_qubit": True, "shots_positive": True}, required_evidence_tier=evidence_tier, semantic_validation_method="immutable_probe_copy", objective_comparison={"objective": "new_basis_conditioned_evidence", "improved": bool(evidence.get("probe_evidence_id"))}, acquires_evidence=True)


def _transpile_action(action_type: str, circuit: QuantumCircuit, backend: BackendSnapshot, *, seed: int, optimization_level: int, layout_method: str | None, routing_method: str | None) -> tuple[QuantumCircuit | None, OperationalActionResult]:
    source_id, before = circuit_semantic_hash(circuit), _cost(circuit)
    preconditions = {"backend_snapshot_present": True, "coupling_map_present": bool(backend.coupling_map), "basis_gates_present": bool(backend.basis_gates), "backend_capacity_sufficient": circuit.num_qubits <= backend.n_qubits}
    if backend.backend_class not in {"fake", "simulator"}:
        return None, _result(action_type, "rejected", "offline compilation requires fake/simulator backend evidence", before, {}, source_circuit_id=source_id, source_backend_snapshot_id=backend.backend_id, preconditions=preconditions, required_evidence_tier="fake_backend_fixture", physical_hardware=backend.backend_class == "physical")
    try:
        transpiled, ev = transpile_with_evidence(circuit, backend, seed=seed, optimization_level=optimization_level, layout_method=layout_method, routing_method=routing_method)
    except ValueError as exc:
        return None, _result(action_type, "rejected", str(exc), before, {}, source_circuit_id=source_id, source_backend_snapshot_id=backend.backend_id, preconditions=preconditions, required_evidence_tier="fake_backend_fixture")
    after = _cost(transpiled)
    objective = {"depth_delta": after["depth"] - before["depth"], "size_delta": after["size"] - before["size"], "two_qubit_gate_delta": after["two_qubit_gate_count"] - before["two_qubit_gate_count"], "swap_count": ev.swap_count}
    evidence = {**asdict(ev), "backend_class": backend.backend_class, "backend_source": backend.backend_source, "coupling_map": [list(edge) for edge in backend.coupling_map]}
    return transpiled, _result(action_type, "accepted", None, before, after, evidence, source_circuit_id=source_id, source_backend_snapshot_id=backend.backend_id, preconditions=preconditions, required_evidence_tier="fake_backend_fixture", semantic_validation_method="backend_bound_transpilation_evidence", objective_comparison=objective)


def layout_selection_action(circuit: QuantumCircuit, backend: BackendSnapshot, *, seed: int = 2026) -> tuple[QuantumCircuit | None, OperationalActionResult]:
    return _transpile_action("layout_selection", circuit, backend, seed=seed, optimization_level=1, layout_method="sabre", routing_method="basic")


def routing_transpilation_action(circuit: QuantumCircuit, backend: BackendSnapshot, *, seed: int = 2026, optimization_level: int = 1) -> tuple[QuantumCircuit | None, OperationalActionResult]:
    if optimization_level not in {0, 1, 2, 3}:
        before = _cost(circuit)
        return None, _result("routing_transpilation", "rejected", "optimization_level must be 0, 1, 2, or 3", before, {}, source_circuit_id=circuit_semantic_hash(circuit), source_backend_snapshot_id=backend.backend_id, required_evidence_tier="fake_backend_fixture")
    return _transpile_action("routing_transpilation", circuit, backend, seed=seed, optimization_level=optimization_level, layout_method="trivial", routing_method="basic")


def semantics_verified_depth_reduction(circuit: QuantumCircuit, candidate: QuantumCircuit, *, tolerance: float = 1e-10, protect_two_qubit_count: bool = True) -> OperationalActionResult:
    source_id, before, after = circuit_semantic_hash(circuit), _cost(circuit), _cost(candidate)
    objective = {"candidate_circuit_id": circuit_semantic_hash(candidate), "depth_delta": after["depth"] - before["depth"], "size_delta": after["size"] - before["size"], "two_qubit_gate_delta": after["two_qubit_gate_count"] - before["two_qubit_gate_count"]}
    objective["objective_improved"] = any(objective[key] < 0 for key in ("depth_delta", "size_delta", "two_qubit_gate_delta"))
    if tolerance < 0 or circuit.num_qubits != candidate.num_qubits:
        return _result("depth_reduction", "rejected", "invalid tolerance or qubit-count mismatch", before, after, source_circuit_id=source_id, semantic_validation_method="statevector_fidelity", semantic_tolerance=tolerance, objective_comparison=objective)
    if not objective["objective_improved"]:
        return _result("depth_reduction", "no_op", "candidate does not improve an objective circuit-cost measure", before, after, source_circuit_id=source_id, semantic_validation_method="statevector_fidelity", semantic_tolerance=tolerance, objective_comparison=objective)
    if protect_two_qubit_count and objective["two_qubit_gate_delta"] > 0:
        return _result("depth_reduction", "rejected", "candidate regresses protected two-qubit-gate count", before, after, source_circuit_id=source_id, semantic_validation_method="statevector_fidelity", semantic_tolerance=tolerance, objective_comparison=objective)
    try:
        fidelity = float(state_fidelity(Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)), Statevector.from_instruction(candidate.remove_final_measurements(inplace=False))))
    except Exception as exc:
        return _result("depth_reduction", "rejected", f"semantic verification failed: {exc}", before, after, source_circuit_id=source_id, semantic_validation_method="statevector_fidelity", semantic_tolerance=tolerance, objective_comparison=objective)
    evidence = {"state_fidelity": fidelity, "semantic_distance": 1.0 - fidelity, "tolerance": tolerance, "candidate_circuit_id": objective["candidate_circuit_id"]}
    if 1.0 - fidelity > tolerance:
        return _result("depth_reduction", "rejected", "semantic distance exceeds tolerance", before, after, evidence, source_circuit_id=source_id, semantic_validation_method="statevector_fidelity", semantic_tolerance=tolerance, objective_comparison=objective)
    return _result("depth_reduction", "accepted", None, before, after, evidence, source_circuit_id=source_id, required_evidence_tier="ideal_simulator", semantic_validation_method="statevector_fidelity", semantic_tolerance=tolerance, objective_comparison=objective)


__all__ = ["OPERATIONAL_ACTION_SCHEMA", "OperationalActionResult", "operational_action_content_hash", "operational_action_payload", "basis_probe_action", "layout_selection_action", "routing_transpilation_action", "semantics_verified_depth_reduction"]
