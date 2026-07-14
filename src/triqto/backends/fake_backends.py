"""Stable local fake-backend fixtures and deterministic transpilation evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from triqto.core.ids import make_deterministic_id

from .backend_metadata import BackendSnapshot, backend_snapshot_id, summarize_coupling_map


@dataclass(frozen=True, slots=True)
class TranspilationEvidence:
    evidence_id: str
    backend_id: str
    transpiler_seed: int
    optimization_level: int
    routing_method: str | None
    layout_method: str | None
    initial_layout: tuple[int, ...] | None
    final_layout: tuple[int, ...] | None
    depth_before: int
    depth_after: int
    size_before: int
    size_after: int
    two_qubit_gates_before: int
    two_qubit_gates_after: int
    swap_count: int
    basis_gates: tuple[str, ...]


def local_line_backend(n_qubits: int = 5, *, name: str = "triqto_local_line_fake_v1") -> BackendSnapshot:
    if n_qubits < 2:
        raise ValueError("local fake backend requires at least two qubits")
    coupling = tuple((i, i + 1) for i in range(n_qubits - 1)) + tuple((i + 1, i) for i in range(n_qubits - 1))
    basis = ("id", "rz", "sx", "x", "cx", "measure")
    feature_values = {"coupling_summary": summarize_coupling_map(n_qubits, coupling)}
    feature_available = {
        "coupling_map": True,
        "basis_gates": True,
        "readout_error_summary": False,
        "gate_error_summary": False,
        "duration_summary": False,
        "t1_t2_summary": False,
    }
    missing = {
        "readout_error_summary": "stable local fixture has no calibration data",
        "gate_error_summary": "stable local fixture has no calibration data",
        "duration_summary": "stable local fixture has no pulse durations",
        "t1_t2_summary": "stable local fixture has no coherence data",
    }
    payload = {
        "backend_name": name,
        "backend_source": "triqto.local_fixture",
        "backend_class": "fake",
        "n_qubits": n_qubits,
        "basis_gates": list(basis),
        "coupling_map": [list(edge) for edge in coupling],
        "calibration_timestamp": None,
        "feature_values": feature_values,
        "feature_available": feature_available,
        "missing_reasons": missing,
    }
    return BackendSnapshot(backend_id=backend_snapshot_id(payload), **payload)  # type: ignore[arg-type]


def _count_two_qubit(circuit: QuantumCircuit) -> int:
    return sum(1 for inst in circuit.data if len(inst.qubits) == 2)


def _final_layout_tuple(transpiled: QuantumCircuit) -> tuple[int, ...] | None:
    layout = getattr(transpiled, "layout", None)
    final_layout = getattr(layout, "final_layout", None)
    if final_layout is None:
        return None
    try:
        return tuple(int(final_layout[i]) for i in range(transpiled.num_qubits))
    except Exception:
        return None


def transpile_with_evidence(circuit: QuantumCircuit, backend: BackendSnapshot, *, seed: int = 2026, optimization_level: int = 1, layout_method: str | None = "trivial", routing_method: str | None = "basic") -> tuple[QuantumCircuit, TranspilationEvidence]:
    if backend.backend_class not in {"fake", "simulator"}:
        raise ValueError("offline transpilation evidence requires fake/simulator backend")
    if circuit.num_qubits > backend.n_qubits:
        raise ValueError("circuit uses more qubits than backend snapshot")
    coupling = CouplingMap([list(edge) for edge in backend.coupling_map])
    transpiled = transpile(
        circuit,
        basis_gates=list(backend.basis_gates),
        coupling_map=coupling,
        seed_transpiler=seed,
        optimization_level=optimization_level,
        layout_method=layout_method,
        routing_method=routing_method,
    )
    initial_layout = tuple(range(circuit.num_qubits)) if layout_method == "trivial" else None
    swap_count = int(transpiled.count_ops().get("swap", 0))
    payload: dict[str, Any] = {
        "backend_id": backend.backend_id,
        "transpiler_seed": seed,
        "optimization_level": optimization_level,
        "routing_method": routing_method,
        "layout_method": layout_method,
        "initial_layout": list(initial_layout) if initial_layout is not None else None,
        "final_layout": list(_final_layout_tuple(transpiled)) if _final_layout_tuple(transpiled) is not None else None,
        "depth_before": circuit.depth(),
        "depth_after": transpiled.depth(),
        "size_before": circuit.size(),
        "size_after": transpiled.size(),
        "two_qubit_gates_before": _count_two_qubit(circuit),
        "two_qubit_gates_after": _count_two_qubit(transpiled),
        "swap_count": swap_count,
        "basis_gates": list(backend.basis_gates),
    }
    evidence = TranspilationEvidence(evidence_id=make_deterministic_id("transpile", {"schema": "triqto.transpile.evidence.v1", **payload}), **payload)  # type: ignore[arg-type]
    return transpiled, evidence


def describe_contract() -> str:
    return "Stable local fake-backend fixtures and deterministic transpilation evidence; no network calls."


__all__ = ["TranspilationEvidence", "local_line_backend", "transpile_with_evidence"]
