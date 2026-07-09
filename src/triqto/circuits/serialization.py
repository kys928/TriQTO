"""Circuit serialization helpers."""
from __future__ import annotations
import re
from qiskit import QuantumCircuit


def safe_circuit_name(circuit: QuantumCircuit) -> str:
    name = circuit.name or "quantum_circuit"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe or "quantum_circuit"


def circuit_to_qasm3_text(circuit: QuantumCircuit) -> str:
    try:
        from qiskit import qasm3
    except Exception as exc:
        raise RuntimeError("QASM3 export is unavailable in this Qiskit installation") from exc
    try:
        return qasm3.dumps(circuit)
    except Exception as exc:
        raise RuntimeError("QASM3 export failed for this circuit") from exc


def circuit_to_qasm2_text(circuit: QuantumCircuit) -> str:
    if not hasattr(circuit, "qasm"):
        raise RuntimeError("QASM2 export via QuantumCircuit.qasm() is unavailable in this Qiskit version")
    return circuit.qasm()  # type: ignore[attr-defined]
