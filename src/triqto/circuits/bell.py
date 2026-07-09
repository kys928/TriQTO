"""Bell-pair circuit family."""
from __future__ import annotations
from qiskit import QuantumCircuit
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def make_bell_circuit(n_qubits: int = 2, measure: bool = True) -> GeneratedCircuit:
    if n_qubits < 2:
        raise ValueError("Bell circuits require n_qubits >= 2")
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"bell_{n_qubits}q")
    circuit.h(0); circuit.cx(0, 1)
    if measure:
        circuit.measure(range(n_qubits), range(n_qubits))
    idle_qubits = list(range(2, n_qubits))
    return make_generated_circuit(circuit, "bell", "bell_pair", False, True, {
        "phase_sensitivity_note": "computational-basis Bell pair; global phase is not probed",
        "idle_qubits": idle_qubits,
        "has_idle_extra_qubits": bool(idle_qubits),
        "tags": ["bell", "entangled", "idle-extra-qubits"] if idle_qubits else ["bell", "entangled"],
    })
