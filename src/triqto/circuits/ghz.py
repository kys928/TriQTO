"""GHZ state-preparation circuit family."""
from __future__ import annotations
from qiskit import QuantumCircuit
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def make_ghz_circuit(n_qubits: int, measure: bool = True) -> GeneratedCircuit:
    if n_qubits < 2:
        raise ValueError("GHZ circuits require n_qubits >= 2")
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"ghz_{n_qubits}q")
    circuit.h(0)
    for q in range(n_qubits - 1):
        circuit.cx(q, q + 1)
    if measure:
        circuit.measure(range(n_qubits), range(n_qubits))
    return make_generated_circuit(circuit, "ghz", "state_preparation", False, True, {
        "phase_sensitivity_note": "standard computational-basis GHZ preparation",
        "cnot_chain": [[q, q + 1] for q in range(n_qubits - 1)],
        "tags": ["ghz", "entangled", "state-preparation"],
    })
