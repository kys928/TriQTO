"""Hardware-efficient parameterized ansatz circuits."""
from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def make_hardware_efficient_ansatz(n_qubits: int, layers: int = 2, entanglement: str = "linear", measure: bool = True) -> GeneratedCircuit:
    if n_qubits < 1: raise ValueError("hardware_efficient_ansatz requires n_qubits >= 1")
    if layers < 1: raise ValueError("hardware_efficient_ansatz requires layers >= 1")
    if entanglement not in {"linear", "none"}: raise ValueError("entanglement must be 'linear' or 'none'")
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"hea_{n_qubits}q_l{layers}_{entanglement}")
    for layer in range(layers):
        for q in range(n_qubits):
            circuit.ry(Parameter(f"theta_{layer}_{q}"), q)
            circuit.rz(Parameter(f"phi_{layer}_{q}"), q)
        if entanglement != "none" and n_qubits > 1:
            for q in range(n_qubits - 1): circuit.cx(q, q + 1)
    if measure: circuit.measure(range(n_qubits), range(n_qubits))
    return make_generated_circuit(circuit, "hardware_efficient_ansatz", "ansatz", True, n_qubits > 1 and entanglement != "none", {"layers": layers, "entanglement": entanglement, "tags": ["ansatz", "hardware-efficient"]})
