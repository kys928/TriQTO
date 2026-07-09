"""Parameterized phase-interference circuits."""
from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def make_phase_interference_circuit(n_qubits: int, layers: int = 1, measure: bool = True) -> GeneratedCircuit:
    if n_qubits < 1: raise ValueError("phase_interference requires n_qubits >= 1")
    if layers < 1: raise ValueError("phase_interference requires layers >= 1")
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"phase_interference_{n_qubits}q_l{layers}")
    for q in range(n_qubits): circuit.h(q)
    for layer in range(layers):
        for q in range(n_qubits):
            circuit.rz(Parameter(f"phi_{layer}_{q}"), q)
            circuit.rx(Parameter(f"theta_{layer}_{q}"), q)
        for q in range(n_qubits): circuit.h(q)
    if measure: circuit.measure(range(n_qubits), range(n_qubits))
    return make_generated_circuit(circuit, "phase_interference", "phase_interference", True, False, {"layers": layers, "tags": ["phase", "interference", "parameterized"]})
