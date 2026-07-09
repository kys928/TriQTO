"""Lightweight QFT-like phase-structure circuits."""
from __future__ import annotations
import math
from qiskit import QuantumCircuit
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def make_qft_like_circuit(n_qubits: int, measure: bool = True, include_swaps: bool = False) -> GeneratedCircuit:
    if n_qubits < 2: raise ValueError("qft_like requires n_qubits >= 2")
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"qft_like_{n_qubits}q")
    for target in range(n_qubits):
        circuit.h(target)
        for control in range(target + 1, n_qubits):
            circuit.cp(math.pi / (2 ** (control - target)), control, target)
    if include_swaps:
        for q in range(n_qubits // 2): circuit.swap(q, n_qubits - q - 1)
    if measure: circuit.measure(range(n_qubits), range(n_qubits))
    return make_generated_circuit(circuit, "qft_like", "phase_structure", True, True, {"include_swaps": include_swaps, "tags": ["qft-like", "phase"]})
