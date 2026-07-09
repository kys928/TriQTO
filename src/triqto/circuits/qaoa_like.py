"""QAOA-like MaxCut circuits on a line graph."""
from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def make_qaoa_like_circuit(n_qubits: int, layers: int = 1, measure: bool = True) -> GeneratedCircuit:
    if n_qubits < 2: raise ValueError("qaoa_like requires n_qubits >= 2")
    if layers < 1: raise ValueError("qaoa_like requires layers >= 1")
    edges = [(q, q + 1) for q in range(n_qubits - 1)]
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"qaoa_like_{n_qubits}q_l{layers}")
    for q in range(n_qubits): circuit.h(q)
    for layer in range(layers):
        gamma = Parameter(f"gamma_{layer}"); beta = Parameter(f"beta_{layer}")
        for a, b in edges:
            circuit.cx(a, b); circuit.rz(gamma, b); circuit.cx(a, b)
        for q in range(n_qubits): circuit.rx(beta, q)
    if measure: circuit.measure(range(n_qubits), range(n_qubits))
    return make_generated_circuit(circuit, "qaoa_like", "qaoa_like", True, True, {"layers": layers, "graph_edges": [[a, b] for a, b in edges], "tags": ["qaoa-like", "maxcut", "line-graph"]})
