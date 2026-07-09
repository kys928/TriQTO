"""Lattice-entangled parameterized circuit family."""
from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def _edges(n_qubits: int, layout: str) -> list[tuple[int, int]]:
    if layout == "line": return [(q, q + 1) for q in range(n_qubits - 1)]
    if layout == "ring": return [(q, q + 1) for q in range(n_qubits - 1)] + [(n_qubits - 1, 0)]
    raise ValueError("layout must be 'line' or 'ring'")


def make_lattice_entangled_circuit(n_qubits: int, layout: str = "line", layers: int = 1, measure: bool = True) -> GeneratedCircuit:
    if n_qubits < 2: raise ValueError("lattice_entangled requires n_qubits >= 2")
    if layers < 1: raise ValueError("lattice_entangled requires layers >= 1")
    edges = _edges(n_qubits, layout)
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"lattice_{layout}_{n_qubits}q_l{layers}")
    for layer in range(layers):
        for q in range(n_qubits): circuit.ry(Parameter(f"theta_{layer}_{q}"), q)
        for a, b in edges: circuit.cz(a, b)
        for q in range(n_qubits): circuit.rz(Parameter(f"phi_{layer}_{q}"), q)
    if measure: circuit.measure(range(n_qubits), range(n_qubits))
    return make_generated_circuit(circuit, "lattice_entangled", "lattice_entanglement", True, True, {"layout": layout, "layers": layers, "lattice_edges": [[a, b] for a, b in edges], "tags": ["lattice", "entangled"]})
