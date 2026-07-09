"""Deterministic shallow random circuit family."""
from __future__ import annotations
import math, random
from qiskit import QuantumCircuit
from .circuit_metadata import GeneratedCircuit, make_generated_circuit


def make_random_shallow_circuit(n_qubits: int, depth: int = 3, seed: int | None = None, measure: bool = True) -> GeneratedCircuit:
    if n_qubits < 1: raise ValueError("random_shallow requires n_qubits >= 1")
    if depth < 1: raise ValueError("random_shallow requires depth >= 1")
    rng = random.Random(seed)
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0, name=f"random_shallow_{n_qubits}q_d{depth}")
    one_qubit = ["h", "x", "rz", "rx", "ry"]
    gate_log: list[dict[str, object]] = []
    for layer in range(depth):
        for q in range(n_qubits):
            gate = rng.choice(one_qubit + (["cx"] if n_qubits > 1 and q < n_qubits - 1 else []))
            if gate == "h": circuit.h(q); gate_log.append({"layer": layer, "gate": gate, "qubits": [q]})
            elif gate == "x": circuit.x(q); gate_log.append({"layer": layer, "gate": gate, "qubits": [q]})
            elif gate in {"rz", "rx", "ry"}:
                angle = rng.uniform(-math.pi, math.pi); getattr(circuit, gate)(angle, q); gate_log.append({"layer": layer, "gate": gate, "qubits": [q], "angle": angle})
            else:
                circuit.cx(q, q + 1); gate_log.append({"layer": layer, "gate": gate, "qubits": [q, q + 1]})
    if measure: circuit.measure(range(n_qubits), range(n_qubits))
    return make_generated_circuit(circuit, "random_shallow", "random_shallow", True, n_qubits > 1, {"seed": seed, "requested_depth": depth, "random_gate_log": gate_log, "tags": ["random", "shallow"]})
