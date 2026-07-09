"""Parameter utilities for TriQTO circuit generators."""
from __future__ import annotations

import math
import random
from qiskit import QuantumCircuit


def parameter_names(circuit: QuantumCircuit) -> list[str]:
    """Return stable, sorted parameter names from a circuit."""
    return sorted(str(parameter.name) for parameter in circuit.parameters)


def encode_angle_sin_cos(theta: float) -> tuple[float, float]:
    """Encode an angle by its sine and cosine."""
    return (math.sin(theta), math.cos(theta))


def sample_angles(
    count: int,
    seed: int | None = None,
    low: float = -math.pi,
    high: float = math.pi,
) -> list[float]:
    """Sample deterministic random angles from a uniform interval."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if high < low:
        raise ValueError("high must be greater than or equal to low")
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(count)]


def bind_random_parameters(circuit: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """Return a copied circuit with all parameters bound to sampled angles."""
    names = parameter_names(circuit)
    values = sample_angles(len(names), seed=seed)
    assignment = {parameter: values[names.index(parameter.name)] for parameter in circuit.parameters}
    return circuit.assign_parameters(assignment, inplace=False)
