"""Circuit family registry for variable-size TriQTO circuits."""
from __future__ import annotations
from collections.abc import Callable
from .bell import make_bell_circuit
from .ghz import make_ghz_circuit
from .phase_interference import make_phase_interference_circuit
from .qft_like import make_qft_like_circuit
from .hardware_efficient import make_hardware_efficient_ansatz
from .random_shallow import make_random_shallow_circuit
from .lattice_entangled import make_lattice_entangled_circuit
from .qaoa_like import make_qaoa_like_circuit
from .circuit_metadata import GeneratedCircuit

CircuitGenerator = Callable[..., GeneratedCircuit]

CIRCUIT_FAMILY_REGISTRY: dict[str, CircuitGenerator] = {
    "bell": make_bell_circuit,
    "ghz": make_ghz_circuit,
    "phase_interference": make_phase_interference_circuit,
    "qft_like": make_qft_like_circuit,
    "hardware_efficient_ansatz": make_hardware_efficient_ansatz,
    "random_shallow": make_random_shallow_circuit,
    "lattice_entangled": make_lattice_entangled_circuit,
    "qaoa_like": make_qaoa_like_circuit,
}


def list_circuit_families() -> list[str]:
    return sorted(CIRCUIT_FAMILY_REGISTRY)


def get_circuit_family(name: str) -> CircuitGenerator:
    try:
        return CIRCUIT_FAMILY_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_circuit_families())
        raise ValueError(f"Unknown circuit family {name!r}. Available families: {available}") from exc


def generate_circuit_family(name: str, n_qubits: int, **kwargs: object) -> GeneratedCircuit:
    return get_circuit_family(name)(n_qubits=n_qubits, **kwargs)
