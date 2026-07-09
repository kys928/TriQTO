"""TriQTO circuit family generators and metadata helpers."""
from .circuit_metadata import GeneratedCircuit, count_two_qubit_gates, has_measurements, make_generated_circuit, summarize_circuit
from .families import generate_circuit_family, get_circuit_family, list_circuit_families
from .bell import make_bell_circuit
from .ghz import make_ghz_circuit
from .phase_interference import make_phase_interference_circuit
from .qft_like import make_qft_like_circuit
from .hardware_efficient import make_hardware_efficient_ansatz
from .random_shallow import make_random_shallow_circuit
from .lattice_entangled import make_lattice_entangled_circuit
from .qaoa_like import make_qaoa_like_circuit

__all__ = [
    "GeneratedCircuit", "count_two_qubit_gates", "has_measurements", "make_generated_circuit", "summarize_circuit",
    "generate_circuit_family", "get_circuit_family", "list_circuit_families", "make_bell_circuit", "make_ghz_circuit",
    "make_phase_interference_circuit", "make_qft_like_circuit", "make_hardware_efficient_ansatz", "make_random_shallow_circuit",
    "make_lattice_entangled_circuit", "make_qaoa_like_circuit",
]
