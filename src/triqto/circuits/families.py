"""Circuit family registry placeholders for variable-size TriQTO circuits."""
from __future__ import annotations
from collections.abc import Callable

CircuitGenerator = Callable[..., object]
CIRCUIT_FAMILY_REGISTRY: dict[str, CircuitGenerator] = {}

# TODO: register Bell, GHZ, phase interference, QFT-like, hardware-efficient ansatz,
# random shallow, lattice-entangled, and QAOA-like generators in Phase 3.
def register_family(name: str, generator: CircuitGenerator) -> None:
    """Register a future circuit generator by family name."""
    CIRCUIT_FAMILY_REGISTRY[name] = generator
