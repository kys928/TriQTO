"""Registry for deterministic Phase 5 TriQTO distortions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .amplitude import apply_rx_overrotation, apply_ry_overrotation
from .base import DistortedCircuit
from .entangling import apply_entangling_rzz_drift
from .layout import apply_layout_permutation_marker
from .mixed import apply_mixed_unitary_drift
from .phase import apply_phase_rz_drift
from .readout import apply_readout_bitflip

DistortionFunction = Callable[..., DistortedCircuit]

DISTORTION_REGISTRY: dict[str, DistortionFunction] = {
    "phase_rz_drift": apply_phase_rz_drift,
    "rx_overrotation": apply_rx_overrotation,
    "ry_overrotation": apply_ry_overrotation,
    "entangling_rzz_drift": apply_entangling_rzz_drift,
    "readout_bitflip": apply_readout_bitflip,
    "layout_permutation_marker": apply_layout_permutation_marker,
    "mixed_unitary_drift": apply_mixed_unitary_drift,
}


def list_distortions() -> list[str]:
    """Return the sorted list of available distortion names."""
    return sorted(DISTORTION_REGISTRY)


def get_distortion(name: str) -> DistortionFunction:
    """Return a registered distortion by name, or raise a helpful error."""
    try:
        return DISTORTION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_distortions())
        raise ValueError(f"Unknown distortion {name!r}. Available distortions: {available}") from exc


def apply_distortion(name: str, circuit_or_generated: Any, **kwargs: Any) -> DistortedCircuit:
    """Apply a registered distortion by name."""
    return get_distortion(name)(circuit_or_generated, **kwargs)
