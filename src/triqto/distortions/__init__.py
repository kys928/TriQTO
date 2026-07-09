"""Deterministic circuit distortion engine for TriQTO Phase 5."""
from .amplitude import apply_rx_overrotation, apply_ry_overrotation
from .base import DistortedCircuit, copy_circuit, extract_circuit, make_distorted_circuit, validate_qubits
from .distortion_registry import apply_distortion, get_distortion, list_distortions
from .entangling import apply_entangling_rzz_drift
from .layout import apply_layout_permutation_marker
from .mixed import apply_mixed_unitary_drift
from .phase import apply_phase_rz_drift
from .readout import apply_readout_bitflip_marker

__all__ = [
    "DistortedCircuit",
    "apply_distortion",
    "apply_entangling_rzz_drift",
    "apply_layout_permutation_marker",
    "apply_mixed_unitary_drift",
    "apply_phase_rz_drift",
    "apply_readout_bitflip_marker",
    "apply_rx_overrotation",
    "apply_ry_overrotation",
    "copy_circuit",
    "extract_circuit",
    "get_distortion",
    "list_distortions",
    "make_distorted_circuit",
    "validate_qubits",
]
