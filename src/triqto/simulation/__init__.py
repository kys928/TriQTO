"""Simulation helpers with lazy optional Aer-backed exports."""
from __future__ import annotations

from .ideal_shot import simulate_ideal_shots
from .ideal_statevector import simulate_ideal_statevector, statevector_probabilities
from .measurement import MeasurementSetting, default_measurement_context, measurement_setting_for
from .result_normalization import (
    bind_parameter_values,
    copy_without_final_measurements,
    counts_to_probabilities,
    extract_quantum_circuit,
    normalize_counts,
    normalize_probabilities,
    sample_counts_from_probabilities,
    validate_no_unbound_parameters,
)
from .results import IdealShotResult, IdealStatevectorResult
from .sampler_runner import run_ideal_sampler

_LAZY_EXPORTS = {
    "NoiseSpec": (".noisy_shot", "NoiseSpec"),
    "simulate_noisy_aer_shots": (".noisy_shot", "simulate_noisy_aer_shots"),
    "DensityMatrixResult": (".density_matrix", "DensityMatrixResult"),
    "simulate_density_matrix": (".density_matrix", "simulate_density_matrix"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "DensityMatrixResult",
    "IdealShotResult",
    "IdealStatevectorResult",
    "MeasurementSetting",
    "NoiseSpec",
    "bind_parameter_values",
    "copy_without_final_measurements",
    "counts_to_probabilities",
    "default_measurement_context",
    "extract_quantum_circuit",
    "measurement_setting_for",
    "normalize_counts",
    "normalize_probabilities",
    "run_ideal_sampler",
    "sample_counts_from_probabilities",
    "simulate_density_matrix",
    "simulate_ideal_shots",
    "simulate_ideal_statevector",
    "simulate_noisy_aer_shots",
    "statevector_probabilities",
    "validate_no_unbound_parameters",
]
