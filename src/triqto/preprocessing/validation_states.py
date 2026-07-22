"""Quantum-state and metric validation rules."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .config import PreprocessingConfig
from .records import ValidationFinding
from .validation_core import ValidationCollector, _finite_numeric


def validate_statevector(statevector: Sequence[complex] | np.ndarray, *, n_qubits: int,
                         config: PreprocessingConfig, collector: ValidationCollector,
                         field_path: str) -> np.ndarray | None:
    vector = np.asarray(statevector, dtype=np.complex128)
    if vector.ndim != 1 or vector.size != 1 << n_qubits:
        collector.add("schema.statevector_dimension", "error", field_path, vector.shape,
                      f"statevector length {1 << n_qubits}", "quarantine")
        return None
    if not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
        collector.add("numerical.statevector_finite", "error", field_path, "nonfinite",
                      "all amplitudes finite", "quarantine")
        return None
    norm = float(np.vdot(vector, vector).real)
    deviation = abs(norm - 1.0)
    tol = config.numerical_tolerances
    if deviation > tol.state_norm_repair:
        collector.add("physics.statevector_norm", "error", field_path, norm,
                      "statevector norm equals one", "quarantine")
        return None
    if deviation > tol.state_norm_warning and config.validation.repair_small_numerical_drift and norm > 0.0:
        vector = vector / np.sqrt(norm)
        collector.add("physics.statevector_norm_repair", "warning", field_path, norm,
                      "small norm deviation", "repaired_with_audit", repair_applied=True)
    return vector


def validate_density_matrix(density_matrix: Sequence[Sequence[complex]] | np.ndarray, *,
                            n_qubits: int, config: PreprocessingConfig,
                            collector: ValidationCollector, field_path: str) -> np.ndarray | None:
    matrix = np.asarray(density_matrix, dtype=np.complex128)
    dimension = 1 << n_qubits
    if matrix.shape != (dimension, dimension):
        collector.add("schema.density_matrix_dimension", "error", field_path, matrix.shape,
                      f"density matrix shape {(dimension, dimension)}", "quarantine")
        return None
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        collector.add("numerical.density_matrix_finite", "error", field_path, "nonfinite",
                      "all entries finite", "quarantine")
        return None
    tol = config.numerical_tolerances
    hermitian_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    if hermitian_error > tol.hermiticity_repair:
        collector.add("physics.density_matrix_hermitian", "error", field_path, hermitian_error,
                      "rho == rho_dagger", "quarantine")
        return None
    if hermitian_error > tol.hermiticity_warning and config.validation.repair_small_numerical_drift:
        matrix = 0.5 * (matrix + matrix.conj().T)
        collector.add("physics.density_matrix_hermiticity_repair", "warning", field_path,
                      hermitian_error, "small Hermiticity deviation", "repaired_with_audit",
                      repair_applied=True)
    trace = complex(np.trace(matrix))
    if abs(trace - 1.0) > tol.trace_repair:
        collector.add("physics.density_matrix_trace", "error", field_path, trace,
                      "Tr(rho) == 1", "quarantine")
        return None
    if abs(trace - 1.0) > tol.trace_warning and config.validation.repair_small_numerical_drift:
        matrix = matrix / trace
        collector.add("physics.density_matrix_trace_repair", "warning", field_path, trace,
                      "small trace deviation", "repaired_with_audit", repair_applied=True)
    minimum = float(np.min(np.linalg.eigvalsh(matrix)))
    if minimum < tol.psd_eigenvalue_failure:
        collector.add("physics.density_matrix_psd", "error", field_path, minimum,
                      "rho positive semidefinite", "quarantine")
        return None
    if minimum < tol.psd_eigenvalue_warning:
        collector.add("physics.density_matrix_psd_warning", "warning", field_path, minimum,
                      "tiny negative eigenvalue consistent with numerical error", "pass_with_warning")
    return matrix


def validate_metric_ranges(metrics: Mapping[str, Any], *, collector: ValidationCollector,
                           field_path: str) -> None:
    bounded = {"fidelity", "hellinger", "hellinger_distance", "jensen_shannon_distance",
               "total_variation", "trace_distance", "pure_trace_distance"}
    nonnegative = {"fubini_study", "fisher_rao", "jensen_shannon_divergence", "persistence_entropy"}
    for name, raw in metrics.items():
        if isinstance(name, str) and name.endswith("__nonfinite"):
            continue
        if raw is None:
            continue
        if not _finite_numeric(raw):
            collector.add("numerical.metric_finite", "error", f"{field_path}.{name}", raw,
                          "finite metric value", "quarantine")
            continue
        value = float(raw)
        if name in bounded and not -1e-10 <= value <= 1.0 + 1e-10:
            collector.add("physics.metric_range_01", "error", f"{field_path}.{name}", value,
                          "metric in [0,1]", "quarantine")
        elif name in nonnegative and value < -1e-10:
            collector.add("physics.metric_nonnegative", "error", f"{field_path}.{name}", value,
                          "nonnegative metric", "quarantine")


def quarantine_reason(findings: Sequence[ValidationFinding]) -> str | None:
    failed = [item.rule_id for item in findings if item.disposition == "quarantine"]
    return None if not failed else ";".join(sorted(set(failed)))
