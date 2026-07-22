"""Hardware-layout and CPTP channel validation rules."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .config import PreprocessingConfig
from .validation_core import ValidationCollector


def validate_layout_context(
    hardware_context: Mapping[str, Any],
    *,
    n_qubits: int,
    config: PreprocessingConfig,
    collector: ValidationCollector,
    field_path: str,
) -> None:
    if not config.validation.validate_layout:
        return
    layout = hardware_context.get("layout")
    coupling_map = hardware_context.get("coupling_map")
    if layout is None:
        return
    if not isinstance(layout, Mapping):
        collector.add(
            "schema.layout_mapping",
            "error",
            f"{field_path}.layout",
            type(layout).__name__,
            "logical-to-physical mapping",
            "quarantine",
        )
        return
    physical_values: list[int] = []
    for logical, physical in layout.items():
        try:
            logical_index = int(logical)
            physical_index = int(physical)
        except (TypeError, ValueError):
            collector.add(
                "schema.layout_indices",
                "error",
                f"{field_path}.layout",
                layout,
                "integer logical and physical qubit indices",
                "quarantine",
            )
            return
        if logical_index < 0 or logical_index >= n_qubits or physical_index < 0:
            collector.add(
                "physics.layout_index_range",
                "error",
                f"{field_path}.layout",
                {logical_index: physical_index},
                "valid logical and nonnegative physical qubit indices",
                "quarantine",
            )
        physical_values.append(physical_index)
    if len(set(physical_values)) != len(physical_values):
        collector.add(
            "physics.layout_injective",
            "error",
            f"{field_path}.layout",
            layout,
            "injective logical-to-physical qubit layout",
            "quarantine",
        )
    if coupling_map is not None and not isinstance(coupling_map, (list, tuple)):
        collector.add(
            "schema.coupling_map_sequence",
            "error",
            f"{field_path}.coupling_map",
            type(coupling_map).__name__,
            "sequence of physical coupling edges",
            "quarantine",
        )


def validate_cptp_channel(
    *,
    kraus_operators: Sequence[Sequence[Sequence[complex]]] | np.ndarray | None = None,
    choi_matrix: Sequence[Sequence[complex]] | np.ndarray | None = None,
    input_dimension: int | None = None,
    config: PreprocessingConfig,
    collector: ValidationCollector,
    field_path: str,
) -> bool:
    """Validate complete positivity and trace preservation when a full channel is supplied.

    Kraus validation is preferred because it avoids ambiguity about Choi index
    conventions.  Choi matrices use column-stacking convention and require an
    explicit input dimension.
    """
    if not config.validation.validate_cptp:
        return True
    if (kraus_operators is None) == (choi_matrix is None):
        collector.add(
            "schema.channel_representation",
            "error",
            field_path,
            "both or neither channel representations supplied",
            "exactly one of Kraus operators or Choi matrix",
            "quarantine",
        )
        return False
    tolerance = config.numerical_tolerances
    if kraus_operators is not None:
        operators = np.asarray(kraus_operators, dtype=np.complex128)
        if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
            collector.add(
                "schema.kraus_dimension",
                "error",
                field_path,
                operators.shape,
                "Kraus array of shape (k, d, d)",
                "quarantine",
            )
            return False
        if not np.isfinite(operators.real).all() or not np.isfinite(operators.imag).all():
            collector.add(
                "numerical.channel_finite",
                "error",
                field_path,
                "contains nonfinite values",
                "finite Kraus entries",
                "quarantine",
            )
            return False
        dimension = operators.shape[1]
        trace_preserving = np.zeros((dimension, dimension), dtype=np.complex128)
        choi = np.zeros((dimension * dimension, dimension * dimension), dtype=np.complex128)
        for operator in operators:
            trace_preserving += operator.conj().T @ operator
            vectorized = operator.reshape(-1, order="F")
            choi += np.outer(vectorized, vectorized.conj())
    else:
        choi = np.asarray(choi_matrix, dtype=np.complex128)
        if choi.ndim != 2 or choi.shape[0] != choi.shape[1]:
            collector.add(
                "schema.choi_dimension",
                "error",
                field_path,
                choi.shape,
                "square Choi matrix",
                "quarantine",
            )
            return False
        if input_dimension is None or input_dimension <= 0:
            collector.add(
                "schema.choi_input_dimension",
                "error",
                field_path,
                input_dimension,
                "positive input dimension for Choi validation",
                "quarantine",
            )
            return False
        dimension = int(input_dimension)
        if choi.shape != (dimension * dimension, dimension * dimension):
            collector.add(
                "schema.choi_dimension",
                "error",
                field_path,
                choi.shape,
                f"Choi shape {(dimension * dimension, dimension * dimension)}",
                "quarantine",
            )
            return False
        # J_{(out,in),(out',in')} with column-vectorization convention.
        tensor = choi.reshape(dimension, dimension, dimension, dimension, order="F")
        trace_preserving = np.einsum("aiaj->ij", tensor)
    hermitian_error = float(np.max(np.abs(choi - choi.conj().T)))
    if hermitian_error > tolerance.hermiticity_repair:
        collector.add(
            "physics.channel_choi_hermitian",
            "error",
            field_path,
            hermitian_error,
            "Choi matrix Hermitian within tolerance",
            "quarantine",
        )
        return False
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T))))
    if minimum_eigenvalue < tolerance.psd_eigenvalue_failure:
        collector.add(
            "physics.channel_complete_positivity",
            "error",
            field_path,
            minimum_eigenvalue,
            "Choi matrix positive semidefinite",
            "quarantine",
        )
        return False
    tp_error = float(np.max(np.abs(trace_preserving - np.eye(dimension))))
    if tp_error > tolerance.trace_repair:
        collector.add(
            "physics.channel_trace_preservation",
            "error",
            field_path,
            tp_error,
            "sum(K^dagger K)=I or partial trace of Choi equals I",
            "quarantine",
        )
        return False
    if minimum_eigenvalue < tolerance.psd_eigenvalue_warning or tp_error > tolerance.trace_warning:
        collector.add(
            "physics.channel_numerical_warning",
            "warning",
            field_path,
            {"minimum_choi_eigenvalue": minimum_eigenvalue, "tp_error": tp_error},
            "CPTP within strict numerical tolerance",
            "pass_with_warning",
        )
    return True
