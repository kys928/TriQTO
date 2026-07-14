"""Basis-conditioned ideal measurement evidence.

This module makes the paper's measurement setting ``M`` explicit.  A setting is
a Pauli-product basis over all active qubits.  Exact probabilities are simulator
privilege; the setting and any observable readout channel are immutable
provenance attached to the result.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from qiskit.quantum_info import Statevector

from triqto.core.ids import make_deterministic_id

from .ideal_statevector import statevector_probabilities
from .result_normalization import (
    bind_parameter_values,
    copy_without_final_measurements,
    extract_quantum_circuit,
    sample_counts_from_probabilities,
)

MEASUREMENT_SCHEMA_VERSION = "triqto.measurement.pauli_product.v1"
PAULI_MEASUREMENT_BASES = ("Z", "X", "Y")


@dataclass(frozen=True, slots=True)
class MeasurementSetting:
    """Validated Pauli-product measurement setting for a fixed qubit width."""

    setting_id: str
    bases: tuple[str, ...]
    schema_version: str = MEASUREMENT_SCHEMA_VERSION

    @property
    def n_qubits(self) -> int:
        return len(self.bases)

    @property
    def label(self) -> str:
        if self.bases and len(set(self.bases)) == 1:
            return self.bases[0]
        return "".join(self.bases)


@dataclass(frozen=True, slots=True)
class MeasurementProbabilityResult:
    """Exact simulator-derived probabilities conditioned on one setting."""

    simulation_mode: str
    n_qubits: int
    measurement_setting: MeasurementSetting
    probabilities: dict[str, float]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MeasurementShotResult:
    """Finite-shot sample from one basis-conditioned exact distribution."""

    simulation_mode: str
    n_qubits: int
    measurement_setting: MeasurementSetting
    shots: int
    counts: dict[str, int]
    source_probabilities: dict[str, float]
    metadata: dict[str, Any]


def _normalized_basis_values(
    value: str | Sequence[str],
    n_qubits: int,
) -> tuple[str, ...]:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
        raise TypeError("n_qubits must be an integer and not bool")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if isinstance(value, str):
        text = value.strip().upper()
        if len(text) == 1:
            bases = (text,) * n_qubits
        else:
            bases = tuple(text)
    elif isinstance(value, Sequence):
        bases = tuple(value)
    else:
        raise TypeError("measurement setting must be a string or sequence of strings")
    if len(bases) != n_qubits:
        raise ValueError(
            f"measurement setting must contain exactly {n_qubits} basis entries"
        )
    normalized: list[str] = []
    for index, basis in enumerate(bases):
        if not isinstance(basis, str):
            raise TypeError(f"measurement basis {index} must be a string")
        item = basis.strip().upper()
        if item not in PAULI_MEASUREMENT_BASES:
            raise ValueError(
                f"measurement basis {index} must be one of {PAULI_MEASUREMENT_BASES}"
            )
        normalized.append(item)
    return tuple(normalized)


def measurement_setting(
    value: str | Sequence[str] | MeasurementSetting,
    n_qubits: int,
) -> MeasurementSetting:
    """Return a canonical setting with a content-derived identifier."""
    if isinstance(value, MeasurementSetting):
        if value.n_qubits != n_qubits:
            raise ValueError("MeasurementSetting width does not match n_qubits")
        expected = make_deterministic_id(
            "measurement_setting",
            {"schema_version": value.schema_version, "bases": list(value.bases)},
        )
        if value.schema_version != MEASUREMENT_SCHEMA_VERSION or value.setting_id != expected:
            raise ValueError("MeasurementSetting has an invalid schema version or setting_id")
        return value
    bases = _normalized_basis_values(value, n_qubits)
    setting_id = make_deterministic_id(
        "measurement_setting",
        {"schema_version": MEASUREMENT_SCHEMA_VERSION, "bases": list(bases)},
    )
    return MeasurementSetting(setting_id=setting_id, bases=bases)


def basis_codes(setting: MeasurementSetting) -> tuple[int, ...]:
    """Encode Z/X/Y as stable integer codes 0/1/2."""
    vocabulary = {name: index for index, name in enumerate(PAULI_MEASUREMENT_BASES)}
    return tuple(vocabulary[basis] for basis in setting.bases)


def _validate_readout_channel(
    readout_bitflip_probability: float | None,
    readout_qubits: Sequence[int] | None,
    n_qubits: int,
) -> tuple[float | None, tuple[int, ...]]:
    if readout_bitflip_probability is None:
        if readout_qubits is not None:
            raise ValueError("readout_qubits require readout_bitflip_probability")
        return None, ()
    if isinstance(readout_bitflip_probability, bool) or not isinstance(
        readout_bitflip_probability, (int, float)
    ):
        raise TypeError("readout_bitflip_probability must be numeric and not bool")
    probability = float(readout_bitflip_probability)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("readout_bitflip_probability must be finite and in [0, 1]")
    selected = tuple(range(n_qubits)) if readout_qubits is None else tuple(readout_qubits)
    if len(set(selected)) != len(selected):
        raise ValueError("readout_qubits must not contain duplicates")
    for qubit in selected:
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError("readout_qubits entries must be integers and not bool")
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(f"readout qubit {qubit} is out of range")
    return probability, selected


def apply_independent_readout_bitflips(
    probabilities: Mapping[str, float],
    *,
    n_qubits: int,
    probability: float,
    qubits: Sequence[int],
) -> dict[str, float]:
    """Apply an exact independent symmetric classical readout channel."""
    checked_probability, selected = _validate_readout_channel(
        probability,
        qubits,
        n_qubits,
    )
    assert checked_probability is not None
    output: dict[str, float] = {}
    for bitstring, raw_mass in probabilities.items():
        mass = float(raw_mass)
        branches = {str(bitstring): mass}
        for qubit in selected:
            position = n_qubits - 1 - qubit
            updated: dict[str, float] = {}
            for current, current_mass in branches.items():
                unchanged = current_mass * (1.0 - checked_probability)
                flipped_bits = list(current)
                flipped_bits[position] = "1" if current[position] == "0" else "0"
                flipped = "".join(flipped_bits)
                updated[current] = updated.get(current, 0.0) + unchanged
                updated[flipped] = updated.get(flipped, 0.0) + (
                    current_mass * checked_probability
                )
            branches = updated
        for outcome, outcome_mass in branches.items():
            output[outcome] = output.get(outcome, 0.0) + outcome_mass
    total = math.fsum(output.values())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("readout channel produced an invalid probability mass")
    return {
        outcome: value / total
        for outcome, value in sorted(output.items())
        if value > 0.0
    }


def simulate_measurement_probabilities(
    circuit_or_generated: Any,
    setting: str | Sequence[str] | MeasurementSetting,
    *,
    parameter_values: Mapping[str, float] | Mapping[Any, float] | None = None,
    readout_bitflip_probability: float | None = None,
    readout_qubits: Sequence[int] | None = None,
) -> MeasurementProbabilityResult:
    """Compute exact ``p(y | M)`` with optional observable readout confusion."""
    original = extract_quantum_circuit(circuit_or_generated)
    bound = bind_parameter_values(original, parameter_values)
    prepared = copy_without_final_measurements(bound)
    resolved = measurement_setting(setting, prepared.num_qubits)
    rotated = prepared.copy()
    for qubit, basis in enumerate(resolved.bases):
        if basis == "X":
            rotated.h(qubit)
        elif basis == "Y":
            rotated.sdg(qubit)
            rotated.h(qubit)
    state = Statevector.from_instruction(rotated)
    probabilities = statevector_probabilities(state, prepared.num_qubits)
    readout_probability, selected = _validate_readout_channel(
        readout_bitflip_probability,
        readout_qubits,
        prepared.num_qubits,
    )
    channel_metadata: dict[str, Any] | None = None
    if readout_probability is not None:
        probabilities = apply_independent_readout_bitflips(
            probabilities,
            n_qubits=prepared.num_qubits,
            probability=readout_probability,
            qubits=selected,
        )
        channel_metadata = {
            "channel": "independent_symmetric_readout_bitflip",
            "probability": readout_probability,
            "qubits": list(selected),
        }
    return MeasurementProbabilityResult(
        simulation_mode="ideal_measurement_probabilities",
        n_qubits=prepared.num_qubits,
        measurement_setting=resolved,
        probabilities=probabilities,
        metadata={
            "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
            "measurement_setting_id": resolved.setting_id,
            "measurement_bases": list(resolved.bases),
            "simulator_privileged_exact_probabilities": True,
            "readout_channel": channel_metadata,
        },
    )


def sample_measurement_counts(
    source: MeasurementProbabilityResult,
    *,
    shots: int,
    seed: int | None,
) -> MeasurementShotResult:
    """Sample finite counts from an explicit basis-conditioned source result."""
    if not isinstance(source, MeasurementProbabilityResult):
        raise TypeError("source must be MeasurementProbabilityResult")
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError("shots must be an integer and not bool")
    if shots <= 0:
        raise ValueError("shots must be positive")
    counts = sample_counts_from_probabilities(source.probabilities, shots, seed)
    return MeasurementShotResult(
        simulation_mode="ideal_measurement_shot",
        n_qubits=source.n_qubits,
        measurement_setting=source.measurement_setting,
        shots=shots,
        counts=counts,
        source_probabilities=dict(source.probabilities),
        metadata={
            "seed": seed,
            "shots": shots,
            "source_simulation_mode": source.simulation_mode,
            "measurement_setting_id": source.measurement_setting.setting_id,
        },
    )


__all__ = [
    "MEASUREMENT_SCHEMA_VERSION",
    "PAULI_MEASUREMENT_BASES",
    "MeasurementProbabilityResult",
    "MeasurementSetting",
    "MeasurementShotResult",
    "apply_independent_readout_bitflips",
    "basis_codes",
    "measurement_setting",
    "sample_measurement_counts",
    "simulate_measurement_probabilities",
]
