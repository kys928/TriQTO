"""Measurement-setting context for basis-conditioned TriQTO evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

BASIS_CODES = {"Z": 0, "X": 1, "Y": 2}
CODE_TO_BASIS = {value: key for key, value in BASIS_CODES.items()}


@dataclass(frozen=True, slots=True)
class MeasurementSetting:
    """Per-qubit measurement basis setting M."""

    bases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.bases:
            raise ValueError("measurement setting must include at least one qubit basis")
        normalized = tuple(str(b).upper() for b in self.bases)
        unknown = sorted(set(normalized) - set(BASIS_CODES))
        if unknown:
            raise ValueError(f"unsupported measurement bases: {unknown}")
        object.__setattr__(self, "bases", normalized)

    @property
    def n_qubits(self) -> int:
        return len(self.bases)

    @property
    def codes(self) -> tuple[int, ...]:
        return tuple(BASIS_CODES[basis] for basis in self.bases)

    @property
    def setting_id_payload(self) -> dict[str, object]:
        return {"measurement_bases": list(self.bases), "basis_code_version": "triqto.measurement_basis.v1"}

    def to_metadata(self) -> dict[str, object]:
        return {**self.setting_id_payload, "basis_codes": list(self.codes)}


def measurement_setting_for(n_qubits: int, bases: str | Iterable[str] | None = None) -> MeasurementSetting:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if bases is None:
        return MeasurementSetting(tuple("Z" for _ in range(n_qubits)))
    if isinstance(bases, str):
        text = bases.upper()
        if len(text) == 1:
            return MeasurementSetting(tuple(text for _ in range(n_qubits)))
        if len(text) != n_qubits:
            raise ValueError("measurement basis string must be length 1 or n_qubits")
        return MeasurementSetting(tuple(text))
    values = tuple(str(value).upper() for value in bases)
    if len(values) != n_qubits:
        raise ValueError("measurement basis iterable length must equal n_qubits")
    return MeasurementSetting(values)


def default_measurement_context(n_qubits: int) -> MeasurementSetting:
    return measurement_setting_for(n_qubits, "Z")
