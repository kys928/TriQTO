"""Credential-gated hardware job/result contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from triqto.core.ids import make_deterministic_id

FORBIDDEN_PHYSICAL_FIELDS = {"statevector", "density_matrix", "exact_probabilities", "clean_target_metrics", "ideal_hilbert"}


@dataclass(frozen=True, slots=True)
class HardwareJobSpec:
    backend_name: str
    circuit_id: str
    shots: int
    measurement_bases: tuple[str, ...]
    backend_id: str
    confirmation_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend_name or not self.circuit_id or not self.backend_id:
            raise ValueError("backend_name, circuit_id, and backend_id must be nonblank")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if not self.measurement_bases or any(b not in {"X", "Y", "Z"} for b in self.measurement_bases):
            raise ValueError("measurement_bases must contain only X/Y/Z")
        forbidden = FORBIDDEN_PHYSICAL_FIELDS & set(self.metadata)
        if forbidden:
            raise ValueError(f"physical hardware spec contains forbidden simulator fields: {sorted(forbidden)}")

    @property
    def job_spec_id(self) -> str:
        return make_deterministic_id("hwjob", {"schema": "triqto.hardware.job.v1", "backend_name": self.backend_name, "circuit_id": self.circuit_id, "shots": self.shots, "measurement_bases": list(self.measurement_bases), "backend_id": self.backend_id, "metadata": self.metadata})


@dataclass(frozen=True, slots=True)
class HardwareResultRecord:
    job_spec_id: str
    backend_id: str
    backend_name: str
    job_id: str
    shots_requested: int
    shots_realized: int
    counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_spec_id or not self.backend_id or not self.backend_name or not self.job_id:
            raise ValueError("hardware result identifiers must be nonblank")
        if self.shots_requested <= 0 or self.shots_realized <= 0:
            raise ValueError("shot counts must be positive")
        if sum(int(v) for v in self.counts.values()) != self.shots_realized:
            raise ValueError("counts must sum to realized shots")
        forbidden = FORBIDDEN_PHYSICAL_FIELDS & set(self.metadata)
        if forbidden:
            raise ValueError(f"physical hardware result contains forbidden simulator fields: {sorted(forbidden)}")

    @property
    def result_id(self) -> str:
        return make_deterministic_id("hwresult", {"schema": "triqto.hardware.result.v1", "job_spec_id": self.job_spec_id, "backend_id": self.backend_id, "job_id": self.job_id, "shots_realized": self.shots_realized, "counts": self.counts})


def describe_contract() -> str:
    return "Typed hardware specs/results reject simulator-only fields and require explicit physical counts."


__all__ = ["FORBIDDEN_PHYSICAL_FIELDS", "HardwareJobSpec", "HardwareResultRecord"]
