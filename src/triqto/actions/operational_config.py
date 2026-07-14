"""Strict CPU-smoke configuration for offline operational actions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import math
import yaml

OPERATIONAL_SMOKE_CONFIG_SCHEMA = "triqto.operational_smoke_config.v1"


@dataclass(frozen=True, slots=True)
class OperationalActionSmokeConfig:
    schema_version: str = OPERATIONAL_SMOKE_CONFIG_SCHEMA
    probe_bases: tuple[str, ...] = ("X", "Y")
    probe_shots: int = 128
    seed: int = 2026
    backend_n_qubits: int = 2
    backend_name: str = "triqto_local_line_fake_operational_smoke"
    transpilation_optimization_level: int = 1
    semantic_tolerance: float = 1e-10
    evidence_tier: str = "mixed_offline"
    physical_hardware: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != OPERATIONAL_SMOKE_CONFIG_SCHEMA:
            raise ValueError("unsupported operational smoke config schema")
        bases = tuple(str(value).upper() for value in self.probe_bases)
        if not bases or any(value not in {"X", "Y", "Z"} for value in bases):
            raise ValueError("probe_bases must contain X/Y/Z values")
        for name, value in (("probe_shots", self.probe_shots), ("seed", self.seed), ("backend_n_qubits", self.backend_n_qubits)):
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name != "seed" else 0):
                raise ValueError(f"{name} has invalid integer value")
        if len(bases) != self.backend_n_qubits:
            raise ValueError("probe_bases length must equal backend_n_qubits")
        if not isinstance(self.backend_name, str) or not self.backend_name.strip():
            raise ValueError("backend_name must be nonblank")
        if self.transpilation_optimization_level not in {0, 1, 2, 3}:
            raise ValueError("transpilation_optimization_level must be 0, 1, 2, or 3")
        if isinstance(self.semantic_tolerance, bool) or not isinstance(self.semantic_tolerance, (int, float)) or not math.isfinite(float(self.semantic_tolerance)) or float(self.semantic_tolerance) < 0.0:
            raise ValueError("semantic_tolerance must be finite and nonnegative")
        if self.evidence_tier != "mixed_offline":
            raise ValueError("the operational smoke config is offline mixed evidence only")
        if self.physical_hardware is not False:
            raise ValueError("operational smoke config cannot enable physical hardware")
        object.__setattr__(self, "probe_bases", bases)
        object.__setattr__(self, "backend_name", self.backend_name.strip())
        object.__setattr__(self, "semantic_tolerance", float(self.semantic_tolerance))


def operational_action_smoke_config_to_dict(config: OperationalActionSmokeConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["probe_bases"] = list(config.probe_bases)
    return payload


def load_operational_action_smoke_config(path: str | Path) -> OperationalActionSmokeConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("operational smoke config must contain a mapping")
    allowed = set(OperationalActionSmokeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    if set(payload) - allowed:
        raise ValueError(f"unknown operational smoke config fields: {sorted(set(payload) - allowed)}")
    data = dict(payload)
    if "probe_bases" in data:
        data["probe_bases"] = tuple(data["probe_bases"])
    return OperationalActionSmokeConfig(**data)


__all__ = ["OPERATIONAL_SMOKE_CONFIG_SCHEMA", "OperationalActionSmokeConfig", "load_operational_action_smoke_config", "operational_action_smoke_config_to_dict"]
