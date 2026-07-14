"""Strict configuration for the offline Phase 15.5 empirical workflow."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from triqto.simulation import NoiseSpec

PHASE155_CONFIG_SCHEMA = "triqto.phase15_5.config.v1"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class NoiseProfileConfig:
    name: str
    channels: tuple[dict[str, Any], ...]
    shots: int = 256

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("noise profile name must be nonblank")
        if isinstance(self.channels, (str, bytes)) or not isinstance(self.channels, Sequence):
            raise TypeError("noise profile channels must be a sequence")
        copied = tuple(json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False)) for value in self.channels)
        NoiseSpec(copied)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "channels", copied)
        object.__setattr__(self, "shots", _positive_int(self.shots, "noise profile shots"))


@dataclass(frozen=True, slots=True)
class Phase155Config:
    schema_version: str = PHASE155_CONFIG_SCHEMA
    seed: int = 2026
    noise_profiles: tuple[NoiseProfileConfig, ...] = (
        NoiseProfileConfig(
            name="depolarizing_readout_smoke",
            channels=(
                {"type": "depolarizing", "probability": 0.01, "qubits": 1, "gates": ["x", "sx", "h", "rx", "ry", "rz"]},
                {"type": "readout_error", "probability": 0.02},
            ),
            shots=128,
        ),
    )
    measurement_bases: tuple[str, ...] = ("Z", "X", "Y")
    max_samples_per_split: int = 2
    include_density_matrix: bool = True
    layout_seeds: tuple[int, ...] = (2026, 2027)
    routing_optimization_levels: tuple[int, ...] = (0, 1, 3)
    depth_optimization_levels: tuple[int, ...] = (1, 3)
    hidden_dim: int = 32
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    utility_mse_weight: float = 0.25
    probe_cost: float = 0.02
    bootstrap_replicates: int = 200
    confidence_level: float = 0.95
    physical_hardware: bool = False
    topology_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != PHASE155_CONFIG_SCHEMA:
            raise ValueError("unsupported Phase 15.5 config schema")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        profiles = tuple(
            value if isinstance(value, NoiseProfileConfig) else NoiseProfileConfig(**dict(value))
            for value in self.noise_profiles
        )
        if not profiles or len({value.name for value in profiles}) != len(profiles):
            raise ValueError("noise profiles must be nonempty with unique names")
        bases = tuple(str(value).upper() for value in self.measurement_bases)
        if not bases or len(set(bases)) != len(bases) or any(value not in {"X", "Y", "Z"} for value in bases):
            raise ValueError("measurement_bases must be unique X/Y/Z values")
        layout_seeds = tuple(int(value) for value in self.layout_seeds)
        if not layout_seeds or len(set(layout_seeds)) != len(layout_seeds) or any(value < 0 for value in layout_seeds):
            raise ValueError("layout_seeds must be unique nonnegative integers")
        routing = tuple(int(value) for value in self.routing_optimization_levels)
        depth = tuple(int(value) for value in self.depth_optimization_levels)
        if not routing or len(set(routing)) != len(routing) or any(value not in {0, 1, 2, 3} for value in routing):
            raise ValueError("routing_optimization_levels must be unique values in 0..3")
        if not depth or len(set(depth)) != len(depth) or any(value not in {0, 1, 2, 3} for value in depth):
            raise ValueError("depth_optimization_levels must be unique values in 0..3")
        hidden_dim = _positive_int(self.hidden_dim, "hidden_dim")
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even")
        confidence = _finite_float(self.confidence_level, "confidence_level", positive=True)
        if confidence >= 1.0:
            raise ValueError("confidence_level must be below one")
        if self.physical_hardware is not False:
            raise ValueError("Phase 15.5 offline config cannot enable physical hardware")
        if float(self.topology_loss_weight) != 0.0:
            raise ValueError("topology_loss_weight must remain exactly 0.0")
        object.__setattr__(self, "noise_profiles", profiles)
        object.__setattr__(self, "measurement_bases", bases)
        object.__setattr__(self, "max_samples_per_split", _positive_int(self.max_samples_per_split, "max_samples_per_split"))
        if not isinstance(self.include_density_matrix, bool):
            raise TypeError("include_density_matrix must be bool")
        object.__setattr__(self, "layout_seeds", layout_seeds)
        object.__setattr__(self, "routing_optimization_levels", routing)
        object.__setattr__(self, "depth_optimization_levels", depth)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "epochs", _positive_int(self.epochs, "epochs"))
        object.__setattr__(self, "learning_rate", _finite_float(self.learning_rate, "learning_rate", positive=True))
        object.__setattr__(self, "weight_decay", _finite_float(self.weight_decay, "weight_decay", nonnegative=True))
        object.__setattr__(self, "utility_mse_weight", _finite_float(self.utility_mse_weight, "utility_mse_weight", nonnegative=True))
        object.__setattr__(self, "probe_cost", _finite_float(self.probe_cost, "probe_cost", nonnegative=True))
        object.__setattr__(self, "bootstrap_replicates", _positive_int(self.bootstrap_replicates, "bootstrap_replicates"))
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "topology_loss_weight", 0.0)
        json.dumps(phase155_config_to_dict(self), sort_keys=True, allow_nan=False)


def phase155_config_to_dict(config: Phase155Config) -> dict[str, Any]:
    payload = asdict(config)
    payload["measurement_bases"] = list(config.measurement_bases)
    payload["layout_seeds"] = list(config.layout_seeds)
    payload["routing_optimization_levels"] = list(config.routing_optimization_levels)
    payload["depth_optimization_levels"] = list(config.depth_optimization_levels)
    return payload


def phase155_config_from_dict(payload: Mapping[str, Any]) -> Phase155Config:
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 15.5 config must be a mapping")
    allowed = set(Phase155Config.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"unknown Phase 15.5 config fields: {sorted(extra)}")
    data = dict(payload)
    if "noise_profiles" in data:
        data["noise_profiles"] = tuple(
            value if isinstance(value, NoiseProfileConfig) else NoiseProfileConfig(**dict(value))
            for value in data["noise_profiles"]
        )
    for name in ("measurement_bases", "layout_seeds", "routing_optimization_levels", "depth_optimization_levels"):
        if name in data:
            data[name] = tuple(data[name])
    return Phase155Config(**data)


def load_phase155_config(path: str | Path) -> Phase155Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return phase155_config_from_dict(payload)


def save_phase155_config(config: Phase155Config, path: str | Path) -> None:
    Path(path).write_text(json.dumps(phase155_config_to_dict(config), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


__all__ = [
    "NoiseProfileConfig",
    "PHASE155_CONFIG_SCHEMA",
    "Phase155Config",
    "load_phase155_config",
    "phase155_config_from_dict",
    "phase155_config_to_dict",
    "save_phase155_config",
]
