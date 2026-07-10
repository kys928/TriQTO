"""Phase 7 deterministic raw dataset generation specifications."""
from __future__ import annotations

import json, math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.circuits.families import get_circuit_family
from triqto.distortions.distortion_registry import get_distortion


def _copy_jsonable(value: Any) -> Any:
    text = json.dumps(value, sort_keys=True, allow_nan=False)
    return json.loads(text)

@dataclass(frozen=True, slots=True)
class CircuitGenerationSpec:
    family: str
    n_qubits: int
    generator_kwargs: dict[str, Any] = field(default_factory=dict)
    repetitions: int = 1
    def __post_init__(self) -> None:
        get_circuit_family(self.family)
        if self.n_qubits <= 0: raise ValueError("n_qubits must be positive")
        if self.repetitions <= 0: raise ValueError("repetitions must be positive")
        object.__setattr__(self, "generator_kwargs", _copy_jsonable(self.generator_kwargs))

@dataclass(frozen=True, slots=True)
class DistortionSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        get_distortion(self.name)
        object.__setattr__(self, "kwargs", _copy_jsonable(self.kwargs))

@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    dataset_name: str
    base_seed: int
    circuit_specs: list[CircuitGenerationSpec]
    distortion_specs: list[DistortionSpec]
    schema_version: str = "triqto.phase7.v1"
    parameter_low: float = -math.pi
    parameter_high: float = math.pi
    ideal_shots: int | None = None
    store_statevectors: bool = True
    max_samples: int = 1000
    def __post_init__(self) -> None:
        if not self.dataset_name: raise ValueError("dataset_name must be non-empty")
        if not self.circuit_specs: raise ValueError("at least one circuit spec is required")
        if not self.distortion_specs: raise ValueError("at least one distortion spec is required")
        if self.parameter_high < self.parameter_low: raise ValueError("parameter_high must be >= parameter_low")
        if self.ideal_shots is not None and self.ideal_shots <= 0: raise ValueError("ideal_shots must be positive")
        if self.max_samples <= 0: raise ValueError("max_samples must be positive")
        cs = [c if isinstance(c, CircuitGenerationSpec) else CircuitGenerationSpec(**c) for c in self.circuit_specs]
        ds = [d if isinstance(d, DistortionSpec) else DistortionSpec(**d) for d in self.distortion_specs]
        object.__setattr__(self, "circuit_specs", cs); object.__setattr__(self, "distortion_specs", ds)
        if predicted_sample_count(self) > self.max_samples: raise ValueError("predicted sample count exceeds max_samples")
        canonical_json(config_to_dict(self))

def predicted_sample_count(config: DatasetGenerationConfig) -> int:
    return sum(s.repetitions for s in config.circuit_specs) * len(config.distortion_specs)

def config_to_dict(config: DatasetGenerationConfig) -> dict[str, Any]:
    return asdict(config)

def config_from_dict(payload: dict[str, Any]) -> DatasetGenerationConfig:
    allowed = set(DatasetGenerationConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra: raise ValueError(f"Unknown config fields: {sorted(extra)}")
    data = dict(payload)
    data["circuit_specs"] = [CircuitGenerationSpec(**x) for x in data.get("circuit_specs", [])]
    data["distortion_specs"] = [DistortionSpec(**x) for x in data.get("distortion_specs", [])]
    return DatasetGenerationConfig(**data)

def config_id(config: DatasetGenerationConfig) -> str:
    return make_deterministic_id("config", {"dataset_name": config.dataset_name, "schema_version": config.schema_version, "config": config_to_dict(config)})

def save_generation_config(config: DatasetGenerationConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config_to_dict(config), sort_keys=True, indent=2, allow_nan=False) + "\n")

def load_generation_config(path: str | Path) -> DatasetGenerationConfig:
    return config_from_dict(json.loads(Path(path).read_text()))
