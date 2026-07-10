"""Phase 7 deterministic raw dataset generation specifications."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from triqto.circuits.families import get_circuit_family
from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.distortions.distortion_registry import get_distortion

PHASE7_METRIC_SCHEMA_VERSION = "triqto.born.phase6"
DEFAULT_BORN_ZERO_ATOL = 1e-12


def _require_nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must contain non-whitespace text")
    return stripped


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number and not bool/string/null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _copy_jsonable_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping/dict")
    text = json.dumps(dict(value), sort_keys=True, allow_nan=False)
    return json.loads(text)


@dataclass(frozen=True, slots=True)
class CircuitGenerationSpec:
    """Specification for one circuit-family generation plan."""

    family: str
    n_qubits: int
    generator_kwargs: dict[str, Any] = field(default_factory=dict)
    repetitions: int = 1

    def __post_init__(self) -> None:
        family = _require_nonblank(self.family, "family")
        n_qubits = _require_int(self.n_qubits, "n_qubits")
        repetitions = _require_int(self.repetitions, "repetitions")
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        get_circuit_family(family)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "generator_kwargs", _copy_jsonable_mapping(self.generator_kwargs, "generator_kwargs"))


@dataclass(frozen=True, slots=True)
class DistortionSpec:
    """Specification for one controlled Phase 7 distortion."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _require_nonblank(self.name, "name")
        get_distortion(name)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kwargs", _copy_jsonable_mapping(self.kwargs, "kwargs"))


@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    """Operational configuration for generating a tiny deterministic raw dataset."""

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
    born_zero_atol: float = DEFAULT_BORN_ZERO_ATOL

    def __post_init__(self) -> None:
        dataset_name = _require_nonblank(self.dataset_name, "dataset_name")
        schema_version = _require_nonblank(self.schema_version, "schema_version")
        base_seed = _require_int(self.base_seed, "base_seed")
        parameter_low = _require_finite_float(self.parameter_low, "parameter_low")
        parameter_high = _require_finite_float(self.parameter_high, "parameter_high")
        born_zero_atol = _require_finite_float(self.born_zero_atol, "born_zero_atol")
        if born_zero_atol < 0:
            raise ValueError("born_zero_atol must be nonnegative")
        if parameter_high < parameter_low:
            raise ValueError("parameter_high must be >= parameter_low")
        if self.ideal_shots is not None:
            ideal_shots = _require_int(self.ideal_shots, "ideal_shots")
            if ideal_shots <= 0:
                raise ValueError("ideal_shots must be positive")
        else:
            ideal_shots = None
        if not isinstance(self.store_statevectors, bool):
            raise TypeError("store_statevectors must be exactly bool")
        max_samples = _require_int(self.max_samples, "max_samples")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        circuit_specs = _coerce_specs(self.circuit_specs, CircuitGenerationSpec, "circuit_specs")
        distortion_specs = _coerce_specs(self.distortion_specs, DistortionSpec, "distortion_specs")
        if not circuit_specs:
            raise ValueError("at least one circuit spec is required")
        if not distortion_specs:
            raise ValueError("at least one distortion spec is required")
        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "base_seed", base_seed)
        object.__setattr__(self, "parameter_low", parameter_low)
        object.__setattr__(self, "parameter_high", parameter_high)
        object.__setattr__(self, "ideal_shots", ideal_shots)
        object.__setattr__(self, "max_samples", max_samples)
        object.__setattr__(self, "born_zero_atol", born_zero_atol)
        object.__setattr__(self, "circuit_specs", circuit_specs)
        object.__setattr__(self, "distortion_specs", distortion_specs)
        if predicted_sample_count(self) > max_samples:
            raise ValueError("predicted sample count exceeds max_samples")
        canonical_json(config_to_dict(self))


def _coerce_specs(values: Any, spec_type: type, field_name: str) -> list[Any]:
    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list")
    result = []
    allowed = set(spec_type.__dataclass_fields__)  # type: ignore[attr-defined]
    for item in values:
        if isinstance(item, spec_type):
            result.append(item)
        elif isinstance(item, Mapping):
            extra = set(item) - allowed
            if extra:
                raise ValueError(f"Unknown {spec_type.__name__} fields: {sorted(extra)}")
            result.append(spec_type(**dict(item)))
        else:
            raise TypeError(f"{field_name} entries must be {spec_type.__name__} or mappings")
    return list(result)


def predicted_sample_count(config: DatasetGenerationConfig) -> int:
    """Return the exact number of clean/distorted comparison samples to be generated."""
    return sum(spec.repetitions for spec in config.circuit_specs) * len(config.distortion_specs)


def config_to_dict(config: DatasetGenerationConfig) -> dict[str, Any]:
    """Convert a generation config to a strict JSON-compatible dictionary."""
    return asdict(config)


def config_from_dict(payload: dict[str, Any]) -> DatasetGenerationConfig:
    """Load a generation config from a dictionary, rejecting unknown fields."""
    if not isinstance(payload, Mapping):
        raise TypeError("config payload must be a mapping")
    allowed = set(DatasetGenerationConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown config fields: {sorted(extra)}")
    data = dict(payload)
    data["circuit_specs"] = _coerce_specs(data.get("circuit_specs", []), CircuitGenerationSpec, "circuit_specs")
    data["distortion_specs"] = _coerce_specs(data.get("distortion_specs", []), DistortionSpec, "distortion_specs")
    return DatasetGenerationConfig(**data)


def scientific_generation_payload(config: DatasetGenerationConfig) -> dict[str, Any]:
    """Payload for the exact scientific sample universe, excluding operational settings."""
    return {
        "schema_version": config.schema_version,
        "base_seed": config.base_seed,
        "circuit_specs": [asdict(spec) for spec in config.circuit_specs],
        "distortion_specs": [asdict(spec) for spec in config.distortion_specs],
        "parameter_low": config.parameter_low,
        "parameter_high": config.parameter_high,
        "metric_schema_version": PHASE7_METRIC_SCHEMA_VERSION,
    }


def scientific_generation_id(config: DatasetGenerationConfig) -> str:
    """Stable ID for the exact simulator-derived scientific sample universe."""
    return make_deterministic_id("generation", scientific_generation_payload(config))


def config_id(config: DatasetGenerationConfig) -> str:
    """Operational config ID that may include labels/storage/shot/test guard settings."""
    return make_deterministic_id("config", config_to_dict(config))


def save_generation_config(config: DatasetGenerationConfig, path: str | Path) -> None:
    """Write a generation config as stable strict JSON."""
    Path(path).write_text(json.dumps(config_to_dict(config), sort_keys=True, indent=2, allow_nan=False) + "\n")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite JSON constant in generation config: {value}")


def load_generation_config(path: str | Path) -> DatasetGenerationConfig:
    """Load a generation config from a strict JSON file."""
    try:
        payload = json.loads(Path(path).read_text(), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed generation config JSON: {path}") from exc
    return config_from_dict(payload)
