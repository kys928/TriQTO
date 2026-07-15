"""Strict configuration for Phase 15.6 research-campaign preparation and execution."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

PHASE156_CONFIG_SCHEMA = "triqto.phase15_6.campaign.v1"
EXECUTION_DEVICES = ("auto", "cpu", "cuda")
RESOURCE_PROFILES = ("pilot", "recommended", "large")


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value.strip()


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _unique_ints(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_int(item, f"{name} item") for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonnegative integers")
    return result


def _unique_floats(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_float(item, f"{name} item") for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique finite nonnegative values")
    return result


@dataclass(frozen=True, slots=True)
class DataBuildConfig:
    """Bounded Phase 8/9/11 construction settings for one Phase 7 universe."""

    include_supplemental_counts: bool = False
    action_candidate_magnitudes: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    max_candidates_per_sample: int = 128
    max_edits_per_action: int = 32
    topology_min_points: int = 3
    topology_betti_grid_size: int = 16
    topology_top_k_lifetimes: int = 4
    topology_max_points_per_group: int = 512
    topology_max_groups: int = 2048
    topology_max_statevector_amplitudes: int = 1024
    topology_include_hilbert: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "include_supplemental_counts",
            _bool(self.include_supplemental_counts, "include_supplemental_counts"),
        )
        object.__setattr__(
            self,
            "action_candidate_magnitudes",
            _unique_floats(self.action_candidate_magnitudes, "action_candidate_magnitudes"),
        )
        for name in (
            "max_candidates_per_sample",
            "max_edits_per_action",
            "topology_min_points",
            "topology_betti_grid_size",
            "topology_top_k_lifetimes",
            "topology_max_points_per_group",
            "topology_max_groups",
            "topology_max_statevector_amplitudes",
        ):
            object.__setattr__(self, name, _int(getattr(self, name), name, minimum=1))
        object.__setattr__(
            self,
            "topology_include_hilbert",
            _bool(self.topology_include_hilbert, "topology_include_hilbert"),
        )


@dataclass(frozen=True, slots=True)
class PodRequirements:
    """Fail-closed minimums checked before a user launches expensive stages."""

    minimum_cpu_cores: int = 8
    minimum_memory_gb: float = 32.0
    minimum_free_disk_gb: float = 100.0
    minimum_gpu_vram_gb: float = 12.0
    require_cuda_for_training: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_cpu_cores", _int(self.minimum_cpu_cores, "minimum_cpu_cores", minimum=1))
        object.__setattr__(self, "minimum_memory_gb", _float(self.minimum_memory_gb, "minimum_memory_gb", minimum=1.0))
        object.__setattr__(self, "minimum_free_disk_gb", _float(self.minimum_free_disk_gb, "minimum_free_disk_gb", minimum=1.0))
        object.__setattr__(self, "minimum_gpu_vram_gb", _float(self.minimum_gpu_vram_gb, "minimum_gpu_vram_gb"))
        object.__setattr__(
            self,
            "require_cuda_for_training",
            _bool(self.require_cuda_for_training, "require_cuda_for_training"),
        )


@dataclass(frozen=True, slots=True)
class Phase156CampaignConfig:
    """Versioned research campaign contract.

    Paths are interpreted relative to the repository root unless absolute.
    Generated data, checkpoints, and reports must live outside the repository.
    """

    campaign_name: str
    generation_config: str
    training_view_config: str
    model_config: str
    training_config: str
    phase15_5_config: str
    schema_version: str = PHASE156_CONFIG_SCHEMA
    campaign_seed: int = 2026
    training_seeds: tuple[int, ...] = (2026, 2027, 2028)
    execution_device: str = "auto"
    resource_profile: str = "recommended"
    data_build: DataBuildConfig = field(default_factory=DataBuildConfig)
    pod_requirements: PodRequirements = field(default_factory=PodRequirements)
    run_phase15_5: bool = True
    physical_hardware: bool = False
    topology_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != PHASE156_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must equal {PHASE156_CONFIG_SCHEMA!r}")
        object.__setattr__(self, "campaign_name", _nonblank(self.campaign_name, "campaign_name"))
        for name in (
            "generation_config",
            "training_view_config",
            "model_config",
            "training_config",
            "phase15_5_config",
        ):
            object.__setattr__(self, name, _nonblank(getattr(self, name), name))
        object.__setattr__(self, "campaign_seed", _int(self.campaign_seed, "campaign_seed"))
        object.__setattr__(self, "training_seeds", _unique_ints(self.training_seeds, "training_seeds"))
        device = _nonblank(self.execution_device, "execution_device")
        if device not in EXECUTION_DEVICES:
            raise ValueError(f"execution_device must be one of {list(EXECUTION_DEVICES)}")
        profile = _nonblank(self.resource_profile, "resource_profile")
        if profile not in RESOURCE_PROFILES:
            raise ValueError(f"resource_profile must be one of {list(RESOURCE_PROFILES)}")
        object.__setattr__(self, "execution_device", device)
        object.__setattr__(self, "resource_profile", profile)
        build = self.data_build
        if isinstance(build, Mapping):
            build = DataBuildConfig(**dict(build))
        if not isinstance(build, DataBuildConfig):
            raise TypeError("data_build must be DataBuildConfig or mapping")
        requirements = self.pod_requirements
        if isinstance(requirements, Mapping):
            requirements = PodRequirements(**dict(requirements))
        if not isinstance(requirements, PodRequirements):
            raise TypeError("pod_requirements must be PodRequirements or mapping")
        object.__setattr__(self, "data_build", build)
        object.__setattr__(self, "pod_requirements", requirements)
        object.__setattr__(self, "run_phase15_5", _bool(self.run_phase15_5, "run_phase15_5"))
        if self.physical_hardware is not False:
            raise ValueError("Phase 15.6 campaign preparation cannot enable physical hardware")
        if float(self.topology_loss_weight) != 0.0:
            raise ValueError("topology_loss_weight must remain exactly 0.0")
        object.__setattr__(self, "physical_hardware", False)
        object.__setattr__(self, "topology_loss_weight", 0.0)
        json.dumps(phase156_config_to_dict(self), sort_keys=True, allow_nan=False)


def phase156_config_to_dict(config: Phase156CampaignConfig) -> dict[str, Any]:
    if not isinstance(config, Phase156CampaignConfig):
        raise TypeError("config must be Phase156CampaignConfig")
    payload = asdict(config)
    payload["training_seeds"] = list(config.training_seeds)
    payload["data_build"]["action_candidate_magnitudes"] = list(config.data_build.action_candidate_magnitudes)
    return payload


def phase156_config_from_dict(payload: Mapping[str, Any]) -> Phase156CampaignConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 15.6 config must be a mapping")
    allowed = set(Phase156CampaignConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"unknown Phase 15.6 config fields: {sorted(extra)}")
    values = dict(payload)
    if "training_seeds" in values:
        values["training_seeds"] = tuple(values["training_seeds"])
    if "data_build" in values:
        raw = dict(values["data_build"])
        if "action_candidate_magnitudes" in raw:
            raw["action_candidate_magnitudes"] = tuple(raw["action_candidate_magnitudes"])
        values["data_build"] = DataBuildConfig(**raw)
    if "pod_requirements" in values:
        values["pod_requirements"] = PodRequirements(**dict(values["pod_requirements"]))
    return Phase156CampaignConfig(**values)


def load_phase156_config(path: str | Path) -> Phase156CampaignConfig:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed Phase 15.6 JSON config: {target}") from exc
    return phase156_config_from_dict(payload)


def save_phase156_config(config: Phase156CampaignConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(phase156_config_to_dict(config), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid non-finite Phase 15.6 JSON constant: {value}")


__all__ = [
    "DataBuildConfig",
    "EXECUTION_DEVICES",
    "PHASE156_CONFIG_SCHEMA",
    "PodRequirements",
    "RESOURCE_PROFILES",
    "Phase156CampaignConfig",
    "load_phase156_config",
    "phase156_config_from_dict",
    "phase156_config_to_dict",
    "save_phase156_config",
]
