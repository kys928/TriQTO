"""Capability-claim validation for repository YAML planning configs.

Executable Phase 7/9/11/12/13/14 configs retain their typed loaders. This
module prevents broad repository YAML files from presenting future capabilities
as active and derives implemented distortion names from the real registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from triqto.distortions import list_distortions

SUPPORTED_SIMULATION_MODES = frozenset({"ideal_statevector", "ideal_shot"})
UNSUPPORTED_SIMULATION_MODES = frozenset(
    {"noisy_shot", "density_matrix", "fake_backend", "hardware_runtime"}
)
SUPPORTED_DISTORTIONS = frozenset(list_distortions())
UNSUPPORTED_DISTORTIONS = frozenset(
    {
        "entangling_overrotation",
        "readout_noise",
        "depolarizing_noise",
        "amplitude_damping",
        "phase_damping",
        "thermal_relaxation",
        "mixed_noise",
        "transpilation_layout_distortion",
    }
)
SUPPORTED_ACTIONS = frozenset(
    {"no_op", "append_rx", "append_ry", "append_rz", "append_rzz"}
)
UNSUPPORTED_ACTIONS = frozenset(
    {
        "rz_phase_shift",
        "rx_amplitude_shift",
        "ry_amplitude_shift",
        "entangler_adjustment",
        "gate_removal",
        "layout_change",
        "transpiler_change",
        "measurement_basis_probe",
        "depth_reduction",
    }
)
SUPPORTED_BACKENDS = frozenset({"offline_ideal"})
UNSUPPORTED_BACKENDS = frozenset({"fake_backend", "ibm_runtime"})


class UnsupportedConfigError(ValueError):
    """Raised when an explicitly unsupported planning config is loaded to run."""


@dataclass(frozen=True)
class ConfigValidationResult:
    path: Path
    active: bool
    unsupported_reason: str | None
    warnings: tuple[str, ...] = ()


def describe_contract() -> str:
    return (
        "TriQTO capability validator: repository planning YAMLs cannot advertise "
        "unimplemented execution modes as active."
    )


def load_yaml_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{source}: capability config must be YAML")
    if not source.is_file():
        raise ValueError(f"{source}: capability config must be a regular file")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if data is None:
        return source, {}
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top-level YAML value must be a mapping")
    return source, data


def _unsupported_state(data: Mapping[str, Any]) -> tuple[bool, str | None]:
    marker = data.get("unsupported", False)
    if not isinstance(marker, bool):
        raise TypeError("unsupported must be exactly boolean")
    reason = data.get("unsupported_reason")
    if marker:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "unsupported configs require non-empty unsupported_reason"
            )
        return True, reason.strip()
    if reason is not None:
        raise ValueError("active configs must not define unsupported_reason")
    return False, None


def _string_list(
    path: Path,
    data: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    values = data.get(key)
    if values is None:
        return ()
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not all(isinstance(value, str) and value.strip() for value in values)
    ):
        raise ValueError(f"{path}: {key} must be a list of nonblank strings")
    normalized = tuple(value.strip() for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{path}: {key} must not contain duplicates")
    return normalized


def _validate_capability_list(
    *,
    path: Path,
    data: Mapping[str, Any],
    key: str,
    supported: frozenset[str],
    unsupported: frozenset[str],
    inactive: bool,
) -> None:
    values = set(_string_list(path, data, key))
    unknown = sorted(values - supported - unsupported)
    if unknown:
        raise ValueError(f"{path}: unknown {key}: {unknown}")
    unsupported_used = sorted(values & unsupported)
    if unsupported_used and not inactive:
        raise ValueError(
            f"{path}: active config references unsupported {key}: "
            f"{unsupported_used}"
        )


def validate_config_data(
    data: Mapping[str, Any],
    *,
    path: str | Path,
) -> ConfigValidationResult:
    """Validate already-parsed YAML without re-reading the source file."""
    yaml_path = Path(path)
    inactive, reason = _unsupported_state(data)
    if "extends" in data and not inactive:
        raise ValueError(
            f"{yaml_path}: extends is unsupported until inheritance is "
            "resolved before validation"
        )

    definitions = (
        (
            "simulation_modes",
            SUPPORTED_SIMULATION_MODES,
            UNSUPPORTED_SIMULATION_MODES,
        ),
        ("distortions", SUPPORTED_DISTORTIONS, UNSUPPORTED_DISTORTIONS),
        ("candidate_actions", SUPPORTED_ACTIONS, UNSUPPORTED_ACTIONS),
        ("backends", SUPPORTED_BACKENDS, UNSUPPORTED_BACKENDS),
    )
    for key, supported, unsupported in definitions:
        _validate_capability_list(
            path=yaml_path,
            data=data,
            key=key,
            supported=supported,
            unsupported=unsupported,
            inactive=inactive,
        )

    hardware = data.get("hardware")
    if hardware is not None:
        if not isinstance(hardware, Mapping):
            raise ValueError(f"{yaml_path}: hardware must be a mapping")
        enabled = hardware.get("enabled_initially", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"{yaml_path}: hardware.enabled_initially must be boolean"
            )
        if enabled and not inactive:
            raise ValueError(
                f"{yaml_path}: hardware-enabled configs must be credential-gated "
                "and explicitly unsupported by default"
            )

    return ConfigValidationResult(
        path=yaml_path,
        active=not inactive,
        unsupported_reason=reason,
    )


def validate_config_file(path: str | Path) -> ConfigValidationResult:
    yaml_path, data = load_yaml_mapping(path)
    return validate_config_data(data, path=yaml_path)


def validate_config_tree(
    root: str | Path = "configs",
) -> list[ConfigValidationResult]:
    config_root = Path(root).expanduser().resolve(strict=True)
    if not config_root.is_dir():
        raise ValueError(f"{config_root}: config root must be a directory")
    return [
        validate_config_file(path)
        for path in sorted(
            (*config_root.rglob("*.yaml"), *config_root.rglob("*.yml"))
        )
    ]
