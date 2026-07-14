"""Validation helpers for TriQTO YAML configuration truthfulness.

The validator is deliberately conservative: active configs may reference only
execution modes, distortions, actions, and backends that are represented by the
offline scaffold. Future or hardware/noisy configs must be explicitly marked
``unsupported: true`` (or ``experimental: false`` with a reason) so they cannot
be mistaken for executable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SIMULATION_MODES = {"ideal_statevector", "ideal_shot"}
UNSUPPORTED_SIMULATION_MODES = {"noisy_shot", "density_matrix", "fake_backend", "hardware_runtime"}
SUPPORTED_DISTORTIONS = {"phase_rz_drift", "rx_overrotation", "ry_overrotation", "entangling_overrotation"}
UNSUPPORTED_DISTORTIONS = {
    "readout_noise",
    "depolarizing_noise",
    "amplitude_damping",
    "phase_damping",
    "thermal_relaxation",
    "mixed_noise",
    "transpilation_layout_distortion",
}
SUPPORTED_ACTIONS = {"rz_phase_shift", "rx_amplitude_shift", "ry_amplitude_shift", "entangler_adjustment"}
UNSUPPORTED_ACTIONS = {"gate_removal", "layout_change", "transpiler_change", "measurement_basis_probe", "depth_reduction"}


@dataclass(frozen=True)
class ConfigValidationResult:
    path: Path
    active: bool
    unsupported_reason: str | None
    warnings: tuple[str, ...] = ()


def describe_contract() -> str:
    """Return the implemented config validation contract."""
    return "TriQTO config validator: active configs must use supported offline modes or be explicitly unsupported."


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return data


def _is_explicitly_unsupported(data: dict[str, Any]) -> tuple[bool, str | None]:
    if data.get("unsupported") is True:
        reason = data.get("unsupported_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("unsupported configs require non-empty unsupported_reason")
        return True, reason
    if data.get("experimental") is False and "unsupported_reason" in data:
        reason = data.get("unsupported_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("experimental:false configs require non-empty unsupported_reason")
        return True, reason
    return False, None


def _reject_unknown_list(path: Path, data: dict[str, Any], key: str, supported: set[str], unsupported: set[str], inactive: bool) -> None:
    values = data.get(key, [])
    if values is None:
        return
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"{path}: {key} must be a list of strings")
    unknown = sorted(set(values) - supported - unsupported)
    if unknown:
        raise ValueError(f"{path}: unknown {key}: {unknown}")
    unsupported_used = sorted(set(values) & unsupported)
    if unsupported_used and not inactive:
        raise ValueError(f"{path}: active config references unsupported {key}: {unsupported_used}")


def validate_config_file(path: str | Path) -> ConfigValidationResult:
    """Validate one YAML config against current executable capability boundaries."""
    yaml_path = Path(path)
    data = _load_yaml(yaml_path)
    inactive, reason = _is_explicitly_unsupported(data)
    if "extends" in data and not inactive:
        raise ValueError(f"{yaml_path}: extending configs must be explicitly unsupported until resolved validation is implemented")
    _reject_unknown_list(yaml_path, data, "simulation_modes", SUPPORTED_SIMULATION_MODES, UNSUPPORTED_SIMULATION_MODES, inactive)
    _reject_unknown_list(yaml_path, data, "distortions", SUPPORTED_DISTORTIONS, UNSUPPORTED_DISTORTIONS, inactive)
    _reject_unknown_list(yaml_path, data, "candidate_actions", SUPPORTED_ACTIONS, UNSUPPORTED_ACTIONS, inactive)
    hardware = data.get("hardware")
    if isinstance(hardware, dict) and hardware.get("enabled_initially") is True and not inactive:
        raise ValueError(f"{yaml_path}: hardware-enabled configs must be credential-gated and explicitly unsupported by default")
    return ConfigValidationResult(path=yaml_path, active=not inactive, unsupported_reason=reason)


def validate_config_tree(root: str | Path = "configs") -> list[ConfigValidationResult]:
    """Validate all YAML files under a config root."""
    config_root = Path(root)
    return [validate_config_file(path) for path in sorted(config_root.rglob("*.yaml"))]
