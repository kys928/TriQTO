"""Fail-closed loader for TriQTO repository capability YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from triqto.config.validators import (
    UnsupportedConfigError,
    load_yaml_mapping,
    validate_config_data,
)


def describe_contract() -> str:
    return (
        "TriQTO capability YAML loader validates one parsed mapping and rejects "
        "unsupported planning configs by default."
    )


def load_config(
    path: str | Path,
    *,
    allow_unsupported_for_planning: bool = False,
) -> dict[str, Any]:
    config_path, data = load_yaml_mapping(path)
    result = validate_config_data(data, path=config_path)
    if not result.active and not allow_unsupported_for_planning:
        raise UnsupportedConfigError(
            f"{config_path} is planning-only and cannot be executed: "
            f"{result.unsupported_reason}"
        )
    return data
