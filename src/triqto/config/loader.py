"""Safe YAML config loading for TriQTO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from triqto.config.validators import validate_config_file


def describe_contract() -> str:
    return "TriQTO config loader uses yaml.safe_load and validates capability boundaries."


def load_config(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    config_path = Path(path)
    if validate:
        validate_config_file(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {} if data is None else data
