from __future__ import annotations

from pathlib import Path

import pytest

from triqto.config.validators import validate_config_file, validate_config_tree


def test_all_repository_configs_validate_capability_boundaries() -> None:
    results = validate_config_tree("configs")
    assert results
    unsupported = {r.path.as_posix() for r in results if not r.active}
    assert "configs/data/monster_generation.yaml" in unsupported
    assert "configs/data/hardware_validation.yaml" in unsupported


def test_active_config_rejects_unsupported_mode(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("simulation_modes: [ideal_statevector, fake_backend]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported simulation_modes"):
        validate_config_file(path)


def test_unsupported_config_requires_reason(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("unsupported: true\nsimulation_modes: [fake_backend]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported_reason"):
        validate_config_file(path)
