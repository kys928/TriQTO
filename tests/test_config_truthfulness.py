from __future__ import annotations

from pathlib import Path

import pytest

from triqto.config.loader import load_config
from triqto.config.validators import (
    SUPPORTED_DISTORTIONS,
    UnsupportedConfigError,
    validate_config_file,
    validate_config_tree,
)
from triqto.data_generation import load_generation_config


def test_all_repository_configs_validate_capability_boundaries() -> None:
    results = validate_config_tree("configs")
    assert results
    unsupported = {
        result.path.relative_to(Path.cwd()).as_posix()
        for result in results
        if not result.active
    }
    assert "configs/data/local_smoke_test.yaml" in unsupported
    assert "configs/data/monster_generation.yaml" in unsupported
    assert "configs/data/hardware_validation.yaml" in unsupported


def test_supported_distortions_come_from_real_registry() -> None:
    assert "entangling_rzz_drift" in SUPPORTED_DISTORTIONS
    assert "mixed_unitary_drift" in SUPPORTED_DISTORTIONS
    assert "entangling_overrotation" not in SUPPORTED_DISTORTIONS


def test_active_config_rejects_unsupported_mode(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "simulation_modes: [ideal_statevector, fake_backend]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported simulation_modes"):
        validate_config_file(path)


def test_active_config_rejects_unknown_backend(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("backends: [invented_backend]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown backends"):
        validate_config_file(path)


def test_capability_lists_reject_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "simulation_modes: [ideal_shot, ideal_shot]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        validate_config_file(path)


def test_unsupported_config_requires_reason(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "unsupported: true\nsimulation_modes: [fake_backend]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported_reason"):
        validate_config_file(path)


def test_loader_rejects_unsupported_config_by_default() -> None:
    with pytest.raises(UnsupportedConfigError, match="planning-only"):
        load_config("configs/data/monster_generation.yaml")


def test_loader_requires_explicit_planning_override() -> None:
    data = load_config(
        "configs/data/monster_generation.yaml",
        allow_unsupported_for_planning=True,
    )
    assert data["unsupported"] is True


def test_phase7_execution_loader_rejects_planning_yaml() -> None:
    with pytest.raises(UnsupportedConfigError, match="planning-only"):
        load_generation_config("configs/data/local_smoke_test.yaml")


def test_active_extends_is_rejected_until_resolved(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("extends: base.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extends is unsupported"):
        validate_config_file(path)
