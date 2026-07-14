from __future__ import annotations

from pathlib import Path

import pytest

from triqto.actions.operational_adapter import (
    OPERATIONAL_ACTION_FAMILIES,
    OPERATIONAL_ACTION_FEATURE_NAMES,
    load_operational_view_adapter_config,
)
from triqto.actions.operational_config import load_operational_action_smoke_config

ROOT = Path(__file__).resolve().parents[1]


def test_active_operational_configs_load_and_preserve_claim_boundaries() -> None:
    operational = load_operational_action_smoke_config(
        ROOT / "configs/actions/operational_smoke.yaml"
    )
    adapter = load_operational_view_adapter_config(
        ROOT / "configs/training_views/operational_actions_smoke.yaml"
    )
    assert operational.evidence_tier == "mixed_offline"
    assert operational.physical_hardware is False
    assert len(operational.probe_bases) == operational.backend_n_qubits
    assert adapter.feature_names == OPERATIONAL_ACTION_FEATURE_NAMES
    assert adapter.family_order == OPERATIONAL_ACTION_FAMILIES
    assert adapter.require_availability_mask is True
    assert adapter.require_zero_operational_targets is True
    assert adapter.require_no_privilege is True


def test_operational_config_fails_closed_on_hardware_or_target_relaxation(tmp_path: Path) -> None:
    bad_operational = tmp_path / "bad_operational.yaml"
    bad_operational.write_text(
        "schema_version: triqto.operational_smoke_config.v1\n"
        "probe_bases: [X, Y]\n"
        "probe_shots: 32\n"
        "seed: 1\n"
        "backend_n_qubits: 2\n"
        "backend_name: local\n"
        "transpilation_optimization_level: 1\n"
        "semantic_tolerance: 1.0e-10\n"
        "evidence_tier: mixed_offline\n"
        "physical_hardware: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="physical hardware"):
        load_operational_action_smoke_config(bad_operational)

    bad_adapter = tmp_path / "bad_adapter.yaml"
    bad_adapter.write_text(
        "schema_version: triqto.operational_view_adapter.v1\n"
        "feature_names: [depth_delta, size_delta, two_qubit_gate_delta, acquires_evidence, is_no_op]\n"
        "family_order: [logical_correction, diagnostic_evidence_acquisition, compilation, semantics_preserving_optimization]\n"
        "require_availability_mask: true\n"
        "require_zero_operational_targets: false\n"
        "require_no_privilege: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="require_zero_operational_targets"):
        load_operational_view_adapter_config(bad_adapter)
