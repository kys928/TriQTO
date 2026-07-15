from __future__ import annotations

import json
from pathlib import Path

import pytest

from triqto.phase15_6 import (
    PodRequirements,
    build_campaign_plan,
    inspect_phase156_environment,
    load_phase156_config,
    phase156_config_from_dict,
    phase156_config_to_dict,
    prepare_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "phase15_6_research_pilot.json"


def test_phase156_config_roundtrip_and_claim_boundaries() -> None:
    config = load_phase156_config(CONFIG)
    assert phase156_config_from_dict(phase156_config_to_dict(config)) == config
    payload = phase156_config_to_dict(config)
    payload["physical_hardware"] = True
    with pytest.raises(ValueError, match="physical hardware"):
        phase156_config_from_dict(payload)
    payload = phase156_config_to_dict(config)
    payload["topology_loss_weight"] = 0.1
    with pytest.raises(ValueError, match="exactly 0.0"):
        phase156_config_from_dict(payload)
    payload = phase156_config_to_dict(config)
    payload["training_seeds"] = [2026, 2026]
    with pytest.raises(ValueError, match="unique"):
        phase156_config_from_dict(payload)


def test_phase156_plan_validates_research_inputs() -> None:
    config = load_phase156_config(CONFIG)
    plan = build_campaign_plan(repo_root=ROOT, config=config)
    assert plan["schema"] == "triqto.phase15_6.plan.v1"
    assert plan["dataset"]["scientific_sample_count"] == 13_440
    assert plan["dataset"]["max_qubits"] == 8
    assert plan["training"]["training_seeds"] == [2026, 2027, 2028]
    assert plan["split_contract"]["test_used_for_optimization"] is False
    assert plan["claim_boundaries"]["physical_hardware"] is False
    assert plan["claim_boundaries"]["topology_loss_weight"] == 0.0
    assert plan["resource_estimate"]["estimate_kind"] == "conservative_heuristic_not_benchmark"


def test_phase156_action_guardrail_covers_observed_pilot_candidate_count() -> None:
    config = load_phase156_config(CONFIG)
    # The first real pilot run produced 266 unique candidates for one sample.
    # Keep operational headroom without changing the scientific action universe.
    assert config.data_build.max_candidates_per_sample >= 512


def test_prepare_campaign_is_idempotent_and_external(tmp_path: Path) -> None:
    config = load_phase156_config(CONFIG)
    workspace = tmp_path / "campaign"
    first = prepare_campaign(repo_root=ROOT, workspace=workspace, config=config)
    second = prepare_campaign(repo_root=ROOT, workspace=workspace, config=config)
    assert first["campaign_id"] == second["campaign_id"]
    assert (workspace / "campaign_plan.json").is_file()
    assert (workspace / "campaign_state.json").is_file()
    snapshot_names = {
        path.name for path in (workspace / "source_config_snapshots").iterdir()
    }
    assert snapshot_names == {
        "generation.json",
        "model.json",
        "phase15_5.json",
        "training.json",
        "training_view.yaml",
    }


def test_preflight_report_is_machine_readable(tmp_path: Path) -> None:
    report = inspect_phase156_environment(
        workspace=tmp_path,
        requirements=PodRequirements(
            minimum_cpu_cores=1,
            minimum_memory_gb=1.0,
            minimum_free_disk_gb=1.0,
            minimum_gpu_vram_gb=0.0,
            require_cuda_for_training=False,
        ),
        training_device="cpu",
    )
    json.dumps(report, sort_keys=True, allow_nan=False)
    assert report["schema"] == "triqto.phase15_6.environment.v1"
    assert isinstance(report["ready"], bool)
    assert report["cuda"]["available"] in {True, False}


def test_unknown_phase156_fields_fail_closed() -> None:
    payload = phase156_config_to_dict(load_phase156_config(CONFIG))
    payload["future_magic"] = True
    with pytest.raises(ValueError, match="unknown Phase 15.6"):
        phase156_config_from_dict(payload)
