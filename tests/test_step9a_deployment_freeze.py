from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v0_2" / "freeze_step9a_deployment_bundle.py"
CONFIG = ROOT / "configs" / "v0_2" / "step9a_deployment_freeze.json"
SPEC = importlib.util.spec_from_file_location("triqto_step9a_freeze", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def refit_record(seed: int, epoch: int, mech: float, effect: float, params: int = 453829) -> dict:
    return {
        "seed": seed,
        "fixed_epoch": epoch,
        "selected_by_current_refit": False,
        "trainable_parameter_count": params,
        "selection_summary": {
            "mechanism_balanced_accuracy": mech,
            "effect_balanced_accuracy": effect,
        },
    }


def test_contract_discloses_post_confirmation_refit_and_no_exact_weight_claim() -> None:
    cfg = load_config()
    assert cfg["schema"] == MODULE.SCHEMA
    assert cfg["status"] == "FROZEN_POST_CONFIRMATION_FIXED_EPOCH_REFIT"
    frozen = cfg["frozen_model"]
    assert frozen["variant"] == "late_concat"
    assert frozen["seeds"] == [1701, 1702, 1703]
    assert frozen["fixed_training_epochs"] == {"1701": 8, "1702": 11, "1703": 17}
    assert frozen["deployment_effect_threshold"] == pytest.approx(0.05939410626888275, abs=0.0)
    assert frozen["current_refit_may_select_epoch"] is False
    assert frozen["current_refit_may_select_threshold"] is False
    assert frozen["exact_step8_checkpoint_weights_claimed"] is False
    assert cfg["scientific_boundaries"]["post_confirmation_fixed_epoch_refit"] is True
    assert cfg["scientific_boundaries"]["exact_step8_weights_reconstructed"] is False
    assert cfg["development_source"]["spent_step8_confirmatory_cohort_used"] is False


def test_replay_failure_record_preserves_observed_epoch_drift() -> None:
    record = load_config()["replay_failure_record"]
    assert record["seed"] == 1701
    assert record["archived_step8_selected_epoch"] == 8
    assert record["v1_replay_selected_epoch"] == 19
    assert record["exact_checkpoint_reconstruction_supported"] is False
    assert record["deployment_bundle_produced"] is False


def test_validate_refit_record_accepts_fixed_epoch_with_descriptive_metrics() -> None:
    cfg = load_config()
    frozen = cfg["frozen_model"]
    row = refit_record(1701, 8, 0.49, 0.70)
    MODULE.validate_refit_record(record=row, frozen=frozen)


def test_validate_refit_record_rejects_epoch_change_or_current_selection() -> None:
    frozen = load_config()["frozen_model"]
    with pytest.raises(RuntimeError, match="fixed epoch changed"):
        MODULE.validate_refit_record(record=refit_record(1701, 9, 0.49, 0.70), frozen=frozen)
    row = refit_record(1701, 8, 0.49, 0.70)
    row["selected_by_current_refit"] = True
    with pytest.raises(RuntimeError, match="not allowed to select an epoch"):
        MODULE.validate_refit_record(record=row, frozen=frozen)


def test_checkpoint_payload_is_cpu_and_disclaims_exact_step8_weight() -> None:
    model = torch.nn.Linear(3, 2)
    payload = MODULE.checkpoint_payload(
        model=model,
        seed=1703,
        fixed_epoch=17,
        config_sha256="sha256:cfg",
        step7_config_sha256="sha256:step7",
    )
    assert payload["architecture"] == "late_concat"
    assert payload["seed"] == 1703
    assert payload["fixed_epoch"] == 17
    assert payload["epoch_source"] == "archived_step8_selected_epoch"
    assert payload["exact_step8_checkpoint_weight"] is False
    assert payload["weight_provenance"] == "post_confirmation_fixed_epoch_refit_development_only"
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())


def test_bundle_id_is_bound_to_actual_checkpoint_hashes() -> None:
    identity = {"architecture": "late_concat", "seeds": [1701, 1702, 1703]}
    first = MODULE.deployment_bundle_id(identity, {"seed1701.pt": "sha256:a"})
    second = MODULE.deployment_bundle_id(identity, {"seed1701.pt": "sha256:b"})
    assert first.startswith("deploy_")
    assert first != second


def test_refuse_existing_successful_bundle_blocks_second_candidate(tmp_path: Path) -> None:
    MODULE.refuse_existing_successful_bundle(tmp_path)
    bundle = tmp_path / "deploy_existing"
    bundle.mkdir()
    (bundle / "bundle_complete.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="authoritative Step-9A deployment bundle already exists"):
        MODULE.refuse_existing_successful_bundle(tmp_path)


def test_inference_contract_matches_confirmed_diagnostic_semantics() -> None:
    contract = load_config()["inference_contract"]
    assert contract["diagnostic_basis_order"] == ["Z", "X", "Y"]
    assert contract["diagnostic_basis_codes"] == [0, 1, 2]
    assert contract["delta_sign"] == "observed_minus_paired_reference"
    assert contract["reference_kind_code"] == 0
    assert contract["graph_semantics"] == "intended_reference_clean_circuit_only"
    assert contract["statevector_model_input"] is False
    assert contract["exact_diagnostic_model_input"] is False
