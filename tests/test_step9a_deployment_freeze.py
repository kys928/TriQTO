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


def selected(seed: int, epoch: int, mech: float, effect: float, params: int = 453829) -> dict:
    return {
        "seed": seed,
        "selected_epoch": epoch,
        "trainable_parameter_count": params,
        "selection_summary": {
            "mechanism_balanced_accuracy": mech,
            "effect_balanced_accuracy": effect,
        },
    }


def test_contract_freezes_confirmed_ensemble_without_confirmatory_reuse() -> None:
    cfg = load_config()
    assert cfg["schema"] == MODULE.SCHEMA
    assert cfg["status"] == "FROZEN_AFTER_STEP8_CONFIRMATION_BEFORE_DEPLOYMENT_BUNDLE"
    frozen = cfg["frozen_model"]
    assert frozen["variant"] == "late_concat"
    assert frozen["seeds"] == [1701, 1702, 1703]
    assert frozen["selected_epochs"] == {"1701": 8, "1702": 11, "1703": 17}
    assert frozen["effect_threshold"] == pytest.approx(0.05939410626888275, abs=0.0)
    assert cfg["development_source"]["spent_step8_confirmatory_cohort_used"] is False
    assert cfg["scientific_boundaries"]["spent_confirmatory_reuse"] is False
    assert cfg["scientific_boundaries"]["new_threshold_selection"] is False
    assert cfg["scientific_boundaries"]["architecture_change"] is False


def test_validate_frozen_selection_accepts_archived_values() -> None:
    cfg = load_config()
    ref = cfg["frozen_selection_reference"]
    frozen = cfg["frozen_model"]
    row = selected(
        1701,
        8,
        ref["1701"]["mechanism_balanced_accuracy"],
        ref["1701"]["effect_balanced_accuracy"],
    )
    MODULE.validate_frozen_selection(selected=row, frozen=frozen, reference=ref)


def test_validate_frozen_selection_rejects_epoch_drift() -> None:
    cfg = load_config()
    ref = cfg["frozen_selection_reference"]
    frozen = cfg["frozen_model"]
    row = selected(
        1701,
        9,
        ref["1701"]["mechanism_balanced_accuracy"],
        ref["1701"]["effect_balanced_accuracy"],
    )
    with pytest.raises(RuntimeError, match="selected epoch changed"):
        MODULE.validate_frozen_selection(selected=row, frozen=frozen, reference=ref)


def test_validate_frozen_selection_rejects_metric_drift() -> None:
    cfg = load_config()
    ref = cfg["frozen_selection_reference"]
    frozen = cfg["frozen_model"]
    row = selected(
        1702,
        11,
        ref["1702"]["mechanism_balanced_accuracy"] + 1e-3,
        ref["1702"]["effect_balanced_accuracy"],
    )
    with pytest.raises(RuntimeError, match="failed frozen reproduction"):
        MODULE.validate_frozen_selection(selected=row, frozen=frozen, reference=ref)


def test_checkpoint_payload_is_cpu_state_dict_and_identity_bound() -> None:
    model = torch.nn.Linear(3, 2)
    payload = MODULE.checkpoint_payload(
        model=model,
        seed=1703,
        selected_epoch=17,
        config_sha256="sha256:cfg",
        step7_config_sha256="sha256:step7",
    )
    assert payload["architecture"] == "late_concat"
    assert payload["seed"] == 1703
    assert payload["selected_epoch"] == 17
    assert payload["config_sha256"] == "sha256:cfg"
    assert payload["step7_config_sha256"] == "sha256:step7"
    assert payload["state_dict"]
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())


def test_inference_contract_matches_confirmed_diagnostic_semantics() -> None:
    contract = load_config()["inference_contract"]
    assert contract["diagnostic_basis_order"] == ["Z", "X", "Y"]
    assert contract["diagnostic_basis_codes"] == [0, 1, 2]
    assert contract["delta_sign"] == "observed_minus_paired_reference"
    assert contract["reference_kind_code"] == 0
    assert contract["graph_semantics"] == "intended_reference_clean_circuit_only"
    assert contract["statevector_model_input"] is False
    assert contract["exact_diagnostic_model_input"] is False
