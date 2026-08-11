from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/audit_bdelta_hardware_feasibility_contract.py"
)
SPEC = importlib.util.spec_from_file_location("step4_bdelta_contract_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return {
        "hardware_measurement_contract": {"exact_probabilities_available_on_hardware": False},
        "reference_contract": {
            "primary_step5_reference_kind": "paired",
            "reference_kinds": {
                "paired": {"deployability": "HARDWARE_VALID"}
            },
        },
        "feature_contract": [
            {
                "name": "dx",
                "hardware_obtainability": "FINITE_SHOT_SAMPLER_DERIVED",
                "step5_role": "PRIMARY_HARDWARE_SCALABLE_CORE",
            },
            {
                "name": "state",
                "hardware_obtainability": "PRIVILEGED_SIMULATOR_ONLY",
                "step5_role": "TARGET_AUDIT_ONLY",
            },
        ],
        "scalable_core_ablation": {
            "minimum_raw_separation": 1e-6,
            "minimum_relative_separation": 0.25,
            "numerical_collision_score_max": 1e-10,
            "minimum_eligible_stratum_pairs": 2,
        },
    }


def counterfactual(matched: str, mechanism: str, phenotype: str, rms: float) -> dict:
    return {
        "matched_context_id": matched,
        "mechanism": mechanism,
        "phenotype": phenotype,
        "expectation_rms_x": rms,
        "expectation_rms_y": rms,
        "expectation_rms_z": rms,
    }


def pair(matched: str, pair_rms: float, *, terminal: bool = False, qubit: int = 1) -> dict:
    return {
        "matched_context_id": matched,
        "clean_context_id": "clean-a",
        "left_mechanism": "rz_drift",
        "right_mechanism": "rx_overrotation",
        "pair_type": "rz_vs_rx",
        "family": "ghz",
        "n_qubits": "4",
        "strength_key": "0.05",
        "affected_qubit": str(qubit),
        "qubit_position_class": "interior",
        "insertion_depth_bin": "middle",
        "terminal_insertion": str(terminal),
        "pair_expectation_rms_x": pair_rms,
        "pair_expectation_rms_y": pair_rms,
        "pair_expectation_rms_z": pair_rms,
    }


def test_hardware_contract_accepts_sampler_core_and_privileged_target_only() -> None:
    result = MODULE.validate_hardware_contract(config())
    assert result["hardware_contract_valid"] is True
    assert result["privileged_features_leaking_into_deployable_inputs"] == []


def test_hardware_contract_rejects_privileged_primary_input() -> None:
    cfg = config()
    cfg["feature_contract"][0]["hardware_obtainability"] = "PRIVILEGED_SIMULATOR_ONLY"
    result = MODULE.validate_hardware_contract(cfg)
    assert result["hardware_contract_valid"] is False


def test_expectation_only_pair_is_strong_when_raw_and_relative_thresholds_pass() -> None:
    cf = [
        counterfactual("m", "rz_drift", "phase_dominant", 0.02),
        counterfactual("m", "rx_overrotation", "population_dominant", 0.02),
    ]
    rows = MODULE.build_scalable_pair_rows(cf, [pair("m", 0.02)], config())
    assert len(rows) == 1
    assert rows[0]["effectful_pair"] is True
    assert rows[0]["expectation_only_strong_pair"] is True


def test_negligible_mechanism_removes_pair_from_primary_effectful_population() -> None:
    cf = [
        counterfactual("m", "rz_drift", "negligible", 0.0),
        counterfactual("m", "rx_overrotation", "population_dominant", 0.02),
    ]
    rows = MODULE.build_scalable_pair_rows(cf, [pair("m", 0.02)], config())
    assert rows[0]["effectful_pair"] is False


def test_stratification_can_expose_severe_family_failure() -> None:
    cfg = config()
    rows = []
    for i in range(3):
        rows.append(
            {
                "pair_type": "rz_vs_rx",
                "family": "ghz",
                "n_qubits": 4,
                "strength_key": "0.05",
                "affected_qubit": 1,
                "qubit_position_class": "interior",
                "insertion_depth_bin": "middle",
                "terminal_insertion": False,
                "expectation_only_strong_pair": i == 0,
                "expectation_only_numerical_collision": False,
                "expectation_only_pair_separation_score": 0.01,
                "expectation_only_relative_separation": 1.0,
            }
        )
    strata = MODULE.stratify_effectful(rows, cfg)
    ghz = next(
        row
        for row in strata
        if row["stratum_type"] == "family" and row["stratum_value"] == "ghz"
    )
    assert ghz["eligible"] is True
    assert abs(float(ghz["strong_pair_fraction"]) - 1.0 / 3.0) < 1e-12
