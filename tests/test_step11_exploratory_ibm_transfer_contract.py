from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "v0_2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_runner():
    path = SCRIPTS / "run_step11_exploratory_ibm_transfer_pilot.py"
    spec = importlib.util.spec_from_file_location("step11_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(
        (ROOT / "configs" / "v0_2" / "step11_exploratory_ibm_transfer_pilot.json").read_text()
    )


def test_step11_is_frozen_exploratory_hardware_stage():
    cfg = _config()
    assert cfg["schema"] == "triqto.v0_2.step11_exploratory_ibm_transfer_pilot.v1"
    assert cfg["status"] == "FROZEN_BEFORE_PHYSICAL_QPU_EXECUTION"
    assert cfg["source_step10d"]["primary_hardware_candidate"] == "step10c_warm_start"
    assert cfg["source_step10d"]["further_simulator_tuning_before_hardware_permitted"] is False
    assert cfg["scientific_boundaries"]["exploratory_only"] is True
    assert cfg["scientific_boundaries"]["simulator_full_gate_previously_met"] is False
    assert cfg["scientific_boundaries"]["confirmatory_claim"] is False
    assert cfg["scientific_boundaries"]["qpu_results_used_for_tuning"] is False
    assert cfg["scientific_boundaries"]["qpu_results_used_for_retroactive_model_selection"] is False


def test_primary_and_paired_baseline_are_fixed_before_qpu():
    cfg = _config()
    primary = cfg["primary_candidate"]
    baseline = cfg["paired_baseline"]
    assert primary["name"] == "step10c_warm_start"
    assert primary["benchmark_id"] == "benchmark_f9478da45d68795655259054"
    assert primary["effect_threshold"] == pytest.approx(-0.125638447701931)
    assert primary["expected_trainable_parameter_count"] == 453829
    assert set(primary["checkpoint_sha256"]) == {
        "warm_start__seed1701.pt",
        "warm_start__seed1702.pt",
        "warm_start__seed1703.pt",
    }
    assert baseline["name"] == "step9a_deployment_ensemble"
    assert baseline["bundle_id"] == "deploy_ac536a74b2f8dd571d353a12"
    assert cfg["predeclared_analysis"]["same_qpu_counts_used_for_both_models"] is True
    assert cfg["scientific_boundaries"]["primary_candidate_may_not_change_after_qpu"] is True


def test_anchor_matrix_and_acquisition_budget_are_exact():
    runner = _load_runner()
    cfg = _config()
    cases = runner.build_pilot_cases(cfg, (0, 1, 2))
    assert len(cases) == 12
    assert {case.family for case in cases} == {"bell_like", "ghz", "phase_interference"}
    assert {case.condition for case in cases} == {"clean", "rz_drift", "rx_overrotation", "ry_overrotation"}
    assert sum(case.expected_effect for case in cases) == 9
    phase_distorted = [
        case for case in cases
        if case.family == "phase_interference" and case.expected_effect
    ]
    assert len(phase_distorted) == 3
    execution = cfg["execution"]
    assert execution["shots_per_program"] == 4096
    assert execution["programs_per_case"] == 6
    assert execution["total_programs"] == 72
    assert execution["total_executions"] == 294912
    assert execution["max_execution_time_seconds"] == 300
    assert execution["measurement_mitigation"] is False
    assert execution["dynamical_decoupling"] is False
    assert execution["twirling"] is False


def test_targeted_phase_interpretation_is_predeclared():
    runner = _load_runner()
    def rows(correct: int):
        out = []
        truths = ["rz_drift", "rx_overrotation", "ry_overrotation"]
        wrong = {"rz_drift": "rx_overrotation", "rx_overrotation": "ry_overrotation", "ry_overrotation": "rz_drift"}
        for i, truth in enumerate(truths):
            pred = truth if i < correct else wrong[truth]
            out.append({
                "family": "phase_interference",
                "expected_effect": True,
                "expected_mechanism": truth,
                "step10c_prediction": {"mechanism_prediction": pred},
            })
        return out

    assert runner._targeted_phase_result(rows(3), "step10c_prediction")["predeclared_interpretation"].startswith("strong")
    assert runner._targeted_phase_result(rows(2), "step10c_prediction")["predeclared_interpretation"].startswith("partial")
    assert runner._targeted_phase_result(rows(1), "step10c_prediction")["predeclared_interpretation"].startswith("weak_or_absent")
    assert runner._targeted_phase_result(rows(0), "step10c_prediction")["mechanism_correct_count"] == 0


def test_frozen_contract_rejects_scientific_relaxation():
    runner = _load_runner()
    cfg = _config()
    runner._assert_frozen_contract(cfg)
    bad = json.loads(json.dumps(cfg))
    bad["execution"]["measurement_mitigation"] = True
    with pytest.raises(RuntimeError, match="measurement_mitigation"):
        runner._assert_frozen_contract(bad)
    bad = json.loads(json.dumps(cfg))
    bad["scientific_boundaries"]["qpu_results_used_for_tuning"] = True
    with pytest.raises(RuntimeError, match="scientific boundary"):
        runner._assert_frozen_contract(bad)


def test_qpu_submission_is_gated_by_explicit_execute_branch_and_token():
    runner = _load_runner()
    cfg = _config()
    source = inspect.getsource(runner.main)
    assert "if args.execute_physical_qpu:" in source
    assert "physical QPU execution requires --confirmation-token" in source
    assert cfg["execution"]["explicit_confirmation_token"] == "STEP11_EXPLORATORY_IBM_QPU"
    execute_pos = source.index("execute_plan(")
    plan_pos = source.index("make_plan(")
    branch_pos = source.index("if args.execute_physical_qpu:")
    assert branch_pos < execute_pos < plan_pos
