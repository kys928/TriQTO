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
    path = SCRIPTS / "run_step12_independent_phase_generalization.py"
    spec = importlib.util.spec_from_file_location("step12_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(
        (ROOT / "configs" / "v0_2" / "step12_independent_phase_generalization.json").read_text()
    )


def test_step12_freezes_same_models_and_forbids_tuning():
    cfg = _config()
    assert cfg["schema"] == "triqto.v0_2.step12_independent_phase_generalization.v1"
    assert cfg["status"] == "FROZEN_BEFORE_PHYSICAL_QPU_EXECUTION"
    assert cfg["primary_candidate"]["name"] == "step10c_warm_start"
    assert cfg["primary_candidate"]["benchmark_id"] == "benchmark_f9478da45d68795655259054"
    assert cfg["primary_candidate"]["effect_threshold"] == pytest.approx(-0.125638447701931)
    assert cfg["primary_candidate"]["model_change_after_step11_allowed"] is False
    assert cfg["paired_baseline"]["name"] == "step9a_deployment_ensemble"
    boundaries = cfg["scientific_boundaries"]
    assert boundaries["new_training"] is False
    assert boundaries["model_weight_change"] is False
    assert boundaries["threshold_change"] is False
    assert boundaries["qpu_results_used_for_tuning"] is False
    assert boundaries["full_triqto_confirmatory_claim"] is False


def test_case_matrix_is_new_balanced_and_exact():
    runner = _load_runner()
    cfg = _config()
    cases = runner.build_generalization_cases(cfg, (10, 11, 12))
    assert len(cases) == 21
    assert sum(bool(case.expected_effect) for case in cases) == 18
    assert sum(not bool(case.expected_effect) for case in cases) == 3
    assert {case.family for case in cases} == {
        "cz_echo_ramsey", "dual_arm_recombination", "three_qubit_phase_fanout"
    }
    distorted = [case for case in cases if case.expected_effect]
    for mechanism in ("rz_drift", "rx_overrotation", "ry_overrotation"):
        assert sum(case.expected_mechanism == mechanism for case in distorted) == 6
    for strength in (0.13, 0.27):
        assert sum(abs(float(case.strength) - strength) < 1e-12 for case in distorted) == 9
    assert {case.physical_layout for case in cases if case.family == "cz_echo_ramsey"} == {(10, 11)}
    assert {case.physical_layout for case in cases if case.family == "dual_arm_recombination"} == {(12, 11)}
    assert {case.physical_layout for case in cases if case.family == "three_qubit_phase_fanout"} == {(12, 11, 10)}


def test_new_motifs_do_not_reuse_targeted_h_rz_h_cx_core():
    runner = _load_runner()
    cfg = _config()
    forbidden = ["h:q0", "rz:q0", "h:q0", "cx:q0-q1"]
    for raw in cfg["generalization_design"]["motifs"]:
        circuit, _ = runner._build_motif(raw["name"])
        signature = runner._operation_signature(circuit)
        assert signature == raw["reference_operation_signature"]
        assert runner._contains_subsequence(signature, forbidden) is False
    assert cfg["generalization_design"]["step9d_anchor_matrix_reused"] is False
    assert cfg["generalization_design"]["step10_bridge_motif_names_reused"] is False

    three, _ = runner._build_motif("three_qubit_phase_fanout")
    for item in three.data:
        q = [three.find_bit(bit).index for bit in item.qubits]
        if len(q) == 2:
            assert tuple(sorted(q)) in {(0, 1), (1, 2)}


def test_model_blind_identifiability_audit_passes_without_predictions():
    runner = _load_runner()
    cfg = _config()
    audit = runner.model_blind_identifiability_audit(cfg)
    assert audit["status"] == "PASS"
    assert audit["uses_model_predictions"] is False
    assert audit["uses_statevector_only"] is True
    assert audit["minimum_observed_low_strength_delta_norm"] >= 0.04
    assert audit["minimum_observed_low_strength_pairwise_mechanism_distance"] >= 0.1


def test_cross_backend_and_acquisition_budget_are_frozen():
    cfg = _config()
    assert cfg["source_step11"]["backend_name"] == "ibm_kingston"
    assert cfg["backend_selection"]["excluded_backend_names"] == ["ibm_kingston"]
    assert cfg["backend_selection"]["require_backend_different_from_step11"] is True
    execution = cfg["execution"]
    assert execution["case_count"] == 21
    assert execution["programs_per_case"] == 6
    assert execution["total_programs"] == 126
    assert execution["shots_per_program"] == 4096
    assert execution["total_executions"] == 516096
    assert execution["max_execution_time_seconds"] == 300
    assert execution["measurement_mitigation"] is False
    assert execution["dynamical_decoupling"] is False
    assert execution["twirling"] is False


def _fake_metrics(total: int, each_mech: int, each_strength: int, detected: int, fp: int):
    return {
        "distorted_mechanism_correct_count": total,
        "distorted_effect_detection_count": detected,
        "clean_effect_false_positive_count": fp,
        "by_mechanism": {
            "rz_drift": {"correct": each_mech, "count": 6},
            "rx_overrotation": {"correct": each_mech, "count": 6},
            "ry_overrotation": {"correct": each_mech, "count": 6},
        },
        "by_strength": {
            "0.13": {"correct": each_strength, "count": 9},
            "0.27": {"correct": each_strength, "count": 9},
        },
    }


def test_predeclared_gate_is_fixed_and_requires_all_criteria():
    runner = _load_runner()
    cfg = _config()
    primary = _fake_metrics(total=15, each_mech=5, each_strength=7, detected=16, fp=0)
    baseline = _fake_metrics(total=10, each_mech=3, each_strength=5, detected=14, fp=1)
    result = runner.evaluate_predeclared_gate(primary, baseline, cfg)
    assert result["passed"] is True
    assert result["interpretation"] == "NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_SUPPORTED"

    weak = _fake_metrics(total=13, each_mech=5, each_strength=7, detected=16, fp=0)
    result = runner.evaluate_predeclared_gate(weak, baseline, cfg)
    assert result["passed"] is False
    assert result["interpretation"] == "NARROW_CROSS_BACKEND_PHASE_GENERALIZATION_NOT_SUPPORTED"


def test_contract_rejects_relaxation_and_submission_requires_token():
    runner = _load_runner()
    cfg = _config()
    runner._assert_frozen_contract(cfg)
    bad = json.loads(json.dumps(cfg))
    bad["execution"]["measurement_mitigation"] = True
    with pytest.raises(RuntimeError, match="measurement_mitigation"):
        runner._assert_frozen_contract(bad)
    bad = json.loads(json.dumps(cfg))
    bad["backend_selection"]["excluded_backend_names"] = []
    with pytest.raises(RuntimeError, match="exclude"):
        runner._assert_frozen_contract(bad)

    source = inspect.getsource(runner.main)
    assert "if args.execute_physical_qpu:" in source
    assert "physical QPU execution requires --confirmation-token" in source
    assert cfg["execution"]["explicit_confirmation_token"] == "STEP12_INDEPENDENT_PHASE_GENERALIZATION_QPU"
    assert source.index("execute_plan(") < source.index("make_plan(")
