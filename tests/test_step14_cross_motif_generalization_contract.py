from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v0_2" / "step14_cross_motif_generalization_training.json"


def _cfg() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_step14_status_and_architecture_are_frozen() -> None:
    cfg = _cfg()
    assert cfg["schema"] == "triqto.v0_2.step14_cross_motif_generalization_training.v1"
    assert cfg["status"] == "FROZEN_BEFORE_STEP14_DATASET_GENERATION"
    arch = cfg["architecture"]
    assert arch["variant"] == "late_concat"
    assert arch["expected_trainable_parameter_count"] == 453829
    assert arch["architecture_change"] is False
    assert arch["graph_information_contract_change"] is False
    assert arch["input_shape_change"] is False
    assert arch["new_auxiliary_head"] is False
    assert arch["new_invariance_loss"] is False


def test_step14_warm_start_identity_is_exact_step10c() -> None:
    cfg = _cfg()
    ws = cfg["warm_start"]
    assert ws["benchmark_id"] == "benchmark_f9478da45d68795655259054"
    assert ws["seeds"] == [1701, 1702, 1703]
    assert ws["load_state_dict_strict"] is True
    assert ws["optimizer_state_reused"] is False
    assert ws["new_optimizer"] is True
    assert ws["scratch_control"] is False
    assert ws["checkpoint_sha256"] == {
        "warm_start__seed1701.pt": "sha256:40f6c86981f038b3a44cd21148ebc67e4bac9db825bac9f4482a7fd098830769",
        "warm_start__seed1702.pt": "sha256:65b4535f89ec4d51f4126e2cb90ce9dc606d67ef2673b33740193e5aea10f39a",
        "warm_start__seed1703.pt": "sha256:d97601bcad8a6256d4433f12ca229ac0387acc927f859fe9e30ba4c2d3f3eb1f",
    }


def test_cross_motif_family_split_and_counts_are_exact() -> None:
    cfg = _cfg()
    d = cfg["cross_motif_dataset"]
    split = d["family_split"]
    counts = d["expected_counts"]
    assert d["family_count_total"] == 1050
    assert d["variants_per_family"] == 4
    assert d["examples_per_root"] == 13
    assert split["fold_expression"] == "family_index mod 7"
    assert split["fit_folds"] == [0, 1, 2, 3]
    assert split["selection_folds"] == [4]
    assert split["simulator_outer_folds"] == [5]
    assert split["future_hardware_reserve_folds"] == [6]
    assert (
        split["expected_fit_families"]
        + split["expected_selection_families"]
        + split["expected_simulator_outer_families"]
        + split["expected_future_hardware_reserve_families"]
        == d["family_count_total"]
    )
    assert counts["fit_roots"] == 2400
    assert counts["fit_examples"] == 31200
    assert counts["selection_roots"] == 600
    assert counts["selection_examples"] == 7800
    assert counts["simulator_outer_roots"] == 600
    assert counts["simulator_outer_examples"] == 7800
    assert counts["future_hardware_reserve_roots"] == 600
    assert counts["future_hardware_reserve_examples_if_later_materialized"] == 7800
    material = d["materialization_policy"]
    assert material["before_training"] == ["fit", "selection"]
    assert material["simulator_outer_materialized_only_after_selection_freeze"] is True
    assert material["future_hardware_reserve_materialized_in_step14"] is False
    assert material["future_hardware_reserve_metadata_inspection_in_step14"] is False


def test_cross_motif_factorial_design_has_no_obvious_label_proxy() -> None:
    cfg = _cfg()
    d = cfg["cross_motif_dataset"]
    assert d["mechanisms"] == ["rz_drift", "rx_overrotation", "ry_overrotation"]
    assert d["clean_controls_per_root"] == 1
    assert d["distorted_examples_per_root"] == 12
    strength = d["strength_design"]
    assert strength["values_per_root"] == 4
    assert strength["same_strengths_for_all_three_mechanisms_within_root"] is True
    assert strength["mechanism_is_fully_crossed_with_strength"] is True
    shots = d["shot_design"]
    assert shots["shot_levels"] == [512, 1024, 2048, 4096]
    assert shots["shot_level_may_not_encode_mechanism"] is True
    assert shots["shot_level_may_not_encode_clean_vs_distorted"] is True
    assert d["circuit_grammar"]["exact_step12_reference_signature_exclusion"] is True
    assert d["circuit_grammar"]["exact_step11_phase_interference_signature_exclusion"] is True


def test_step14_input_and_qpu_boundaries_are_hardware_facing_and_nonprivileged() -> None:
    cfg = _cfg()
    contract = cfg["cross_motif_dataset"]["input_contract"]
    assert contract["reuse_current_step7_graph_adapter"] is True
    assert contract["hardware_facing_graph_plus_paired_diagnostic_contract"] is True
    for key in (
        "statevector_as_model_input",
        "exact_ideal_diagnostics_as_model_input",
        "hidden_mechanism_as_model_input",
        "affected_qubit_as_model_input",
        "injection_boundary_as_model_input",
        "strength_as_model_input",
        "family_id_as_model_input",
        "topology_class_as_model_input",
        "split_fold_as_model_input",
        "qpu_counts_as_training_input",
    ):
        assert contract[key] is False
    b = cfg["scientific_boundaries"]
    assert b["step11_qpu_counts_used_for_training"] is False
    assert b["step12_qpu_counts_used_for_training"] is False
    assert b["step12_exact_cases_used_for_training"] is False
    assert b["qpu_execution_in_step14"] is False
    assert b["future_hardware_reserve_access"] is False


def test_training_is_single_predeclared_data_first_intervention() -> None:
    cfg = _cfg()
    t = cfg["training"]
    assert t["seeds"] == [1701, 1702, 1703]
    assert t["root_batch_size"] == 32
    assert t["max_epochs"] == 30
    assert t["early_stopping_patience"] == 5
    assert t["optimizer"] == "AdamW"
    assert t["learning_rate"] == 0.0001
    assert t["learning_rate_schedule"] == "constant"
    assert t["learning_rate_sweep"] is False
    assert t["weight_decay"] == 0.0001
    assert t["gradient_clip_norm"] == 1.0
    assert t["effect_loss_weight"] == 1.0
    assert t["mechanism_loss_weight"] == 1.0
    assert t["new_loss_terms"] is False
    assert t["domain_weights"] == {
        "legacy_original": 1.0,
        "legacy_bridge": 1.0,
        "cross_motif": 1.0,
    }


def test_selection_and_outer_gate_are_predeclared_and_outer_selects_nothing() -> None:
    cfg = _cfg()
    sel = cfg["per_seed_checkpoint_selection"]
    assert sel["outer_data_selects_nothing"] is True
    assert sel["step12_data_selects_nothing"] is True
    assert sel["human_override"] is False
    threshold = cfg["ensemble_effect_threshold"]
    assert threshold["selected_after_all_three_seed_checkpoints_are_frozen"] is True
    assert threshold["frozen_before_any_outer_materialization"] is True
    outer = cfg["outer_evaluation"]
    gate = outer["support_gate"]
    assert gate == {
        "cross_motif_mechanism_balanced_accuracy_minimum": 0.80,
        "cross_motif_mechanism_bootstrap_ci_lower_minimum": 0.75,
        "cross_motif_minimum_class_recall": 0.70,
        "cross_motif_effect_balanced_accuracy_minimum": 0.90,
        "candidate_minus_step10c_cross_motif_mechanism_ba_minimum": 0.05,
        "paired_bootstrap_ci_lower_for_candidate_minus_step10c_mechanism_ba_must_exceed": 0.0,
        "legacy_original_mechanism_ba_max_drop_vs_step10c": 0.02,
        "legacy_original_effect_ba_max_drop_vs_step10c": 0.02,
        "legacy_bridge_mechanism_ba_max_drop_vs_step10c": 0.02,
        "legacy_bridge_effect_ba_max_drop_vs_step10c": 0.02,
    }
    assert outer["gate_interpretation"]["all_criteria_met"] == "CROSS_MOTIF_GENERALIZATION_REPAIR_SUPPORTED_IN_SIMULATION"
    assert outer["gate_interpretation"]["otherwise"] == "CROSS_MOTIF_GENERALIZATION_REPAIR_NOT_SUPPORTED_IN_SIMULATION"


def test_future_hardware_is_conditioned_on_success_but_not_part_of_step14() -> None:
    cfg = _cfg()
    post = cfg["post_outer_policy"]
    assert post["step12_replay_allowed_only_after_step14_outer_result_is_frozen"] is True
    assert "Step 15 may freeze" in post["if_support_gate_passes"]
    assert "Do not spend Step-15 QPU shots" in post["if_support_gate_fails"]
