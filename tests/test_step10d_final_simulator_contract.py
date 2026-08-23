from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "v0_2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_step10d_is_explicit_final_simulator_intervention():
    cfg = _json(ROOT / "configs/v0_2/step10d_final_simulator_lr_refinement.json")
    assert cfg["status"] == "FROZEN_BEFORE_STEP10D_OUTCOME"
    assert cfg["final_simulator_intervention_before_ibm_hardware"] is True
    assert cfg["hard_stop_after_step10d"]["before_ibm_hardware"] is True
    assert cfg["hard_stop_after_step10d"]["no_more_lr_sweeps"] is True
    assert cfg["hard_stop_after_step10d"]["no_more_gradient_clip_tuning"] is True
    assert cfg["hard_stop_after_step10d"]["no_more_loss_weight_tuning"] is True
    assert cfg["hard_stop_after_step10d"]["no_more_simulator_dataset_redesign"] is True
    assert cfg["hard_stop_after_step10d"]["no_more_architecture_changes"] is True
    assert cfg["outer_and_qpu_boundary"]["hardware_stage_proceeds_regardless_of_step10d_improvement"] is True
    assert cfg["outer_and_qpu_boundary"]["qpu_tuning"] is False


def test_step10d_warm_trajectory_changes_only_learning_rate_schedule():
    step10c = _json(ROOT / "configs/v0_2/step10c_crashsafe_long_horizon.json")
    step10d = _json(ROOT / "configs/v0_2/step10d_final_simulator_lr_refinement.json")
    assert step10d["architecture"]["variant"] == step10c["architecture"]["variant"] == "late_concat"
    assert step10d["architecture"]["expected_trainable_parameter_count"] == 453829
    for key in (
        "root_batch_size", "max_epochs", "early_stopping_patience",
        "early_stopping_min_delta", "optimizer", "weight_decay",
        "gradient_clip_norm", "effect_loss_weight", "mechanism_loss_weight",
        "domain_schedule",
    ):
        assert step10d["training"][key] == step10c["training"][key]
    assert step10c["training"]["learning_rate"] == 0.0003
    assert step10d["training"]["learning_rate_schedule"] == [
        {"epoch_start": 1, "epoch_end": 20, "learning_rate": 0.0003},
        {"epoch_start": 21, "epoch_end": 40, "learning_rate": 0.0001},
    ]
    assert step10d["development_candidate"]["initialization"] == "warm_start"
    assert step10d["development_candidate"]["scratch_rerun"] is False
    assert step10d["development_candidate"]["seeds"] == [1701, 1702, 1703]


def test_step10d_lr_schedule_boundary_exact():
    runner = _load("step10d_lr_contract", "run_step10d_final_simulator_lr_refinement.py")
    cfg = _json(ROOT / "configs/v0_2/step10d_final_simulator_lr_refinement.json")
    training = cfg["training"]
    assert runner._lr_for_epoch(training, 1) == pytest.approx(3e-4)
    assert runner._lr_for_epoch(training, 20) == pytest.approx(3e-4)
    assert runner._lr_for_epoch(training, 21) == pytest.approx(1e-4)
    assert runner._lr_for_epoch(training, 40) == pytest.approx(1e-4)
    with pytest.raises(RuntimeError):
        runner._lr_for_epoch(training, 41)


def test_step10d_source_has_no_outer_materialization_or_qpu_path():
    runner = _load("step10d_source_contract", "run_step10d_final_simulator_lr_refinement.py")
    source = inspect.getsource(runner.main)
    assert "fresh_outer_product" not in source
    assert "fresh_original_outer" not in source
    assert "fresh_bridge_outer" not in source
    assert "outer_predictions" not in source
    assert "qiskit_ibm" not in source
    assert "QPU executed: NO" in source
    assert "FURTHER SIMULATOR TUNING BEFORE HARDWARE: FORBIDDEN BY FROZEN PROTOCOL" in source


def test_step10d_hardware_candidate_rule_is_predeclared():
    cfg = _json(ROOT / "configs/v0_2/step10d_final_simulator_lr_refinement.json")
    rule = cfg["hardware_candidate_decision"]
    assert rule["minimum_material_improvement"] == 0.0005
    assert rule["otherwise_primary"] == "frozen Step-10C warm-start ensemble"
    assert rule["human_override_after_qpu_results"] is False
    assert rule["hardware_results_may_not_retroactively_select_simulator_candidate"] is True


def test_step10d_architecture_parameter_count_unchanged():
    step7 = _load("step10d_test_step7", "run_step7_full_development_benchmark.py")
    experiment = _json(ROOT / "configs/v0_2/step7_structured_diagnostic_model.json")
    model = step7.instantiate_model("late_concat", 1701, experiment, torch.device("cpu"))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 453829
