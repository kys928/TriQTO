from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from triqto.step7.graph_adapter import batch_from_step5_examples
from triqto.step7_1.model import STEP71_VARIANTS, Step71DiagnosticModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v0_2/step7_1_parity_residual_benchmark.json"
RUNNER = ROOT / "scripts/v0_2/run_step7_1_parity_residual_benchmark.py"


def example(*, parity=(0.10, 0.11, 0.12), effect=True, mechanism=0, mask=True):
    return {
        "x__graph_gate_names": np.asarray(["h", "cx", "rz"]),
        "x__graph_gate_qubit_ptr": np.asarray([0, 1, 3, 4], dtype=np.int32),
        "x__graph_gate_qubit_indices": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "x__graph_gate_parameter_ptr": np.asarray([0, 0, 0, 1], dtype=np.int32),
        "x__graph_gate_parameter_sin": np.asarray([math.sin(0.3)], dtype=np.float64),
        "x__graph_gate_parameter_cos": np.asarray([math.cos(0.3)], dtype=np.float64),
        "x__layout_logical_to_physical": np.asarray([0, 1], dtype=np.int16),
        "x__diagnostic_basis_codes": np.asarray([0, 1, 2], dtype=np.int8),
        "x__delta_local_expectations": np.asarray([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]], dtype=np.float64),
        "x__delta_pairwise_correlations": np.asarray([[0.07], [0.08], [0.09]], dtype=np.float64),
        "x__delta_global_parity": np.asarray(parity, dtype=np.float64),
        "x__pair_indices": np.asarray([[0, 1]], dtype=np.int16),
        "x__observed_shots": np.full(3, 1024, dtype=np.int32),
        "x__reference_shots": np.full(3, 1024, dtype=np.int32),
        "x__reference_available_mask": np.ones(3, dtype=bool),
        "x__reference_kind_code": np.asarray([0], dtype=np.int8),
        "y__effect_present_target": np.asarray([effect], dtype=bool),
        "y__mechanism_target": np.asarray([mechanism], dtype=np.int8),
        "y__mechanism_loss_mask": np.asarray([mask], dtype=bool),
    }


def model(variant: str, seed: int = 1701) -> Step71DiagnosticModel:
    return Step71DiagnosticModel(
        variant=variant,
        hidden_dim=64,
        graph_message_passing_layers=3,
        residual_mlp_layers=2,
        dropout=0.0,
        layer_norm_eps=1e-5,
        initialization_seed=seed,
    )


def load_runner():
    scripts = str(ROOT / "scripts/v0_2")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("triqto_step71_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_contract_is_one_shot_final_revision() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_STEP7_1_OUTCOME"
    assert config["architecture"]["champion"] == "late_concat"
    assert config["architecture"]["candidate"] == "late_concat_parity_residual"
    assert config["architecture"]["local_graph_diagnostic_interaction"] is False
    assert config["architecture"]["pair_graph_diagnostic_interaction"] is False
    assert config["architecture"]["effect_magnitude_shot_path_retained"] is True
    assert config["variants"]["expected_model_runs"] == 7
    assert config["decision"]["confirmation_unlocked_automatically"] is False
    assert "No further Step 7.x" in config["decision"]["stop_rule"]


def test_all_step71_variants_have_equal_frozen_parameter_count() -> None:
    counts = {
        variant: sum(parameter.numel() for parameter in model(variant).parameters() if parameter.requires_grad)
        for variant in STEP71_VARIANTS
    }
    assert set(counts.values()) == {453830}


def test_parity_residual_candidate_starts_exactly_from_late_concat() -> None:
    batch, _ = batch_from_step5_examples([example(), example(mechanism=1)])
    champion = model("late_concat").eval()
    candidate = model("late_concat_parity_residual").eval()
    with torch.no_grad():
        champion_output = champion(batch)
        candidate_output = candidate(batch)
    assert float(candidate.parity_residual_logit) == 0.0
    assert torch.equal(champion_output.representation, candidate_output.representation)
    assert torch.equal(champion_output.effect_logit, candidate_output.effect_logit)
    assert torch.equal(champion_output.mechanism_logits, candidate_output.mechanism_logits)


def test_candidate_parity_residual_gate_receives_mechanism_gradient() -> None:
    batch, targets = batch_from_step5_examples([
        example(mechanism=0),
        example(parity=(-0.18, 0.04, 0.21), mechanism=1),
        example(parity=(0.31, -0.17, 0.08), mechanism=2),
    ])
    candidate = model("late_concat_parity_residual")
    output = candidate(batch)
    loss = F.cross_entropy(output.mechanism_logits, targets.mechanism)
    loss.backward()
    gradient = candidate.parity_residual_logit.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert abs(float(gradient)) > 0.0


def test_no_parity_ablation_removes_parity_from_late_fusion() -> None:
    batch_a, _ = batch_from_step5_examples([example(parity=(0.1, 0.2, 0.3))])
    batch_b, _ = batch_from_step5_examples([example(parity=(-0.9, 0.8, -0.7))])
    no_parity = model("late_concat_no_parity").eval()
    champion = model("late_concat").eval()
    with torch.no_grad():
        no_a = no_parity(batch_a).mechanism_logits
        no_b = no_parity(batch_b).mechanism_logits
        yes_a = champion(batch_a).mechanism_logits
        yes_b = champion(batch_b).mechanism_logits
    assert torch.equal(no_a, no_b)
    assert not torch.equal(yes_a, yes_b)


def test_runner_instantiates_exact_expected_parameter_count() -> None:
    runner = load_runner()
    experiment = json.loads((ROOT / "configs/v0_2/step7_structured_diagnostic_model.json").read_text(encoding="utf-8"))
    for variant in STEP71_VARIANTS:
        instance = runner.instantiate_step71(variant, 1701, experiment, torch.device("cpu"))
        assert sum(parameter.numel() for parameter in instance.parameters() if parameter.requires_grad) == 453830


def test_decision_gate_requires_reproduction_mechanism_ci_and_effect_noninferiority() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    text = config["decision"]["candidate_wins"]
    assert "CI lower bound" in text
    assert "effect BA" in text
    assert "within 0.01" in text
