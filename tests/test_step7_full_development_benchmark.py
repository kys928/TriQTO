from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from triqto.step7.model import ALL_VARIANTS, Step7DiagnosticModel


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/v0_2"
SCRIPT = SCRIPT_DIR / "run_step7_full_development_benchmark.py"
EXECUTION_CONFIG = ROOT / "configs/v0_2/step7_full_development_benchmark.json"
EXPERIMENT_CONFIG = ROOT / "configs/v0_2/step7_structured_diagnostic_model.json"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("triqto_step7_full_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_execution_contract_binds_exact_pre_smoke_architecture_and_smoke_gate() -> None:
    execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
    assert execution["status"] == "FROZEN_AFTER_SMOKE_BEFORE_FULL_NEURAL_OUTCOME"
    assert execution["experiment_config"]["sha256"] == sha256(EXPERIMENT_CONFIG)
    assert execution["required_smoke"]["smoke_id"] == "smoke_185f69415ea6bb082dd93ef7"
    assert execution["required_smoke"]["decision"] == "STEP7_SMOKE_PASS"
    assert execution["required_smoke"]["scientific_metric_claim"] is False
    assert execution["source_product_id"] == "product_b2d78ad2309b71a55f9bb54f"
    assert execution["runs"]["expected_model_runs"] == 15
    assert execution["scientific_boundaries"]["step8_not_automatically_unlocked"] is True


def test_root_blocks_never_split_a_root_and_keep_frozen_block_size() -> None:
    module = load_module()
    roots = list(range(70))
    blocks = module.root_blocks(roots, 32)
    assert [len(block) for block in blocks] == [32, 32, 6]
    assert [root for block in blocks for root in block] == roots
    assert len({root for block in blocks for root in block}) == len(roots)


def test_checkpoint_selection_is_mechanism_first_then_effect_with_frozen_delta() -> None:
    module = load_module()
    best = {"mechanism_balanced_accuracy": 0.50, "effect_balanced_accuracy": 0.60}
    assert module.checkpoint_is_better(
        {"mechanism_balanced_accuracy": 0.51, "effect_balanced_accuracy": 0.10},
        best,
        0.0005,
    )
    assert module.checkpoint_is_better(
        {"mechanism_balanced_accuracy": 0.5002, "effect_balanced_accuracy": 0.61},
        best,
        0.0005,
    )
    assert not module.checkpoint_is_better(
        {"mechanism_balanced_accuracy": 0.49, "effect_balanced_accuracy": 0.99},
        best,
        0.0005,
    )


def test_stratified_single_class_effect_ba_is_explicitly_undefined() -> None:
    module = load_module()
    outer = module.PredictionSet(
        source_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        root_indices=np.asarray([10, 11, 12, 13], dtype=np.int64),
        effect_truth=np.asarray([0, 0, 1, 1], dtype=np.int8),
        mechanism_truth_all=np.asarray([-1, -1, 0, 1], dtype=np.int8),
        mechanism_mask=np.asarray([False, False, True, True]),
        effect_logits=np.zeros(4, dtype=np.float32),
        mechanism_logits=np.zeros((4, 3), dtype=np.float32),
    )
    rows = [
        {"family": "clean", "n_qubits": "2", "shots": "512", "strength": "0.0", "insertion_depth_bin": "clean"},
        {"family": "clean", "n_qubits": "2", "shots": "1024", "strength": "0.0", "insertion_depth_bin": "clean"},
        {"family": "inj", "n_qubits": "2", "shots": "512", "strength": "0.05", "insertion_depth_bin": "early"},
        {"family": "inj", "n_qubits": "2", "shots": "1024", "strength": "0.05", "insertion_depth_bin": "early"},
    ]
    output = module.stratified_metric_rows(
        name="structured_interaction",
        outer=outer,
        rows=rows,
        effect_pred=np.asarray([0, 0, 1, 1], dtype=np.int8),
        mechanism_pred_all=np.asarray([0, 0, 0, 1], dtype=np.int8),
        strata=["strength"],
        minimum=1,
    )
    clean = [row for row in output if row["task"] == "effect_detection" and row["value"] == "0.0"][0]
    assert clean["balanced_accuracy_defined"] is False
    assert "balanced_accuracy" not in clean


def test_frozen_variants_have_equal_parameter_count_at_full_width() -> None:
    counts = {}
    for variant in sorted(ALL_VARIANTS):
        model = Step7DiagnosticModel(
            variant=variant,
            hidden_dim=64,
            graph_message_passing_layers=3,
            residual_mlp_layers=2,
            dropout=0.1,
            initialization_seed=1701,
        )
        counts[variant] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert set(counts.values()) == {453829}
